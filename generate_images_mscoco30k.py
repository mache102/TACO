import os, sys
import torch, torchvision
import torch.nn.functional as F

import collections

from transformers import CLIPTextModel, AutoTokenizer
import lpips
import time 
from PIL import Image

from models import TACO
from config.config import model_config 
from utils.utils import *

from pytorch_msssim import ms_ssim as ms_ssim_func

import json
import shutil
import math
import pandas as pd

from tqdm import tqdm

# ============
device = 'cuda'

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_alex = loss_fn_alex.to(device)
loss_fn_alex.requires_grad_(False)

# ============

def parse_args_for_inference(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    
    parser.add_argument(
        "--image_folder_root", type=str, default='/data/MSCOCO/val2014', help="image folder path"
    )

    parser.add_argument(
        "--checkpoint", type=str, default='./checkpoint/0.0004.pth.tar', help="path of the pretrained checkpoint"
    )
    parser.add_argument("--out", type=str, required=True, help="output directory")

    parser.add_argument(
        "--num_images", type=int, default=5000, help="Number of images to evaluate (subsampled from 30k, default 5000)"
    )
    parser.add_argument(
        "--use_coco5k", action='store_true', help="Use COCO 5k val2017 images and captions (from annotations/captions_val2017.json and instances_val2017.json)"
    )
    parser.add_argument(
        "--use_coco50", action='store_true', help="Use COCO 50 images and captions (for quick testing)"
    )
    args = parser.parse_args(argv)
    return args

def main(argv):

    args = parse_args_for_inference(argv)

    clip_model_name = "openai/clip-vit-base-patch32"

    CLIP_text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
    CLIP_text_model.requires_grad_(False)
    CLIP_tokenizer = AutoTokenizer.from_pretrained(clip_model_name)

    if args.use_coco50:
        # Use COCO 50 images and captions
        coco50_path = './materials/coco_50_imgs.txt'
        with open(coco50_path, 'r') as f:
            coco50_imgs = set([line.strip() for line in f if line.strip()])
        # Use COCO 5k val2017 images and captions
        ann_dir = '/home/ying/datasets/annotations'
        captions_path = os.path.join(ann_dir, 'captions_val2017.json')
        with open(captions_path, 'r') as f:
            coco_caps = json.load(f)
        imgid_to_filename = {img['id']: img['file_name'] for img in coco_caps['images']}
        image_cap_dict = collections.defaultdict(list)
        for ann in coco_caps['annotations']:
            fname = imgid_to_filename[ann['image_id']]
            if fname in coco50_imgs:
                image_cap_dict[fname].append(ann['caption'])
        image_list = sorted(list(coco50_imgs & set(image_cap_dict.keys())))
        # No subsampling for coco50
    else:
        with open('./materials/mscoco_30k_list.json', 'r') as f:
            image_list = json.load(f)
        image_list.sort()
        # Subsample image_list if requested
        N = args.num_images
        total = 30000
        step = max(1, math.floor(total / N))
        if N < len(image_list):
            image_list = image_list[::step]
        with open('./materials/mscoco_val41k_img_cap_pair.json', 'r') as f:
            image_cap_dict = json.load(f)

    params_name = f'{args.checkpoint}'

    taco_config = model_config()
    net = TACO(taco_config, text_embedding_dim = CLIP_text_model.config.hidden_size)
    net = net.to(device)

    print(f"checkpoint: {params_name}")
    
    params_path = f'{params_name}' 

    state_dict = torch.load(params_path, map_location = device)['state_dict']
    
    try:
        try:
            net.load_state_dict(state_dict)
        except:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            net.load_state_dict(new_state_dict)
    except:
        try:
            net.module.load_state_dict(state_dict)
        except:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            net.module.load_state_dict(new_state_dict)

    del state_dict

    net.requires_grad_(False)
    net.update()
    
    stat_csv = {
        'image_name': [],
        'bpp': [],
        'psnr': [],
        'ms_ssim': [],
        'lpips': []
        }

    # save_folder = f'./compression_mscoco_val30k'
    if args.use_coco50:
        save_folder = os.path.join("out", "coco50", args.out)
    else:
        save_folder = os.path.join("out", "mscoco_val30k", args.out)
    # If output dir exists and is not empty, clear it
    if os.path.exists(save_folder):
        if os.listdir(save_folder):
            shutil.rmtree(save_folder)
    os.makedirs(f"{save_folder}", exist_ok=True)
    os.makedirs(f"{save_folder}/figures", exist_ok=True)
    os.makedirs(f"{save_folder}/temp", exist_ok=True)

    mean_csv = {
        'bpp':0.0,
        'psnr': 0.0,
        'ms_ssim': 0.0,
        'lpips':0.0
    }

    # Inference timing storage
    inference_times = []
    avg_times = {'clip_tokenize': 0.0, 'compress': 0.0, 'decompress': 0.0, 'total': 0.0}

    for img_name in tqdm(image_list, desc=f"compress:"):
        img_path = f'{args.image_folder_root}/{img_name}'
        img = torchvision.transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
        x = img.unsqueeze(0).to(device)

        _, _, H, W = x.shape
        pad_h = 0
        pad_w = 0
        if H % 64 != 0:
            pad_h = 64 * (H // 64 + 1) - H
        if W % 64 != 0:
            pad_w = 64 * (W // 64 + 1) - W

        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

        pred_image_list = []
        pred_bpp_list = []

        # For COCO5k, image_cap_dict is a defaultdict(list)
        image_times = []
        
        for caption in image_cap_dict[img_name]:
            t0 = time.perf_counter()
            clip_token = CLIP_tokenizer([caption], padding="max_length", max_length=38, truncation=True, return_tensors="pt").to(device)
            t1 = time.perf_counter()
            text_embeddings = CLIP_text_model(**clip_token).last_hidden_state
            t2 = time.perf_counter ()
            out_enc = net.compress(x_padded, text_embeddings)
            t3 = time.perf_counter()
            shape = out_enc["shape"]

            output = os.path.join(f'{save_folder}/temp', f'{img_name}')
            with Path(output).open("wb") as f:
                write_uints(f, (H, W))
                write_body(f, shape, out_enc["strings"])

            size = filesize(output)
            bpp = float(size) * 8 / (H * W)

            with Path(output).open("rb") as f:
                original_size = read_uints(f, 2)
                strings, shape = read_body(f)

            t4 = time.perf_counter()
            out = net.decompress(strings, shape, text_embeddings)
            t5 = time.perf_counter()
            x_hat = out["x_hat"]
            x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]

            pred_image_list.append(x_hat.detach().clone())
            pred_bpp_list.append(bpp)

            # Save times for this caption
            image_times.append({
                'clip_tokenize': t1-t0,
                'compress': t3-t2,
                'decompress': t5-t4,
                'total': (t1-t0)+(t3-t2)+(t5-t4)
            })
        # Find the avg inference time for this img 
        inference_times.append({
            'image_name': img_name,
            'clip_tokenize': sum([t['clip_tokenize'] for t in image_times]) / len(image_times),
            'compress': sum([t['compress'] for t in image_times]) / len(image_times),
            'decompress': sum([t['decompress'] for t in image_times]) / len(image_times),
            'total': sum([t['total'] for t in image_times]) / len(image_times)
        })
            
        best_image = {
            'BPP':0.0,
            'PSNR': 0.0,
            'MS-SSIM': 0.0,
            'LPIPS':1e6, # this was originally 0, which is incorrect for lpips minimization
            'caption': "",
            'image': None
        }

        for i, pred_image in enumerate(pred_image_list):
            bpp = pred_bpp_list[i]

            psnr = compute_psnr(x, pred_image)
            try:
                ms_ssim = ms_ssim_func(x, pred_image, data_range=1.).item()
            except:
                ms_ssim = ms_ssim_func(torchvision.transforms.Resize(256)(x), torchvision.transforms.Resize(256)(pred_image), data_range=1.).item()

            lpips_score = loss_fn_alex(x, pred_image).item()
            # print(f" score (lpips): {lpips_score} ")

            # we choose the lowest lpips image as best
            if best_image['LPIPS'] > lpips_score:
                best_image['BPP']=bpp
                best_image['PSNR']=psnr
                best_image['MS-SSIM']=ms_ssim
                best_image['LPIPS']=lpips_score
                best_image['caption'] = image_cap_dict[img_name][i]
                best_image['image'] = pred_image.detach().clone()

        final_caption = best_image['caption']
        print(f'Checkpoint: {params_name}, img_name: {img_name}, Caption: {final_caption}')

        torchvision.utils.save_image(best_image['image'], f'{save_folder}/figures/{img_name}', nrow=1)
        mean_csv['bpp'] += bpp
        mean_csv['psnr'] += psnr
        mean_csv['ms_ssim'] += ms_ssim
        mean_csv['lpips'] += lpips_score

        stat_csv['image_name'].append(img_name)
        stat_csv['bpp'].append(bpp)
        stat_csv['psnr'].append(psnr)
        stat_csv['ms_ssim'].append(ms_ssim)
        stat_csv['lpips'].append(lpips_score)

    shutil.rmtree(f"{save_folder}/temp")

    data_lpips_best = pd.DataFrame(stat_csv)
    data_lpips_best.to_csv(f'{save_folder}/stat_per_image.csv')

    mean_csv['bpp'] /= len(image_list)
    mean_csv['psnr'] /= len(image_list)
    mean_csv['ms_ssim'] /= len(image_list)
    mean_csv['lpips'] /= len(image_list)

    print(f"checkpoint: {params_name}")

    print(f"\nBPP: {mean_csv['bpp']}, PSNR: {mean_csv['psnr']}, MS-SSIM: {mean_csv['ms_ssim']}, LPIPS: {mean_csv['lpips']}\n")

    with open(f'{save_folder}/mean_stat.json', 'w') as f:
        json.dump(mean_csv, f, indent=4)

    # Save inference times
    df_times = pd.DataFrame(inference_times)
    df_times.to_csv(f'{save_folder}/inference_times.csv', index=False)
    avg_times['clip_tokenize'] = float(df_times['clip_tokenize'].mean())
    avg_times['compress'] = float(df_times['compress'].mean())
    avg_times['decompress'] = float(df_times['decompress'].mean())
    avg_times['total'] = float(df_times['total'].mean())
    with open(f'{save_folder}/avg_inference_times.json', 'w') as f:
        json.dump(avg_times, f, indent=4)