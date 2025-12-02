clone the repo https://github.com/effl-lab/TACO?tab=readme-ov-file

## downloads
Download ckpts with google drive downloader (linux cli):

```bash
mkdir ckpts
cd ckpts
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1zOLRtr3AMJ2KChCDjCeBicI871Pzw28K
```
You will get ckpts for lambdas 0.0004, 0.0008, 0.0016, 0.004, 0.009, 0.0015

### mscoco (eval)
Download MSCOCO val 2014:
```bash
cd 
cd datasets
wget http://images.cocodataset.org/zips/val2014.zip 
mkdir MSCOCO
unzip val2014.zip -d MSCOCO
```

### kodak (eval)
https://r0k.us/graphics/kodak/

```
cd datasets
mkdir kodak
cd kodak
wget https://r0k.us/graphics/kodak/kodak.zip
unzip kodak.zip

results will be in ~/datasets/kodak/dataset/ (kodim01.png ... kodim24.png)
```

## deps

```
# Core ML frameworks 

# the first ver of the code was released in may 2024, around the same time as pytorch 2.3.0. this means it's unlikely to have used features from pytorch 2.3.0 or later.
torch==2.6.0
torchvision
transformers>=4.20.0  
  
# Image processing and metrics  
Pillow>=9.0.0  
lpips>=0.1.4  
pytorch-msssim>=1.0.0  
  
# Data handling  
pandas>=1.4.0  
numpy>=1.21.0  
tqdm>=4.64.0  
compressai
```
```bash
python -m venv env
source env/bin/activate

pip install -r requirements.txt
pip install compressai

```
## edits

go to utils/utils.py 
in this line:

import lpips, clip
remove clip, since it's not used at all




## running (mscoco)

Run:

```bash
# python -u generate_images_using_image_cap_dataset.py --image_folder_root (path for original image data) --checkpoint ckpts/lambda_0.0004.pth.tar
# e.g. python -u generate_mscoco30k.py --image_folder_root /data/MSCOCO/val2014 --checkpoint /checkpoint/pre_trained_ckpt.pth.tar

python -u generate_images_mscoco30k.py --image_folder_root /datasets/MSCOCO/val2014 --checkpoint ckpts/lambda_0.0004.pth.tar
```


began running at 17:36. 

ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.

but now we get this:
Traceback (most recent call last):
  File "/home/ying/experiments/TACO/generate_images_mscoco30k.py", line 237, in <module>
    main(sys.argv[1:])
  File "/home/ying/experiments/TACO/generate_images_mscoco30k.py", line 133, in main
    img = torchvision.transforms.ToTensor()(Image.open(img_path).convert('RGB')).to(device)
                                            ^^^^^^^^^^^^^^^^^^^^
  File "/home/ying/experiments/TACO/env/lib/python3.12/site-packages/PIL/Image.py", line 3493, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/datasets/MSCOCO/val2014/COCO_val2014_000000000073.jpg'

## running (kodak)

```bash
python -u generate_images_using_image_cap_dataset.py --image_folder_root /home/ying/datasets/kodak/dataset/ --image_cap_dict_root ./materials/kodak_ofa.json --checkpoint ./ckpts/lambda_0.0004.pth.tar
```

In `generate_images_using_image_cap_dataset.py`, add `.to(device)` to `x`.

We also added an `--out` argument to specify output directory.


Also this script for easily iterating over all lambda ckpts:

```bash
#!/bin/bash
declare -a lambdas=("0.0004" "0.0008" "0.0016" "0.004" "0.009" "0.015")

for lambda in "${lambdas[@]}"; do
    python -u generate_images_using_image_cap_dataset.py --image_folder_root /home/ying/datasets/kodak/dataset/ --image_cap_dict_root ./materials/kodak_ofa.json --checkpoint ./ckpts/lambda_${lambda}.pth.tar --out kodak/lambda_${lambda}
done
```


## other
if you cloned the original repo, you can redirect its origin to your own fork:

```bash
git remote set-url origin https://github.com/mache102/TACO
```
Then push to your own fork:

```bash
git add .
git commit -m "your commit message"
git push origin main
```


if git creds are needed:

```bash
# list releases
curl -s https://api.github.com/repos/git-ecosystem/git-credential-manager/releases/latest | grep browser_download_url

#pick a .deb for linux
wget https://github.com/git-ecosystem/git-credential-manager/releases/download/v2.6.1/gcm-linux_amd64.2.6.1.deb

sudo dpkg -i gcm-linux_amd64.2.6.1.deb

git config --global credential.helper manager


gpg --full-generate-key
# pick rsa + rsa
# key size: 4096
# expiration: 0
# ...

gpg --list-secret-keys --keyid-format=long
# copy the long key id after sec rsa4096/

sudo apt install pass
pass init ABCDEF1234567890

git config --global credential.credentialStore gpg

git-credential-manager-core configure

```