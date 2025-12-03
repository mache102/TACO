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

### coco 5k (eval)

```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
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
```bash
python -m venv env
source env/bin/activate

pip install -r requirements.txt
pip install compressai

```
## running (mscoco)

Run:

```bash
python -u generate_images_mscoco30k.py --image_folder_root /home/ying/datasets/MSCOCO/val2014 --checkpoint ckpts/lambda_0.0004.pth.tar --out lambda_0.0004
```
This may take **~5h** for the full 30k images.

## running (kodak)

```bash
python -u generate_images_using_image_cap_dataset.py --image_folder_root /home/ying/datasets/kodak/dataset/ --image_cap_dict_root ./materials/kodak_ofa.json --checkpoint ./ckpts/lambda_0.0004.pth.tar
```

In `generate_images_using_image_cap_dataset.py`, add `.to(device)` to `x`.

We also added an `--out` argument to specify output directory.


## running (coco 5k)

```bash
python -u generate_images_mscoco30k.py --image_folder_root /home/ying/datasets/val2017 --checkpoint ckpts/lambda_0.0004.pth.tar --out lambda_0.0004 --use_coco5k
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
