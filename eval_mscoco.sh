#!/bin/bash
declare -a lambdas=("0.0004" "0.0008" "0.0016" "0.004" "0.009" "0.015")

for lambda in "${lambdas[@]}"; do
    python -u generate_images_mscoco30k.py --image_folder_root /home/ying/datasets/MSCOCO/val2014 --checkpoint ./ckpts/lambda_${lambda}.pth.tar --out lambda_${lambda}
done