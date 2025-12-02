#!/bin/bash
declare -a lambdas=("0.0004" "0.0008" "0.0016" "0.004" "0.009" "0.015")

for lambda in "${lambdas[@]}"; do
    python -u generate_images_using_image_cap_dataset.py --image_folder_root /home/ying/datasets/kodak/dataset/ --image_cap_dict_root ./materials/kodak_ofa.json --checkpoint ./ckpts/lambda_${lambda}.pth.tar --out lambda_${lambda}
done