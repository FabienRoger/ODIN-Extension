#!/bin/bash

nns=(densenet10 densenet100)
dss=(Imagenet FSGM Places365)

for nn in ${nns[@]}; do
    for ds in ${dss[@]}; do
        python main.py --nn $nn --out_dataset $ds --magnitude 0.0014 --temperature 1000 --max_images 2048 --batch_size 64
    done
done

# expanded:
# python main.py --nn densenet10 --out_dataset Imagenet --magnitude 0.0014 --temperature 1000 --max_images 2048 --batch_size 64
# python main.py --nn densenet10 --out_dataset FSGM --magnitude 0.0014 --temperature 1000 --max_images 2048 --batch_size 64
# python main.py --nn densenet10 --out_dataset Places365 --magnitude 0.0014 --temperature 1000 --max_images 2048 --batch_size 64
# python main.py --nn densenet100 --out_dataset Imagenet --magnitude 0.0014 --temperature 1000 --max_images 2048 --batch_size 64
# python main.py --nn densenet100 --out_dataset FSGM --magnitude 0.0014 --temperature 1000 --max_images 2048 --batch_size 64
# python main.py --nn densenet100 --out_dataset Places365 --magnitude 0.0014 --temperature 1000 --max_images 2048 --batch_size 64