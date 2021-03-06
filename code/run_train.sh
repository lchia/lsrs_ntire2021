CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train.py \
    -opt ./confs/SRFlow_DF2K_4X.yml \
    --launcher pytorch
