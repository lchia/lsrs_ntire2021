CUDA_VISIBLE_DEVICES=0,1 \
python train.py \
    -opt ./confs/SRFlow_DF2K_4X_patch256.yml \
    --launcher pytorch
