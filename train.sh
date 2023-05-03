# export CUDA_VISIBLE_DEVICES=$1
# =================================================================================
# Train MFFANet
# =================================================================================

!python /content/drive/MyDrive/BTP_Major/Our_Method/train.py --gpus 1 --name MFFANet3D --model mffanet \
    --res_depth 1 --att_name mffa3d \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 --total_epochs 20 \
    --dataroot /content/drive/MyDrive/BTP_Major/img_align_celeba.rar --dataset_name celeba --batch_size 32 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 100 \
    --checkpoints_dir /content/drive/MyDrive/BTP_Major/check_points \
    --save_iter_freq 5000 \
    --save_epoch_freq 1 \
    --continue_train


