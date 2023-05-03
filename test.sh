# export CUDA_VISIBLE_DEVICES=$1
# ================================================================================
# Test MFFANet on Helen test dataset provided by DICNet
# ================================================================================

python test.py --gpus 1 --model mffanet --name MFFANet3D --load_size 128 \
     --dataset_name single --dataroot ./test/gamma-1 --pretrain_model_path ./pretrain_models/MFFANet3D.pth \
     --save_as_dir ./result/gamma-1


# ----------------- calculate PSNR/SSIM scores ----------------------------------
python psnr_ssim.py
# ------------------------------------------------------------------------------- 
