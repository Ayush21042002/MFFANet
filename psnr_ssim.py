import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import os
import cv2
import rarfile

def rgb2y_matlab(x):
    """Convert RGB image to illumination Y in Ycbcr space in matlab way.
    -------------
    # Args
        - Input: x, byte RGB image, value range [0, 255]
        - Ouput: byte gray image, value range [16, 235] 

    # Shape
        - Input: (H, W, C)
        - Output: (H, W) 
    """
    K = np.array([65.481, 128.553, 24.966]) / 255.0
    Y = 16 + np.matmul(x, K)
    return Y.astype(np.uint8)


def PSNR(im1, im2, use_y_channel=True):
    """Calculate PSNR score between im1 and im2
    --------------
    # Args
        - im1, im2: input byte RGB image, value range [0, 255]
        - use_y_channel: if convert im1 and im2 to illumination channel first
    """
    if use_y_channel:
        im1 = rgb2y_matlab(im1)
        im2 = rgb2y_matlab(im2)
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    mse = np.mean(np.square(im1 - im2)) 
    return 10 * np.log10(255**2 / mse) 


def SSIM(gt_img, noise_img):
    """Calculate SSIM score between im1 and im2 in Y space
    -------------
    # Args
        - gt_img: ground truth image, byte RGB image
        - noise_img: image with noise, byte RGB image
    """
    gt_img = rgb2y_matlab(gt_img)
    noise_img = rgb2y_matlab(noise_img)
     
    ssim_score = compare_ssim(gt_img, noise_img, gaussian_weights=True, 
            sigma=1.5, use_sample_covariance=False)
    return ssim_score

def psnr_ssim_dir(gt_dir, test_dir):
    # gt_img_list = sorted([x for x in sorted(os.listdir(gt_dir))])
    gt_img_list = os.listdir(gt_dir)
    # test_img_list = sorted([x for x in sorted(os.listdir(test_dir))])
    #  assert gt_img_list == test_img_list, 'Test image names are different from gt images.' 
    imgrar_HR = rarfile.RarFile(test_dir)
    # print(imgrar_HR.infolist()[0].filename)
    # imgrar_LR = rarfile.RarFile(gt_dir)
    psnr_score = 0
    ssim_score = 0
    for name in gt_img_list:
        gt_img = Image.open(os.path.join(gt_dir,name))
        test_img = imgrar_HR.open(imgrar_HR.getinfo('HR/' + name))
        test_img = Image.open(test_img)
        gt_img = np.array(gt_img)
        test_img = np.array(test_img)
        test_img = cv2.resize(test_img,(128,128))
        psnr_score += PSNR(gt_img, test_img)
        ssim_score += SSIM(gt_img, test_img)
    # for gt_name, test_name in zip(gt_img_list, test_img_list):
    #     gt_img = Image.open(os.path.join(gt_dir, gt_name))
    #     test_img = Image.open(os.path.join(test_dir, test_name))
    #     gt_img = np.array(gt_img)
    #     test_img = np.array(test_img)
    #     test_img = cv2.resize(test_img,(128,128))
    #     psnr_score += PSNR(gt_img, test_img)
    #     ssim_score += SSIM(gt_img, test_img)
    return psnr_score / len(gt_img_list), ssim_score / len(gt_img_list)

if __name__ == '__main__':
    # print(os.curdir)
    gt_dir = '/content/drive/MyDrive/BTP_Major/test_hr/HR.rar'
    test_dirs = [
            '/content/drive/MyDrive/BTP_Major/CelebA_Results/SPARNet/gamma-1',
            '/content/drive/MyDrive/BTP_Major/CelebA_Results/SPARNet/gamma-1.5',
            '/content/drive/MyDrive/BTP_Major/CelebA_Results/SPARNet/gamma-2',
            ]
    for td in test_dirs:
        result = psnr_ssim_dir(td, gt_dir)
        print(td, result)



