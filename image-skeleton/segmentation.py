from glob import glob
import cv2
import re
import matplotlib.pyplot as plt
import numpy as np
import os
from skan.csr import make_degree_image


def load(dir_path, fmt, is_color=True):
    files = glob(dir_path + '/*.' + fmt)
    res = {}
    if is_color:
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            num = re.findall(rf'[\d]+(?=\.{fmt})', file)[0]
            res[int(num)] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            num = re.findall(rf'[\d]+(?=\.{fmt})', file)[0]
            res[int(num)] = img
    return res

def imshow(img_dict):
    n = len(img_dict.keys())
    fig, ax = plt.subplots(1, n, figsize=(n*3, 4))
    for i, key in enumerate(img_dict.keys()):
        ax[i].imshow(img_dict[key])
        ax[i].axis('off')
        ax[i].set_title(key)
    plt.show()

def rem_shadows(img):
    """Taken from https://stackoverflow.com/a/44752405"""
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    
    for plane in rgb_planes:
        # dilate
        dilated_img = cv2.dilate(plane, kernel=np.ones((7,7), np.uint8))
        
        # median blur
        bg_img = cv2.medianBlur(dilated_img, ksize=21)

        # invert
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        
        # normalize
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def binarize(img_dict, do_rem_shadows=True):
    binarized = {}
    thresh = {}
    kernel_blu = (3, 3)
    kernel_dil = np.ones((20, 20), np.uint8)
    kernel_ero = np.ones((30, 30), np.uint8)
    for key, img in img_dict.items():
        if do_rem_shadows:
            # remove shadows and normalize
            img = rem_shadows(img)

        # blurring
        blurred = cv2.GaussianBlur(img, kernel_blu, sigmaX=0.1)

        # otsu
        img_gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        th, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)
        thresh[key] = th

        # closing
        inverted = cv2.bitwise_not(img_bin)
        dilated = cv2.dilate(inverted, kernel_dil, iterations=1)
        eroded = cv2.erode(dilated, kernel_ero, iterations=1)

        binarized[key] = eroded
        
    return binarized

def save(img_dict, dir_path):
    os.mkdir(dir_path)
    res = True
    for key, img in img_dict.items():
        res &= cv2.imwrite(f'{dir_path}/{key}.bmp', img)
    return res

def find_circles(img_dict):
    res = {}
    for key, img in img_dict.items():
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.blur(gray, (2, 2))

        res[key] = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            1, 20, param1 = 50, param2 = 30,
            minRadius = 1, maxRadius = 40
        )
    return res

def should_draw(img, a, b, r):
    degree_image = make_degree_image(img)
    mask = np.zeros_like(degree_image).astype(float)
    cv2.circle(mask, (a, b), r, color=1, thickness=-1)
    return np.max(degree_image * mask) < 3

def add_notches(skeletonized, detected_circles_dict):
    res = {}
    for key, detected_circles in detected_circles_dict.items():
        res[key] = skeletonized[key].copy().astype(float)
        if detected_circles is None:
            continue

        detected_circles = np.uint16(np.around(detected_circles))
        for a, b, r in detected_circles[0, :]:
            if not should_draw(res[key], a, b, r):
                continue
            angle = round(np.random.uniform(low=0, high=180))
            axes=(round(1.25*r), 1)
            cv2.ellipse(res[key], center=(a,b), axes=axes, angle=angle, startAngle=0, endAngle=180, color=1, thickness=1)
    return res 