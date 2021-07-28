import glob
import os
import shutil
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import platform

parser = argparse.ArgumentParser()
parser.add_argument('--imgs_dir', type=str, default='', help='imgs dir.')
parser.add_argument('--center_file', type=str, default='', help='file to define eval centers.')
parser.add_argument('--out_dir_base', type=str, default='./real_data', help='output dir.')
args = parser.parse_args()

if platform.system().lower()=="linux":
    split_symbol = '/'
elif platform.system().lower()=="windows":
    split_symbol = '\\'
else:
    raise ValueError('Do not support!!!')

NC=3
IMG_SIZE=128
RADIUS=2 # the unit is degree

def get_file_list(dataset_dir):
    file_list = glob.glob(os.path.join(dataset_dir, '*.png')) + glob.glob(os.path.join(dataset_dir, '*.jpg'))
    file_list.sort()
    return file_list


### image directory
imgs_dir = args.imgs_dir
output_dir_base = os.path.join(args.out_dir_base, 'centers')
print(output_dir_base)
os.makedirs(output_dir_base, exist_ok=True)


### load all images and labels first
img_lists = get_file_list(imgs_dir)
n_img = len(img_lists)
images = np.zeros((n_img, IMG_SIZE, IMG_SIZE, NC))
labels = np.zeros(n_img)

for i in range(n_img):
    fullpath_i = img_lists[i]
    filename_i = fullpath_i.split(split_symbol)[-1]
    label_i = float((filename_i.split('.png')[0]).split('_')[-1])
    labels[i] = label_i
    images[i] = cv2.imread(fullpath_i)
#end for i

### load centers
centers = np.loadtxt(args.center_file)
num_centers = len(centers)

for i in range(num_centers):
    center_i = centers[i]
    lb_i = center_i - RADIUS
    ub_i = center_i + RADIUS
    indx_i = np.where((labels>=lb_i)*(labels<=ub_i)==True)[0]
    images_indx_i = images[indx_i]
    labels_indx_i = labels[indx_i]

    print('\r {}/{}, center={}, lb={}, ub={}, num_img={}.\n'.format(i, num_centers, center_i, lb_i, ub_i, len(indx_i)))

    output_dir = os.path.join(output_dir_base, str(i+1)) #the center ID is from 1 to num_centers
    os.makedirs(output_dir, exist_ok=True)

    for j in range(len(indx_i)):
        image_j_indx_i = images_indx_i[j]
        filename_i = os.path.join(output_dir, '{}_{}.png'.format(j, labels_indx_i[j]))
        cv2.imwrite(filename_i, image_j_indx_i)
