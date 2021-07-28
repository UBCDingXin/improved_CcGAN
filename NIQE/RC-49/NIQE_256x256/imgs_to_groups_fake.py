import glob
import os
import shutil
import argparse
from tqdm import tqdm
import platform

parser = argparse.ArgumentParser(description='separate fake angles according to angles.')
parser.add_argument('--imgs_dir', type=str, default='', help='imgs dir.')
parser.add_argument('--out_dir_base', type=str, default='', help='output dir.')
args = parser.parse_args()

if platform.system().lower()=="linux":
    split_symbol = '/'
elif platform.system().lower()=="windows":
    split_symbol = '\\'
else:
    raise ValueError('Do not support!!!')

def get_file_list(dataset_dir):
    file_list = glob.glob(os.path.join(dataset_dir, '*.png')) + glob.glob(os.path.join(dataset_dir, '*.jpg'))
    file_list.sort()
    return file_list


### image directory
imgs_dir = args.imgs_dir
output_dir_base = os.path.join(args.out_dir_base, 'fake_images_by_angles')
print(output_dir_base)
os.makedirs(output_dir_base, exist_ok=True)

img_lists = get_file_list(imgs_dir)

for img_idx, img_path in tqdm(enumerate(img_lists)):
    img_name = img_path.split(split_symbol)[-1]
    angle_tmp = img_name.split('.png')[0]
    angle = round(float(angle_tmp.split('_')[-1]), ndigits=1)

    output_dir = output_dir_base + split_symbol + str(angle) + split_symbol
    os.makedirs(output_dir, exist_ok=True)

    shutil.copy(img_path, output_dir)
