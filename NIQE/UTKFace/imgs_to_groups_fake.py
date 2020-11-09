import glob
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='separate fake ages according to ages.')
parser.add_argument('--imgs_dir', type=str, default='', help='imgs dir.')
args = parser.parse_args()

def get_file_list(dataset_dir):
    file_list = glob.glob(os.path.join(dataset_dir, '*.png')) + glob.glob(os.path.join(dataset_dir, '*.jpg'))
    file_list.sort()
    return file_list


### image directory
imgs_dir = args.imgs_dir
output_dir_base = imgs_dir.split('/')[0] + '/' + imgs_dir.split('/')[1] + '/ages/'
print(output_dir_base)
os.makedirs(output_dir_base, exist_ok=True)

img_lists = get_file_list(imgs_dir)

for img_idx, img_path in enumerate(img_lists):
    img_name = img_path.split('/')[-1]
    age_tmp = img_name.split('.png')[0]
    age = age_tmp.split('_')[-1]

    output_dir = output_dir_base + str(age) + '/'
    os.makedirs(output_dir, exist_ok=True)

    shutil.copy(img_path, output_dir)
