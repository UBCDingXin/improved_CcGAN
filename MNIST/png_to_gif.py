import os
from PIL import Image, ImageDraw, ImageSequence
import io

GAN = 'cDCGAN'

path = '/home/xin/OneDrive/Working_directory/Continuous_cGAN/MNIST/Output/saved_images/' + GAN + '_InTrain'

gif_filename = path + '/' + GAN + '.gif'

png_filenames = os.listdir(path)
steps_sorted = []
for filename in png_filenames:
    step = filename.split('.')[0]
    if step.isnumeric():
        steps_sorted.append(int(step))
steps_sorted.sort()


images = []

for i in range(len(steps_sorted)):
    filename = path + '/' + str(steps_sorted[i]) + '.png'
    im = Image.open(filename)
    
    d = ImageDraw.Draw(im)
    d.text((10,10), str(steps_sorted[i]), fill=(255, 0, 0))
    del d
    
    images.append(im)

images[0].save(gif_filename, save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)


