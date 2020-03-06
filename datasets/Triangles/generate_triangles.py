#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:45:00 2020
create random triangle dataset
@author: yongwei
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy import ndimage


def gen_triangle(theta, img_size=28, border_space=8):
    #generate triangles with vertices A,B,C. 
    theta_radians = theta * math.pi / 180 
    x_min, y_min = border_space + 1, border_space + 1
    x_max, y_max = img_size - border_space, img_size - border_space 
    A_x, A_y = random.randint(x_min, x_max), random.randint(y_min, y_max) #5,5

    B_x = random.randint(A_x + border_space, img_size)
    B_y = A_y

    deltaX_AC = random.randint(3,img_size - A_x-1)  #img_size - A_x, 2, 23
    C_x = A_x + deltaX_AC
    C_y_tmp = A_y + int(round(deltaX_AC * math.tan(theta_radians)))
    C_y = min(img_size, C_y_tmp)

    vertices = np.matrix([[A_x,A_y],[B_x,B_y],[C_x,C_y]])


 # Create the bitmask by dividing each cell into subpixels.
    r, dim = border_space, img_size #r=8, dim=28
    poly_as_path = path.Path(vertices * r) 
    grid_x, grid_y = np.mgrid[0:dim * r, 0:dim * r] #(224,224)
    flattened_grid = np.column_stack((grid_x.ravel(), grid_y.ravel())) #(50176,2)

    mask = poly_as_path.contains_points(flattened_grid).reshape(dim * r, dim * r)
    mask = np.array(~mask, dtype=np.float32)

    mask = filters.convolve(mask, np.ones((r, r)), mode="constant", cval=1.0)   
    mask = mask[::r, ::r] / (r * r)
    mask = np.rot90(mask)
    #rot_degree = random.randint(0,180)
    #rot_degree = 0
    #mask = ndimage.rotate(mask, rot_degree, reshape=False,mode='constant', cval=1.0)
    return mask


def GenerateDataset(n_instances, theta=30, raster_dim=28, border_space=8):

  x = np.zeros((n_instances, raster_dim, raster_dim))
  
  for i in range(n_instances):
    x[i] = gen_triangle(theta=theta, img_size=raster_dim, border_space=border_space)
  ids = np.random.permutation(x.shape[0])
  return x[ids]

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
 
def visualize(dataset, n_rows):
  fig = plt.figure(1, (n_rows, n_rows))
  n = dataset.shape[0]
  n_cols = n//n_rows + (n % n_rows > 0)
  grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=(0, 0))
  for i in range(n):
    grid[i].set_xlim([0, dataset[0].shape[0]])
    grid[i].set_ylim([0, dataset[0].shape[0]])
    grid[i].get_xaxis().set_ticks([])
    grid[i].get_yaxis().set_ticks([])
  for row in range(n_rows):
    for col in range(n_cols):
      target = dataset[row * n_cols + col]
      grid[row * n_cols + col].imshow(target, cmap='gray')
      

triangles = GenerateDataset(
    n_instances=10, 
    theta=50,
    raster_dim=28,
    border_space=8) 
#visualize(triangles, n_rows=10)     

sample = triangles[2,:,:]
plt.imshow((sample), cmap='gray')
plt.show()


