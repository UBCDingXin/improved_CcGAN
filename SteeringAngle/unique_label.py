"""
Number of unique labels of steering angle dataset. 
"""
import h5py
import numpy as np 
import matplotlib.pyplot as plt

data_path = 'data/SteeringAngle_64x64_all.h5'

hf = h5py.File(data_path, 'r')
labels = hf['labels'][:]
labels = labels.astype(np.float)
images = hf['images'][:]
hf.close()

print("number of images: %d" %(len(labels)))

labels.sort()

unq_labels = []
unq_labels_num = []

# for the first one, no one to compare with
current_idx = 0 
num = 1 
unq_labels.append(labels[0])
unq_labels_num.append(num)
label_old = labels[current_idx]


for k in range(1, len(labels)):
    label_new = labels[k]
    print("k=%d, label-old=%.3f, label-new=%.3f" %(k, label_old, label_new) )
    if label_new == label_old:
        num += 1
        unq_labels_num[current_idx] = num
        
    else:
        num = 1
        unq_labels.append(label_new)
        unq_labels_num.append(num)
        current_idx += 1
    label_old = label_new
    

print(len(labels), len(unq_labels))
print(min(unq_labels), max(unq_labels))
print(min(unq_labels_num), max(unq_labels_num))

plt.plot(unq_labels, unq_labels_num)
plt.show()

np.savetxt("unq_labels.csv", unq_labels, delimiter=",", fmt="%.3f")
np.savetxt("unq_labels_num.csv", unq_labels_num, delimiter=",", fmt="%d")
 
        
    


