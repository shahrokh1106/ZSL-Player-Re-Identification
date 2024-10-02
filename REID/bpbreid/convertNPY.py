import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import glob
import os


for filename in glob.glob(os.path.join("npy", "*")):
    if '.npy' in filename:
        img_array = np.load(filename, allow_pickle=True)
        plt.imshow(img_array, cmap="gray")
        img_name = filename+".png"
        matplotlib.image.imsave(img_name, img_array)
        print(filename)