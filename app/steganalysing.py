import os, sys

from numpy import asarray, uint8, expand_dims
from PIL import Image
import matplotlib.pyplot as plt


def visual_attack(stego_file_path: str, result_file_path: str):

    with Image.open(stego_file_path) as F:
        stego_object = asarray(F, dtype=uint8)

    if len(stego_object.shape) < 3:
        stego_object = expand_dims(stego_object, axis=2)

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5)
    ax = fig.subplots(1, stego_object.shape[2])

    for i in range(stego_object.shape[2]):
        color_plane = stego_object[:, :, i]
        LSB_plane = color_plane % 2
        axis = ax if stego_object.shape[2] == 1 else ax[i]
        axis.set_title(f'plane {i}')
        axis.imshow(LSB_plane, cmap='gray')

    if result_file_path:
        result_file_path = result_file_path + '.png'
        plt.savefig(result_file_path, bbox_inches='tight', dpi=1200)
    else:
        plt.show()


if __name__ == '__main__':
    sys.exit()
