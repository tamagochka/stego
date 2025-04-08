import os

from numpy import asarray, uint8
from PIL import Image
import matplotlib.pyplot as plt

from config import AppConfig


def visual_attack(stego_file_path: str, result_file_path: str):
    with Image.open(stego_file_path) as F:
        stego_object = asarray(F, dtype=uint8)

    stego_red = stego_object[:, :, 0]
    stego_green = stego_object[:, :, 1]
    stego_blue = stego_object[:, :, 2]

    LSB_plane_red = stego_red % 2
    LSB_plane_green = stego_green % 2
    LSB_plane_blue = stego_blue % 2

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.5)
    ax = fig.subplots(1, 3)
    ax[0].set_title('red plane')
    ax[0].imshow(LSB_plane_red, cmap='gray')
    ax[1].set_title('green plane')
    ax[1].imshow(LSB_plane_green, cmap='gray')
    ax[2].set_title('blue plane')
    ax[2].imshow(LSB_plane_blue, cmap='gray')

    if result_file_path:
        result_file_path = result_file_path + '.png'
        plt.savefig(result_file_path, bbox_inches='tight', dpi=1200)
    else:
        plt.show()

