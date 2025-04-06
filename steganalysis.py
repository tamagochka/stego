import os
from numpy import asarray, uint8
from PIL import Image
import matplotlib.pyplot as plt

from config import AppCongig


def visual_attack(app_config: AppCongig, stego_file_name: str, result_file_name: str = None):
    stego_file_path = None
    if app_config.stegos_folder:
        stego_file_path = os.path.join(app_config.stegos_folder, stego_file_name)
    else:
        stego_file_path = stego_file_name
    with Image.open(stego_file_path) as F:
        stego_object = asarray(F, dtype=uint8)

    stego_red = stego_object[:, :, 0]
    stego_green = stego_object[:, :, 1]
    stego_blue = stego_object[:, :, 2]

    LSB_plane_red = stego_red % 2
    LSB_plane_green = stego_green % 2
    LSB_plane_blue = stego_blue % 2

    fig = plt.figure()
    ax = fig.subplots(1, 3)
    ax[0].set_title('red plane')
    ax[0].imshow(LSB_plane_red, cmap='gray')
    ax[1].set_title('green plane')
    ax[1].imshow(LSB_plane_green, cmap='gray')
    ax[2].set_title('blue plane')
    ax[2].imshow(LSB_plane_blue, cmap='gray')
    
    if result_file_name:
        result_file_name = result_file_name + '.png'
        result_file_path = None
        if app_config.analysis_folder:
            result_file_path = os.path.join(app_config.analysis_folder, result_file_name)
        else:
            result_file_path = result_file_name
        plt.savefig(result_file_path)
    else:
        plt.show()

