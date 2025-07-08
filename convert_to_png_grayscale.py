from sys import argv
import os

from PIL import Image

if len(argv) > 1:
    input_file = argv[1]
    dir_name = os.path.dirname(input_file)
    file_name = os.path.basename(input_file)
    base_name, extension = os.path.splitext(file_name)
    output_file = os.path.join(dir_name, base_name + '_gs' + '.png')
    print(output_file)
    img = Image.open(argv[1]).convert('L')
    img.save(output_file)
else:
    print('укажите имя файла, который следует конвертировать')
