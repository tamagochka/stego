import os, sys
from random import random

from PIL import Image
from numpy import uint8, copy, zeros, asarray, fromfile, concatenate, array_split, dstack

from .utils import D2B, B2D, chars2bytes, bytes2chars, to_bit_vector, from_bit_vector


default_start_label: str = 'H@4@l0'
defualt_end_label: str = 'k0HEU'


from .Embedder import Embedder


class LSB_embedding(Embedder):
    
    def embeding(self):
        fill_rest: bool = True
        if self.params and 'fill_rest' in self.params:
            fill_rest = self.params['fill_rest']

        # получаем цветовые составляющие изображения
        if self.cover_object is None: return
        cover_red = concatenate(self.cover_object[:, :, 0])
        cover_green = concatenate(self.cover_object[:, :, 1])
        cover_blue = concatenate(self.cover_object[:, :, 2])
        # собираем все цветовые составляющие в одну вектор-строку байт
        cover_vect = concatenate([cover_red, cover_green, cover_blue])
        count_lines = len(self.cover_object[:, :, 0])
        cover_len = len(cover_vect)
        # стеганограмма - копия покрывающего объекта с измененными НЗБ
        stego_vect = copy(cover_vect)

        # погружение
        if self.message_len is None or self.message_bits is None:
            return

        for i in range(self.message_len):
            # преобразуем байт покрывающего объекта в двоичный вид
            b = D2B(cover_vect[i])
            # замещаем НЗБ покрывающего объекта битом вложения
            b[0] = self.message_bits[i]
            # сохраняем стеганограмму
            stego_vect[i] = B2D(b)

        # заполняем оставшуюся пустую часть покрывающего объекта случайными битами
        if fill_rest:
            for i in range(self.message_len, cover_len):
                b = D2B(cover_vect[i])
                b[0] = round(random())
                stego_vect[i] = B2D(b)

        # собираем изображение обратно
        stego_red, stego_green, stego_blue = array_split(stego_vect, 3)
        stego_red = array_split(stego_red, count_lines)
        stego_green = array_split(stego_green, count_lines)
        stego_blue = array_split(stego_blue, count_lines)
        self.stego_object = dstack((stego_red, stego_green, stego_blue))

        # сравнение попиксельно покрывающего объекта и стеганограммы
        # for i in range(len(cover_object)):
        #     for j in range(len(cover_object[i])):
        #         c = ''
        #         for k in range(len(cover_object[i][j])):
        #             c = c + f'{cover_object[i][j][k]:4}'
        #         c = '[' + c + ']'
        #         s = ''
        #         for k in range(len(stego[i][j])):
        #             s = s + f'{stego[i][j][k]:4}'
        #         s = '[' + s + ']'
        #         print(c, ' - ', s, '+' if c != s else ' ')
        #     print()





# def LSB_embedding(
#         cover_file_path: str,
#         stego_file_path: str,
#         message_file_path: str,
#         start_label: str = default_start_label,
#         end_label: str = defualt_end_label,
#         fill_rest: bool = True
#     ):
#     """
#     Погружение в НЗБ вложения, с использованием непрерывного замещения бит.

#     Parameters
#     ----------
#     cover_file_path: str
#         имя/путь к покрывающему объекту
#     stego_file_path: str
#         имя/путь к стеганограмме
#     message_file_path: str
#         имя/путь к файлу вложения
#     start_label: str = 'H@4@l0'
#         метка начала места погружения
#     end_label: str = 'k0HEU'
#         метка конца места погружения
#     fill_rest: bool = True
#         заполнять незаполненную часть покрывающего объекта случайными битами
#     """

#     # загрузка покрывающего объекта
#     cover_object = None
#     with Image.open(cover_file_path) as F:
#         cover_object = asarray(F, dtype=uint8)

#     # загрузка вложения
#     message_object = None
#     with open(message_file_path, 'rb') as F:
#         message_object = fromfile(F, dtype=uint8)
#         message_file_name = os.path.basename(F.name)

#     # преобразуем метки начала и конца места погружения,
#     # а также имя файла вложения и длинну его имени в байтовые вектор-строки
#     start_label_bytes = chars2bytes(start_label)
#     end_label_bytes = chars2bytes(end_label)
#     message_file_name_bytes = chars2bytes(message_file_name)
#     message_file_name_bytes_len = asarray([len(message_file_name_bytes)])
#     # собираем все в одину вектор-строку байт
#     message_bytes = concatenate((
#         start_label_bytes,
#         message_file_name_bytes_len,
#         message_file_name_bytes,
#         message_object,
#         end_label_bytes))
#     # преобразуем вктор-строку байт в вектор-строку бит
#     message_bits = to_bit_vector(message_bytes)
#     message_len = len(message_bits)

#     # получаем цветовые составляющие изображения
#     cover_red = concatenate(cover_object[:, :, 0])
#     cover_green = concatenate(cover_object[:, :, 1])
#     cover_blue = concatenate(cover_object[:, :, 2])
#     # собираем все цветовые составляющие в одну вектор-строку байт
#     cover_vect = concatenate([cover_red, cover_green, cover_blue])
#     count_lines = len(cover_object[:, :, 0])
#     cover_len = len(cover_vect)
#     # стеганограмма - копия покрывающего объекта с измененными НЗБ
#     stego_vect = copy(cover_vect)

#     # погружение
#     for i in range(message_len):
#         # преобразуем байт покрывающего объекта в двоичный вид
#         b = D2B(cover_vect[i])
#         # замещаем НЗБ покрывающего объекта битом вложения
#         b[0] = message_bits[i]
#         # сохраняем стеганограмму
#         stego_vect[i] = B2D(b)

#     # заполняем оставшуюся пустую часть покрывающего объекта случайными битами
#     if fill_rest:
#         for i in range(message_len, cover_len):
#             b = D2B(cover_vect[i])
#             b[0] = round(random())
#             stego_vect[i] = B2D(b)

#     # собираем изображение обратно
#     stego_red, stego_green, stego_blue = array_split(stego_vect, 3)
#     stego_red = array_split(stego_red, count_lines)
#     stego_green = array_split(stego_green, count_lines)
#     stego_blue = array_split(stego_blue, count_lines)
#     stego_object = dstack((stego_red, stego_green, stego_blue))

#     # сравнение попиксельно покрывающего объекта и стеганограммы
#     # for i in range(len(cover_object)):
#     #     for j in range(len(cover_object[i])):
#     #         c = ''
#     #         for k in range(len(cover_object[i][j])):
#     #             c = c + f'{cover_object[i][j][k]:4}'
#     #         c = '[' + c + ']'
#     #         s = ''
#     #         for k in range(len(stego[i][j])):
#     #             s = s + f'{stego[i][j][k]:4}'
#     #         s = '[' + s + ']'
#     #         print(c, ' - ', s, '+' if c != s else ' ')
#     #     print()

#     with Image.fromarray(stego_object) as F:
#         F.save(stego_file_path)


def LSB_extracting(
        stego_file_path: str,
        extract_file_path: str,
        start_label: str = default_start_label,
        end_label: str = defualt_end_label
    ):
    """
    Извлечение из НЗБ вложения, погруженного с использованием непрерывного замещения бит.

    Parameters
    ----------
    stego_file_path: str
        имя/путь к стеганограмме
    extract_file_path: str
        путь к файлу вложения (только директория)
    start_label: str = 'H@4@l0'
        метка начала места погружения
    end_label: str = 'k0HEU'
        метка конца места погружения
    """
    
    # загрузка стеганограммы
    stego_object = None
    with Image.open(stego_file_path, 'r') as F:
        stego_object = asarray(F, dtype=uint8)

    # получаем цветовые составляющие изображения
    stego_red = concatenate(stego_object[:, :, 0])
    stego_green = concatenate(stego_object[:, :, 1])
    stego_blue = concatenate(stego_object[:, :, 2])
    # собираем все цветовые составляющие в одну вектор-строку байт
    stego_vect = concatenate([stego_red, stego_green, stego_blue])

    stego_len = len(stego_vect)
    # резервируем место под вложение
    message_bits = zeros(stego_len, dtype=uint8)

    for i in range(stego_len):
        # байт стеганограммы в двоичный вид
        b = D2B(stego_vect[i])
        # сохраняем НЗБ стеганограммы как бит вложения
        message_bits[i] = b[0]

    # выделяем вложение
    message_bytes = from_bit_vector(message_bits)
    message = bytes2chars(message_bytes)
    message = message[message.find(start_label) + len(start_label):message.find(end_label)]

    message_file_name_len = chars2bytes(message[0])[0]
    message_file_name = message[1:message_file_name_len + 1]

    if not extract_file_path:
        extract_file_path = '.'
    message_file_path = os.path.join(extract_file_path, message_file_name)
    with open(message_file_path, 'bw') as F:
        F.write(chars2bytes(message[message_file_name_len + 1:]))


if __name__ == '__main__':
    sys.exit()
