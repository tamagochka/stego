import os, sys
import math

from PIL import Image
from numpy import copy, uint8, asarray, concatenate, fromfile, zeros, roll, array_equal

from .utils import chars2bytes, bytes2chars, to_bit_vector, from_bit_vector, MersenneTwister


luminance: float = 0.1  # коээфициент яркости пикселя с погруженными битами
repeats: int = 5  # число мест погружения каждого бита вложения
sigma: int = 3  # размер креста из пикселей на основе которых вычисляется прогнозируемое значение яркости пикселя, расположенного в центре креста

default_key: int = 42  # ключ на основе которого генерируются ПСЧ для определения координат места погружения бит вложения
default_end_label: str = 'k0HEU'  # метка конца места встраивания


def LSB_kdb_embedding(
        cover_file_path: str,
        stego_file_path: str,
        message_file_path: str,
        end_label: str = default_end_label,
        key: int = default_key
    ):
    """
    Погружение на основе метода Куттера-Джордана-Боссена

    """

    # загрузка покрывающего объекта
    cover_object = None
    with Image.open(cover_file_path) as F:
        cover_object = asarray(F, dtype=uint8)

    # загрузка вложения
    message_object = None
    with open(message_file_path, 'rb') as F:
        message_object = fromfile(F, dtype=uint8)
        message_file_name = os.path.basename(F.name)

    # преобразуем метку конца сообщения, имя файла вложения и длинну его имени в байтовые вектор-строки
    end_label_bytes = chars2bytes(end_label)
    message_file_name_bytes = chars2bytes(message_file_name)
    message_file_name_bytes_len = asarray([len(message_file_name_bytes)])
    # собираем все в одину вектор-строку байт
    message_bytes = concatenate((
        message_file_name_bytes_len,
        message_file_name_bytes,
        message_object,
        end_label_bytes))
    # преобразуем вктор-строку байт в вектор-строку бит
    message_bits = to_bit_vector(message_bytes)
    message_len = len(message_bits)

    # инициализация генератора случайных чисел на основе ключа
    mstw = MersenneTwister(key)
    
    # стеганограмма - копия покрывающего объекта с измененными пикселями
    stego_object = copy(cover_object)

    for i in range(message_len):
        for j in range(repeats):
            # генерируем координаты пикселя в который будет производиться погружение
            Cx = mstw.randint((4, cover_object.shape[0] - 4))
            Cy = mstw.randint((4, cover_object.shape[1] - 4))
            # рассчитываем яркость пикселя
            Y = (0.298 * cover_object[Cx, Cy, 0]) + \
                (0.586 * cover_object[Cx, Cy, 1]) + \
                (0.114 * cover_object[Cx, Cy, 2])
            if math.isclose(Y, 0):
                Y = 5 / luminance
            # если бит вложения равен 1, то прибавляем яркость, иначе вычитаем
            Y = Y if message_bits[i] else -Y
            b = cover_object[Cx, Cy, 2] + Y * luminance
            # если вылезли за пределы допустимых значений
            b = 255 if b > 255 else b
            b = 0 if b < 0 else b
            # изменяем значение пикселя
            stego_object[Cx, Cy, 2] = b

    with Image.fromarray(stego_object) as F:
        F.save(stego_file_path)


def LSB_kdb_extracting(
        stego_file_path: str,
        extract_file_path: str,
        end_label: str = default_end_label,
        key: int = default_key
    ):
    """
    Извлечение на основе метода Куттера-Джордана-Боссена

    """

    # загрузка стеганограммы
    stego_object = None
    with Image.open(stego_file_path, 'r') as F:
        stego_object = asarray(F, dtype=int)

    # резервируем место под битовую вектор-строку вложения
    message_len = stego_object.shape[0] * stego_object.shape[1]
    message_bits = zeros(message_len, dtype=uint8)
    
    # переводим метку конца места погружения в биты
    end_label_bytes = chars2bytes(end_label)
    end_label_bits = to_bit_vector(end_label_bytes)
    end_label_bits_len = len(end_label_bits)
    # буффер в который будут помещаться последние извлеченные биты
    # для проверки их на совпадение с меткой конца места погружения
    check_end_label_bits = zeros(end_label_bits_len, dtype=int)

    # инициализация генератора случайных чисел на основе ключа
    mstw = MersenneTwister(key)

    for i in range(message_len):
        s = 0
        for j in range(repeats):
            # генерируем координаты пикселя в который будет производиться погружение
            Cx = mstw.randint((4, stego_object.shape[0] - 4))
            Cy = mstw.randint((4, stego_object.shape[1] - 4))
            # делаем предсказание о яркости текущего пикселя по значениям соседних
            prediction = (sum(stego_object[Cx - sigma : Cx + sigma + 1, Cy, 2]) + \
                sum(stego_object[Cx, Cy - sigma : Cy + sigma + 1, 2]) - \
                2 * stego_object[Cx, Cy, 2]) / (4 * sigma)
            # вычисляем разницу между предсказанным и реальным значением яркости
            diff = stego_object[Cx, Cy, 2] - prediction
            if math.isclose(diff, 0) and math.isclose(prediction, 255):
                diff = 0.5
            if math.isclose(diff, 0) and math.isclose(prediction, 0):
                diff = -0.5
            if diff > 0:
                s += 1
        message_bits[i] = round(s / repeats)
        # сдвигаем биты в буффере влево
        check_end_label_bits = roll(check_end_label_bits, -1)
        # добавляем последний извлеченный бит в конец буффера
        check_end_label_bits[end_label_bits_len - 1] = message_bits[i]
        # сравниваем буффер с битами метки, если нашли метку, прекращаем извлечение
        if array_equal(check_end_label_bits, end_label_bits):
            break

    message_bytes = from_bit_vector(message_bits)
    message = bytes2chars(message_bytes)
    message = message[0:message.find(end_label)]
    message_file_name_len = chars2bytes(message[0])[0]
    message_file_name = message[1:message_file_name_len + 1]
    message_file_path = os.path.join(extract_file_path, message_file_name)
    with open(message_file_path, 'bw') as F:
        F.write(chars2bytes(message[message_file_name_len + 1:]))


if __name__ == '__main__':
    sys.exit()
