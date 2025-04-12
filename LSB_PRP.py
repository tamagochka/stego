import os
from math import floor

from PIL import Image
from numpy import concatenate, asarray, fromfile, uint8, empty, uint16, ndarray, copy, hstack, dstack, roll, array_equal, zeros

from utils import chars2bytes, bytes2chars, to_bit_vector, from_bit_vector, D2B, B2D

# метка конца места погружения вложения в покрывающий объект ключ по умолчанию и количество пар ключей
default_primary_key: int = 125
default_count_key_pairs: int = 10
default_end_label: str = 'k0HEU'


def key_pairs_gen(primary_key: int, count_key_pairs: int) -> ndarray[uint8]:
    """
    Генерация вектора пар ключей на основании первичного ключа

    Parameters
    ----------
        primary_key: int
            первичный ключ
        count_key_pairs: int
            количество генерируемых пар ключей
    Returns
    -------
        ndarray[uint8]
            вектор-строка, содержащая key_pairs пар ключей

    """

    key_pairs = empty(2 * count_key_pairs, dtype=uint16)
    key_pairs[0] = primary_key
    for i in range(1, 2 * count_key_pairs):
        key_pairs[i] = int(str(key_pairs[i - 1] ** 2)[:3])
        if key_pairs[i] > 255:
            key_pairs[i] = int(str(key_pairs[i])[0:2])
    return key_pairs.astype(uint8)


def LSB_PRP_embedding(
        cover_file_path: str,
        stego_file_path: str,
        message_file_path: str,
        primary_key: int = default_primary_key,
        count_key_pairs: int = default_count_key_pairs,
        end_label: str = default_end_label
    ):
    """
    Погружение в НЗБ с псевдослучайной перестановкой бит вложения.

    Parameters
    ----------
        cover_file_path: str
            имя/путь к покрывающему объекту
        stego_file_path: str
            имя/путь к стеганограмме
        message_file_path: str
            имя/путь к файлу вложения
        primary_key: int = 125
            первичный ключ
        count_key_pairs: int = 10
            количество пар ключей
        end_label: str = 'k0HEU'
            метка конца места погружения
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
    # преобразуем метку конца места погружения в байтовую вектор-строку
    end_label_bytes = chars2bytes(end_label)
    # преобразуем имя файла вложения и длинну его имени в байтовые вектор-строки
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

    # # получаем цветовые составляющие изображения
    cover_red = cover_object[:, :, 0]
    cover_green = cover_object[:, :, 1]
    cover_blue = cover_object[:, :, 2]

    # соединяем двумерные цветовые плоскости в один двумерный массив
    cover_arr = hstack((cover_red, cover_green, cover_blue))
    
    # стеганограмма - копия покрывающего объекта с измененными НЗБ
    stego_arr = copy(cover_arr)
    # размеры массива
    X, Y = stego_arr.shape
    # генерируем ключевые пары
    key_pairs = key_pairs_gen(primary_key, count_key_pairs)
    
    # погружение
    for i in range(message_len):
        # округляем в меньшую сторону
        x = floor(i / Y)
        # остаток от деления
        y = i % Y
        for j in range(count_key_pairs):
            x = (x + int(B2D(D2B(key_pairs[2 * j - 1]) ^ D2B(y)))) % X
            y = (y + int(B2D(D2B(key_pairs[2 * j]) ^ D2B(x)))) % Y
        b = D2B(stego_arr[x, y])
        b[0] = message_bits[i]
        stego_arr[x, y] = B2D(b)

    # собираем обратно изображение из цветовых составляющих
    stego_red = stego_arr[:, :cover_blue.shape[1]]
    stego_green = stego_arr[:, cover_blue.shape[1]: cover_blue.shape[1] + cover_green.shape[1]]
    stego_blue = stego_arr[:, cover_blue.shape[1] + cover_green.shape[1]: cover_blue.shape[1] + cover_green.shape[1] + cover_red.shape[1]]
    stego_object = dstack((stego_red, stego_green, stego_blue))

    with Image.fromarray(stego_object) as F:
        F.save(stego_file_path)


def LSB_PRP_extracting(
        stego_file_path: str,
        extract_file_path: str,
        primary_key: int = default_primary_key,
        count_key_pairs: int = default_count_key_pairs,
        end_label: str = default_end_label
    ):
    """
    Извлечение из НЗБ с псевдослучайной перестановкой бит вложения.

    Parameters
    ----------
        stego_file_path: str
            имя/путь к стеганограмме
        extract_file_path: str
            путь к файлу вложения (только директория)
        primary_key: int = 125
            первичный ключ
        count_key_pairs: int = 10
            количество пар ключей
        end_label: str = 'k0HEU'
            метка конца места погружения
    """

    # загрузка стеганограммы
    stego_object = None
    with Image.open(stego_file_path, 'r') as F:
        stego_object = asarray(F, dtype=uint8)

    # получаем цветовые составляющие изображения
    stego_red = stego_object[:, :, 0]
    stego_green = stego_object[:, :, 1]
    stego_blue = stego_object[:, :, 2]
    # соединяем двумерные цветовые плоскости в один двумерный массив
    stego_arr = hstack((stego_red, stego_green, stego_blue))
    
    # размеры массива
    X, Y = stego_arr.shape
    # генерируем ключевые пары
    key_pairs = key_pairs_gen(primary_key, count_key_pairs)

    # резервируем место под битовую вектор-строку вложения
    message_len = X * Y
    message_bits = empty(message_len, dtype=uint8)

    # переводим метку конца места погружения в биты
    end_label_bytes = chars2bytes(end_label)
    end_label_bits = to_bit_vector(end_label_bytes)
    end_label_bits_len = len(end_label_bits)
    # буффер в который будут помещаться последние извлеченные биты
    # для проверки их на совпадение с меткой конца места погружения
    check_end_label_bits = zeros(end_label_bits_len, dtype=int)

    for i in range(message_len):
        x = floor(i / Y)
        y = i % Y
        for j in range(count_key_pairs):
            x = (x + int(B2D(D2B(key_pairs[2 * j - 1]) ^ D2B(y)))) % X
            y = (y + int(B2D(D2B(key_pairs[2 * j]) ^ D2B(x)))) % Y
        b = D2B(stego_arr[x, y])
        message_bits[i] = b[0]
        # сдвигаем биты в буффере влево
        check_end_label_bits = roll(check_end_label_bits, -1)
        # добавляем последний извлеченный бит в конец буффера
        check_end_label_bits[end_label_bits_len - 1] = b[0]
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
