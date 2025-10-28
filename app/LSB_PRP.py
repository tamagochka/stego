import sys
from math import floor

from numpy import uint8, copy, roll, array_equal, zeros

from .utils import chars2bytes, to_bit_vector, D2B, B2D, key_pairs_gen, one_arr_to_img_arr_surfs, img_arr_surfs_to_one_arr
from .config import default_end_label
from .Embedder import Embedder
from .Extractor import Extractor


# значения по умолчанию параметров уникальных для алгоритма
default_primary_key: int = 42
default_count_key_pairs: int = 10


class LSB_PRP_embedding(Embedder):
    """
    Реализация алгоритма погружения в НЗБ вложения с псевдослучайной перестановкой бит вложения (prp).
    Получает из свойства родителя params параметр работы:
    {'primary_key': 42}
        первичный ключ
    {'count_key_pairs': 10}
        количество пар ключей
    """

    def embedding(self):
        # получаем параметры работы алгоритма
        primary_key = (self.params or {}).get('primary_key', default_primary_key)
        count_key_pairs = (self.params or {}).get('count_key_pairs', default_count_key_pairs)
        
        # соединяем двумерные цветовые плоскости в один двумерный массив
        cover_arr, *widths = img_arr_surfs_to_one_arr(self.cover_object)
        if cover_arr is None: return

        # стеганограмма - копия покрывающего объекта с измененными НЗБ
        stego_arr = copy(cover_arr)

        # размеры массива
        X, Y = stego_arr.shape
        # генерируем ключевые пары
        key_pairs = key_pairs_gen(primary_key, count_key_pairs)
        
        # погружение
        if self.message_bits is None: return
        message_len = len(self.message_bits)

        for i in range(message_len):
            # округляем в меньшую сторону
            x = floor(i / Y)
            # остаток от деления
            y = i % Y
            for j in range(count_key_pairs):
                x = (x + int(B2D(D2B(key_pairs[2 * j - 1]) ^ D2B(uint8(y & 0xFF))))) % X
                y = (y + int(B2D(D2B(key_pairs[2 * j]) ^ D2B(uint8(x & 0xFF))))) % Y
            b = D2B(stego_arr[x, y])
            b[0] = self.message_bits[i]
            stego_arr[x, y] = B2D(b)

        # собираем изображение обратно
        self.stego_object = one_arr_to_img_arr_surfs(stego_arr, *widths)


class LSB_PRP_extracting(Extractor):
    """
    Реализация алгоритма извлечения из НЗБ вложения, погруженного с использованием алгоритма с псевдослучайной перестановкой бит вложения (prp).
    Получает из свойства родителя params параметр работы:
    {'primary_key': 42}
        первичный ключ
    {'count_key_pairs': 10}
        количество пар ключей
    {'end_label': 'k0HEU'}
        метка конца места погружения
    """

    def extracting(self):
        # получаем параметры работы алгоритма
        primary_key = (self.params or {}).get('primary_key', default_primary_key)
        count_key_pairs = (self.params or {}).get('count_key_pairs', default_count_key_pairs)
        end_label = (self.params or {}).get('end_label', default_end_label)

        # соединяем двумерные цветовые плоскости в один двумерный массив
        stego_arr = img_arr_surfs_to_one_arr(self.stego_object)[0]
        if stego_arr is None: return

        # размеры массива
        X, Y = stego_arr.shape
        # генерируем ключевые пары
        key_pairs = key_pairs_gen(primary_key, count_key_pairs)

        # резервируем место под битовую вектор-строку вложения
        message_len = X * Y
        self.message_bits = zeros(message_len, dtype=uint8)

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
                x = (x + int(B2D(D2B(key_pairs[2 * j - 1]) ^ D2B(uint8(y & 0xFF))))) % X
                y = (y + int(B2D(D2B(key_pairs[2 * j]) ^ D2B(uint8(x & 0xFF))))) % Y
            b = D2B(stego_arr[x, y])
            self.message_bits[i] = b[0]
            # сдвигаем биты в буффере влево
            check_end_label_bits = roll(check_end_label_bits, -1)
            # добавляем последний извлеченный бит в конец буффера
            check_end_label_bits[end_label_bits_len - 1] = b[0]
            # сравниваем буффер с битами метки, если нашли метку, прекращаем извлечение
            if array_equal(check_end_label_bits, end_label_bits):
                break


if __name__ == '__main__':
    sys.exit()
