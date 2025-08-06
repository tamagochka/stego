import sys
import math

from numpy import copy, uint8, zeros, roll, array_equal

from .utils import chars2bytes, to_bit_vector, MersenneTwister
from .config import default_end_label
from .Embedder import Embedder
from .Extractor import Extractor


default_key: int = 42  # ключ на основе которого генерируются ПСЧ для определения координат места погружения бит вложения
default_luminance: float = 0.1  # коээфициент яркости пикселя с погруженными битами
default_repeats: int = 5  # число мест погружения каждого бита вложения
default_sigma: int = 3  # размер креста из пикселей на основе которых вычисляется прогнозируемое значение яркости пикселя, расположенного в центре креста


class LSB_kdb_embedding(Embedder):
    """
    Реализация алгоритма погружения в НЗБ с использованием метода Куттера-Джордана-Боссена.
    Получает из свойства родителя params параметр работы:
    {'key': 42}
        ключ на основе которого генерируются ПСЧ для определения координат места погружения бит вложения
    {'luminance': 0.1}
        коээфициент яркости пикселя с погруженными битами
    {'repeats': 5}
        число мест погружения каждого бита вложения
    """

    def embeding(self):
        # получаем параметры работы алгоритма
        key = default_key
        if self.params and 'key' in self.params:
            key = self.params['key']
        luminance = default_luminance
        if self.params and 'luminance' in self.params:
            luminance = self.params['luminance']
        repeats = default_repeats
        if self.params and 'repeats' in self.params:
            repeats = self.params['repeats']
        # инициализация генератора случайных чисел на основе ключа
        mstw = MersenneTwister(key)
        # стеганограмма - копия покрывающего объекта с измененными пикселями
        if self.cover_object is None: return
        self.stego_object = copy(self.cover_object)

        # погружение
        if self.message_bits is None: return
        message_len = len(self.message_bits)

        for i in range(message_len):
            for j in range(repeats):
                # генерируем координаты пикселя в который будет производиться погружение
                Cx = mstw.randint((4, self.cover_object.shape[0] - 4))
                Cy = mstw.randint((4, self.cover_object.shape[1] - 4))
                # рассчитываем яркость пикселя
                Y = (0.298 * self.cover_object[Cx, Cy, 0]) + \
                    (0.586 * self.cover_object[Cx, Cy, 1]) + \
                    (0.114 * self.cover_object[Cx, Cy, 2])
                if math.isclose(Y, 0):
                    Y = 5 / luminance
                # если бит вложения равен 1, то прибавляем яркость, иначе вычитаем
                Y = Y if self.message_bits[i] else -Y
                b = self.cover_object[Cx, Cy, 2] + Y * luminance
                # если вылезли за пределы допустимых значений
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b
                # изменяем значение пикселя
                self.stego_object[Cx, Cy, 2] = b


class LSB_kdb_extracting(Extractor):
    """
    Реализация алгоритма извлечения из НЗБ вложения, погруженного с использованием метода Куттера-Джордана-Боссена.
    Получает из свойства родителя params параметр работы:
    {'key': 42}
        ключ на основе которого генерируются ПСЧ для определения координат места погружения бит вложения
    {'sigma': 3}
        размер креста из пикселей на основе которых вычисляется прогнозируемое значение яркости пикселя, расположенного в центре креста
    {'repeats': 5}
        число мест погружения каждого бита вложения
    {'end_label': 'k0HEU'}
        метка конца места погружения
    """

    def extracting(self):
        # получаем параметры работы алгоритма
        key = default_key
        if self.params and 'key' in self.params:
            key = self.params['key']
        sigma = default_sigma
        if self.params and 'sigma' in self.params:
            sigma = self.params['sigma']
        repeats = default_repeats
        if self.params and 'repeats' in self.params:
            repeats = self.params['repeats']
        end_label = default_end_label
        if self.params and 'end_label' in self.params:
            end_label = self.params['end_label']
        # резервируем место под битовую вектор-строку вложения
        if self.stego_object is None: return
        message_len = self.stego_object.shape[0] * self.stego_object.shape[1]
        self.message_bits = zeros(message_len, dtype=uint8)
        # переводим метку конца места погружения в биты
        end_label_bytes = chars2bytes(end_label)
        end_label_bits = to_bit_vector(end_label_bytes)
        end_label_bits_len = len(end_label_bits)
        # буффер в который будут помещаться последние извлеченные биты
        # для проверки их на совпадение с меткой конца места погружения
        check_end_label_bits = zeros(end_label_bits_len, dtype=int)
        # инициализация генератора случайных чисел на основе ключа
        mstw = MersenneTwister(key)

        # извлечение
        for i in range(message_len):
            s = 0
            for j in range(repeats):
                # генерируем координаты пикселя из которого будем извлекать бит вложения
                Cx = mstw.randint((4, self.stego_object.shape[0] - 4))
                Cy = mstw.randint((4, self.stego_object.shape[1] - 4))
                # делаем предсказание о яркости текущего пикселя по значениям соседних
                prediction = (sum(self.stego_object[Cx - sigma : Cx + sigma + 1, Cy, 2]) + \
                    sum(self.stego_object[Cx, Cy - sigma : Cy + sigma + 1, 2]) - \
                    2 * self.stego_object[Cx, Cy, 2]) / (4 * sigma)
                # вычисляем разницу между предсказанным и реальным значением яркости
                diff = self.stego_object[Cx, Cy, 2] - prediction
                if math.isclose(diff, 0) and math.isclose(prediction, 255):
                    diff = 0.5
                if math.isclose(diff, 0) and math.isclose(prediction, 0):
                    diff = -0.5
                if diff > 0:
                    s += 1
            self.message_bits[i] = round(s / repeats)
            # сдвигаем биты в буффере влево
            check_end_label_bits = roll(check_end_label_bits, -1)
            # добавляем последний извлеченный бит в конец буффера
            check_end_label_bits[end_label_bits_len - 1] = self.message_bits[i]
            # сравниваем буффер с битами метки, если нашли метку, прекращаем извлечение
            if array_equal(check_end_label_bits, end_label_bits):
                break


if __name__ == '__main__':
    sys.exit()
