import sys
from math import sqrt, exp, log

from numpy import uint8, zeros, uint32, inf, float32, copy, int16, int32, float64, finfo, roll, array_equal, int8
from numpy.typing import NDArray

from .utils import MersenneTwister, chars2bytes, to_bit_vector
from .config import default_end_label
from .Embedder import Embedder
from .Extractor import Extractor


# значения по умолчанию параметров уникальных для алгоритма
default_T: int = 90
default_inv_sigma: float = 1
default_inv_gamma: float = 1
default_corr_strategy: int = 2
default_seed: int = 42


class HugoModel(object):
    """
    Модель покрывающего объекта учитывающая шум НЗБ
    """

    width: uint32 = uint32(0)  # ширина изображения
    height: uint32 = uint32(0)  # высота изображения
    count_pixels: uint32 = uint32(0)  # количество пикселе в изображении
    T: int32 = int32(0)  # диапазон различий между соседними пикселями для построения Марковского процесса
    inv_sigma: float32 = float32(0) 
    inv_gamma: float32 = float32(0)
    cover_pixel: NDArray[uint8]  # покрывающий объект
    stego_noise: NDArray[int8]  # шум НЗБ
    cooc_diff: NDArray
    distortion: float64 = float64(0)  # искажение


    def __init__(self, cover: NDArray[uint8], T: uint32, inv_sigma: float32, inv_gamma: float32):
        self.width = uint32(cover.shape[0])
        self.height = uint32(cover.shape[1])
        self.count_pixels = self.width * self.height
        self.T = int32(T)
        self.inv_sigma = inv_sigma
        self.inv_gamma = inv_gamma
        self.cover_pixel = copy(cover)
        self.stego_noise = zeros((self.width, self.height), dtype=int8)
        self.cooc_diff = zeros(2 * (2 * T + 1) * (2 * T + 1) * (2 * T + 1), dtype=int)
        self.distortion = float64(0)


    def CD(self, type: int, d1: int, d2: int, d3: int) -> int32:
        assert d1 <= self.T
        assert d1 >= -self.T
        assert d2 <= self.T
        assert d2 >= -self.T
        assert d3 <= self.T
        assert d3 >= -self.T
        assert type >= 0
        assert type <= 1
        T2 = 2 * self.T + 1
        # в оригинальном алгоритме возвращает ссылку на элемент массива,
        # но в виду того, что python так не умеет, то ф. возрващает его индекс
        return type * T2 * T2 * T2 + (d1 + self.T) * T2 * T2 + (d2 + self.T) * T2 + (d3 + self.T)


    def weight(self, d1, d2, d3) -> float32:
        y = float(d1 * d1 + d2 * d2 + d3 * d3)
        return float32(pow(sqrt(y) + self.inv_gamma, -self.inv_gamma))
    

    def set_stego_noise(self, i, j, value) -> float64:
        """
        Устанавливает значение шума НЗБ для пикселя в заданную величину и считает расчитывает искажение.
        """

        cp = self.cover_pixel[i, j]
        assert (int(cp) + int(value) >= 0) and (int(cp) + int(value) <= 255)
        dirs = [0, 1, 0, -1, 1, 0, -1, 0, 1, 1, -1, -1, 1, -1, -1, 1]
        for sum_type in [+1, -1]:
            type = 0
            for dir_id in range(len(dirs) // 2):
                if dir_id > 3:
                    type = 1
                dir1 = dirs[2 * dir_id + 0]
                dir2 = dirs[2 * dir_id + 1]
                pix_i = 0
                pix_j = 0
                for shift in range(4):
                    pix_i = int(i) + shift * dir1
                    pix_j = int(j) + shift * dir2
                in_range = (pix_i >= 0) and (pix_i < self.height) and (pix_j >= 0) and (pix_j < self.width)
                in_range &= (pix_i - 3 * dir1 >= 0) and (pix_i - 3 * dir1 < self.height) and (pix_j - 3 * dir2 >= 0) and(pix_j - 3 * dir2 < self.width)
                if in_range:
                    p0 = int(self.cover_pixel[pix_i, pix_j] + self.stego_noise[pix_i, pix_j])
                    p1 = int(self.cover_pixel[pix_i - 1 * dir1, pix_j - 1 * dir2] + self.stego_noise[pix_i - 1 * dir1, pix_j - 1 * dir2])
                    p2 = int(self.cover_pixel[pix_i - 2 * dir1, pix_j - 2 * dir2] + self.stego_noise[pix_i - 2 * dir1, pix_j - 2 * dir2])
                    p3 = int(self.cover_pixel[pix_i - 3 * dir1, pix_j - 3 * dir2] + self.stego_noise[pix_i - 3 * dir1, pix_j - 3 * dir2])
                    d1 = int(p0 - p1)
                    d2 = int(p1 - p2)
                    d3 = int(p2 - p3)
                    if (d1 >= -self.T) and (d1 <= self.T) and (d2 >= -self.T) and (d2 <= self.T) and (d3 >= -self.T) and (d3 <= self.T):
                        cd = self.CD(type, d1, d2, d3)
                        w = self.weight(d1, d2, d3)
                        if not self.cooc_diff[cd]:
                            self.distortion += w
                        elif self.cooc_diff[cd] < 0:
                            self.distortion -= sum_type * w
                        else:
                            self.distortion += sum_type * w
                        self.cooc_diff[cd] += sum_type
            self.stego_noise[i, j] = value
        return self.distortion


    def get_stego_pixel(self, i: uint32, j: uint32) -> uint8:
        """
        Получить значение пикселя покрывающего объекта с битом вложения (пикселя стеганограммы).
        """

        p = self.cover_pixel[i, j] + self.stego_noise[i, j]
        assert p >= 0 and p <= 255
        return uint8(p)


    def get_cover_pixel(self, i: uint32, j: uint32) -> uint8:
        """
        Получить значение пикселя стеганограммы.
        """

        return self.cover_pixel[i, j]
    

class HugoAlgEmbeder(object):
    """
    
    """

    model: HugoModel | None = None
    generator: MersenneTwister
    pixel_perm: NDArray[uint32]  # случайные перестановки пикселей
    pixel_perm_inv: NDArray[uint32]  # обратные перестановки для реализации стратегии коррекции №1


    def __init__(self, cover: NDArray[uint8], T: uint32, inv_sigma: float32, inv_gamma: float32, key: uint32):
        # инициализация генератора случайных чисел на основе ключа
        self.generator = MersenneTwister(int(key))
        # инициализируем модель покрывающего объекта
        self.model = HugoModel(cover, T, inv_sigma, inv_gamma)

        # генерируем перестановки
        self.pixel_perm = zeros(self.model.count_pixels, dtype=uint32)
        for i in range(self.model.count_pixels):
            self.pixel_perm[i] = i
        for i in range(self.model.count_pixels):
            j = self.generator.randint() % (self.model.count_pixels - i)
            self.pixel_perm[i + j], self.pixel_perm[i] = self.pixel_perm[i], self.pixel_perm[i + j]
        # генерируем обратные перестановки
        self.pixel_perm_inv = zeros(self.model.count_pixels, dtype=uint32)
        for i in range(self.model.count_pixels):
            self.pixel_perm_inv[self.pixel_perm[i]] = i


    def get_image(self) -> NDArray[uint8] | None:
        """
        Получить изображение со погруженным в него вложением.
        """

        if not self.model: return None
        stego: NDArray[uint8] = zeros((self.model.width, self.model.height), dtype=uint8)
        for i in range(self.model.width):
            for j in range(self.model.height):
                stego[i, j] = self.model.get_stego_pixel(uint32(i), uint32(j))
        return stego


    def embed_message(self, message_bits: NDArray[uint8], corr_strategy: uint32):
        """
        Погружение вложения, и коррекция искажений, вызваыннх погружением.
        """

        if not self.model: return
        dist_plus: NDArray[float32] = zeros(self.model.count_pixels, dtype=float32)
        dist_minus: NDArray[float32] = zeros(self.model.count_pixels, dtype=float32)
        dist_min: NDArray[float64] = zeros(self.model.count_pixels, dtype=float64)
        cover: NDArray[uint8] = zeros(self.model.count_pixels, dtype=uint8)
        stego: NDArray[uint8] = zeros(self.model.count_pixels, dtype=uint8)
        message_len = len(message_bits)

        # расчет начального искажения до встраивания
        for i in range(self.model.count_pixels):
            ip = self.pixel_perm[i]
            x = ip % self.model.height
            y = ip // self.model.height
            cp = self.model.get_cover_pixel(x, y)
            if cp <= 254:
                dist_plus[i] = self.model.set_stego_noise(x, y, +1)
            else:
                dist_plus[i] = inf
            if cp >= 1:
                dist_minus[i] = self.model.set_stego_noise(x, y, -1)
            else:
                dist_minus[i] = inf
            assert (dist_plus[i] != inf) or (dist_minus[i] != inf)
            self.model.set_stego_noise(x, y, 0)
            dist_min[i] = dist_plus[i] if dist_plus[i] < dist_minus[i] else dist_minus[i]
            # отсекаем плоскость НЗБ, в cover теперь лежат биты НЗБ
            cover[i] = self.model.get_cover_pixel(x, y) % 2

        # погружение
        stego = copy(cover)
        for i in range(message_len):
            stego[i] = message_bits[i]

        match corr_strategy:
            case 0:  # без коррекции модели, простая аддитивная апроксимация
                for i in range(self.model.count_pixels):
                    if cover[i] != stego[i]:
                        ip = self.pixel_perm[i]
                        x = ip % self.model.height
                        y = ip // self.model.height
                        if dist_plus[i] < dist_minus[i]: self.model.set_stego_noise(x, y, +1)
                        else: self.model.set_stego_noise(x, y, -1)
            case 1:  # стратегия коррекции начанющаяся от пикселя с максимальным искажением к пикселю с минимальным
                v: list[tuple[uint32, float32]] = []
                for i in range(self.model.count_pixels):
                    if cover[i] != stego[i]:
                        v.append((i, dist_min[i]))
                v.sort(key=lambda pair: pair[1], reverse=True)
                for i in range(len(v)):
                    ip = self.pixel_perm[v[i][0]]
                    x = ip % self.model.height
                    y = ip // self.model.height
                    cp = self.model.get_cover_pixel(x, y)
                    d_plus, d_minus = inf, inf
                    if cp <= 254: d_plus = self.model.set_stego_noise(x, y, +1)
                    if cp >= 1: d_minus = self.model.set_stego_noise(x, y, -1)
                    if d_plus < d_minus: self.model.set_stego_noise(x, y, +1)
            case 2:  # стратегия коррекции начанющаяся от пикселя с минимальным искажением к пикселю с максимальным
                v: list[tuple[uint32, float32]] = []
                for i in range(self.model.count_pixels):
                    if cover[i] != stego[i]:
                        v.append((i, dist_min[i]))
                v.sort(key=lambda pair: pair[1])
                for i in range(len(v)):
                    ip = self.pixel_perm[v[i][0]]
                    x = ip % self.model.height
                    y = ip // self.model.height
                    cp = self.model.get_cover_pixel(x, y)
                    d_plus, d_minus = inf, inf
                    if cp <= 254: d_plus = self.model.set_stego_noise(x, y, +1)
                    if cp >= 1: d_minus = self.model.set_stego_noise(x, y, -1)
                    if d_plus < d_minus: self.model.set_stego_noise(x, y, +1)
            case 3:  # случайная стратегия корекции
                for i in range(self.model.count_pixels):
                    if cover[i] != stego[i]:
                        ip = self.pixel_perm[i]
                        x = ip % self.model.height
                        y = ip // self.model.height
                        cp = self.model.get_cover_pixel(x, y)
                        d_plus, d_minus = inf, inf
                        if cp <= 254: d_plus = self.model.set_stego_noise(x, y, +1)
                        if cp >= 1: d_minus = self.model.set_stego_noise(x, y, -1)
                        if d_plus < d_minus: self.model.set_stego_noise(x, y, +1)
            case _:
                raise ValueError('This model correction strategy is not implemented.')


class LSB_hugo_embedding(Embedder):
    """
    Реализация алгоритма погружения в НЗБ HUGO (High Undetectable steGO)
    Получает из свойства родителя params параметр работы:
    {'T': 90}

    {'inv_sigma': 1}

    {'inv_gamma': 1}

    {'seed': 42}
        зерно для инициализации ГСПЧ
    {}
    
        
            
    {'alpha'}
        процент заполнения покрывающего объекта полезной нагрузкой (битами вложения) [0, 1]
    {'corr_strategy'}


    """

    def embeding(self):
        if self.message_bits is None: return
        # получаем параметры работы алгоритма
        T = (self.params or {}).get('T', default_T)
        inv_sigma = (self.params or {}).get('inv_sigma', default_inv_sigma)
        inv_gamma = (self.params or {}).get('inv_gamma', default_inv_gamma)
        seed = (self.params or {}).get('seed', default_seed)
        corr_strategy = (self.params or {}).get('corr_strategy', default_corr_strategy)

        # соединяем двумерные цветовые плоскости в один двумерный массив
        # cover_arr = hstack((cover_object))  # TODO распространить на несколько цветовых плоскостей

        if self.cover_object is None: return
        hugo = HugoAlgEmbeder(self.cover_object, uint32(T), float32(inv_sigma), float32(inv_gamma), uint32(seed))
        hugo.embed_message(self.message_bits, corr_strategy=corr_strategy)
        self.stego_object = hugo.get_image()


class LSB_hugo_extracting(Extractor):

    def extracting(self):
        if self.stego_object is None: return
        # получаем параметры работы алгоритма
        seed = (self.params or {}).get('seed', default_seed)
        end_label = (self.params or {}).get('end_label', default_end_label)

        # инициализация генератора случайных чисел на основе зерна
        mstw = MersenneTwister(seed)
        # резервируем место под битовую вектор-строку вложения
        H, W = self.stego_object.shape
        message_len = W * H
        self.message_bits = zeros(message_len, dtype=uint8)

        # переводим метку конца места погружения в биты
        end_label_bytes = chars2bytes(end_label)
        end_label_bits = to_bit_vector(end_label_bytes)
        end_label_bits_len = len(end_label_bits)
        # буффер в который будут помещаться последние извлеченные биты
        # для проверки их на совпадение с меткой конца места погружения
        check_end_label_bits = zeros(end_label_bits_len, dtype=int)

        # генерируем перестановки
        pixel_perm = zeros(message_len, dtype=uint32)
        for i in range(message_len):
            pixel_perm[i] = i
        for i in range(message_len):
            j = mstw.randint() % (message_len - i)
            pixel_perm[i + j], pixel_perm[i] = pixel_perm[i], pixel_perm[i + j]

        # извлечение
        for i in range(message_len):
            ip = pixel_perm[i]
            y = ip % H
            x = ip // H
            self.message_bits[i] = self.stego_object[y, x] % 2

            # сдвигаем биты в буффере влево
            check_end_label_bits = roll(check_end_label_bits, -1)
            # добавляем последний извлеченный бит в конец буффера
            check_end_label_bits[end_label_bits_len - 1] = self.message_bits[i]
            # сравниваем буффер с битами метки, если нашли метку, прекращаем извлечение
            if array_equal(check_end_label_bits, end_label_bits):
                break


if __name__ == '__main__':
    sys.exit()
