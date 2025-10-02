import sys
from math import sqrt, exp, log
from abc import ABC, abstractmethod

from numpy import uint8, zeros, uint32, inf, float32, copy, int32, float64, finfo, roll, array_equal, int8
from numpy.typing import NDArray
from hamming_code import encode

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
default_rel_payload: float = 0.5


def binary_entropyf(x: float32) -> float32:
    LOG2 = log(2.0)
    EPS = finfo(float32).eps
    z = float32(0)
    if (x < EPS) or ((1 - x) < EPS):
        return float32(0)
    else:
        z = float32((-x * log(x) - (1 - x) * log(1 - x)) / LOG2)
        return z        


def calc_lambda_from_payload(message_length: uint32, weigts: NDArray[float64], n: uint32) -> float32:
    
    l1 = float32(0)
    l2 = float32(0)
    l3 = float32(1000.0)
    m1 = float32(n)
    m2 = float32(0)
    m3: float32 = float32(message_length + 1)
    j = int32(0)
    iterations = uint32(0)

    while m3 > message_length:
        l3 *= 2
        m3 = float32(0)
        for i in range(n):
            m3 += binary_entropyf(float32(1 / (1 + exp(-l3 * weigts[i]))))
        j += 1
        if j > 10:
            return l3
        iterations += 1
    alpha: float32 = message_length / n
    while ((m1 - m3) / n > alpha / 1000.0) and (iterations < 30):
        l2 = l1 + (l3 - l1) / 2
        m2 = float32(0)
        for i in range(n):
            m2 += binary_entropyf(float32(1 / (1 + exp(-l2 * weigts[i]))))
        if m2 < message_length:
            l3 = l2
            m3 = m2
        else:
            l1 = l2
            m1 = m2
        iterations += 1
    return l2


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
        Устанавливает значение шума НЗБ для пикселя в заданную величину и расчитывает искажение.
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
        Получить значение пикселя покрывающего объекта.
        """

        return self.cover_pixel[i, j]
    


class HugoAlgBase(ABC):
    """
    
    """

    model: HugoModel | None = None
    generator: MersenneTwister
    pixel_perm: NDArray[uint32]  # случайные перестановки пикселей
    pixel_perm_inv: NDArray[uint32]  # обратные перестановки для реализации стратегии коррекции №1
    corr_strategy: int = default_corr_strategy
    dist_min: NDArray[float64]

    def __init__(self, cover: NDArray[uint8], T: uint32, inv_sigma: float32, inv_gamma: float32, seed: uint32, corr_strategy: int):
        # инициализация генератора случайных чисел на основе ключа
        self.generator = MersenneTwister(int(seed))
        # инициализируем модель покрывающего объекта
        self.model = HugoModel(cover, T, inv_sigma, inv_gamma)

        self.corr_strategy = corr_strategy

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
        Получить изображение с погруженным в него вложением.
        """

        if not self.model: return None
        stego: NDArray[uint8] = zeros((self.model.width, self.model.height), dtype=uint8)
        for i in range(self.model.width):
            for j in range(self.model.height):
                stego[i, j] = self.model.get_stego_pixel(uint32(i), uint32(j))
        return stego


    def embed_message(self):
        """
        Погружение вложения, и коррекция искажений, вызваыннх погружением.
        """

        if not self.model: return
        dist_plus: NDArray[float32] = zeros(self.model.count_pixels, dtype=float32)
        dist_minus: NDArray[float32] = zeros(self.model.count_pixels, dtype=float32)
        self.dist_min: NDArray[float64] = zeros(self.model.count_pixels, dtype=float64)
        cover: NDArray[uint8] = zeros(self.model.count_pixels, dtype=uint8)
        stego: NDArray[uint8] = zeros(self.model.count_pixels, dtype=uint8)

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
            self.dist_min[i] = dist_plus[i] if dist_plus[i] < dist_minus[i] else dist_minus[i]
            # отсекаем плоскость НЗБ, в cover теперь лежат биты НЗБ
            cover[i] = self.model.get_cover_pixel(x, y) % 2

        # погружение
        self.binary_embed(cover, stego)

        match self.corr_strategy:
            case 0:  # без коррекции модели, простая аддитивная апроксимация
                for i in range(self.model.count_pixels):
                    if cover[i] != stego[i]:
                        ip = self.pixel_perm[i]
                        x = ip % self.model.height
                        y = ip // self.model.height
                        if dist_plus[i] < dist_minus[i]: self.model.set_stego_noise(x, y, +1)
                        else: self.model.set_stego_noise(x, y, -1)
            case 1:
                for i in range(self.model.count_pixels):
                    ip = self.pixel_perm_inv[i]
                    if cover[ip] != stego[ip]:
                        x = ip % self.model.height
                        y = ip // self.model.height
                        cp = self.model.get_cover_pixel(x, y)
                        d_plus, d_minus = inf, inf
                        if cp <= 254: d_plus = self.model.set_stego_noise(x, y, +1)
                        if cp >= 1: d_minus = self.model.set_stego_noise(x, y, -1)
                        if d_plus < d_minus: self.model.set_stego_noise(x, y, +1)
            case 2:  # стратегия коррекции начанющаяся от пикселя с максимальным искажением к пикселю с минимальным
                v: list[tuple[int, float]] = []
                for i in range(self.model.count_pixels):
                    if cover[i] != stego[i]:
                        v.append((i, self.dist_min[i]))
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
            case 3:  # стратегия коррекции начанющаяся от пикселя с минимальным искажением к пикселю с максимальным
                v: list[tuple[int, float]] = []
                for i in range(self.model.count_pixels):
                    if cover[i] != stego[i]:
                        v.append((i, self.dist_min[i]))
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
            case 4:  # случайная стратегия корекции
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


    @abstractmethod
    def binary_embed(self, cover: NDArray[uint8], stego: NDArray[uint8]):
        pass



class HugoAlgEmbeder(HugoAlgBase):

    message_bits: NDArray[uint8] | None = None


    def __init__(self, cover: NDArray[uint8], message_bits: NDArray[uint8], T: uint32, inv_sigma: float32, inv_gamma: float32, seed: uint32, corr_strategy: int):
        super().__init__(cover, T, inv_sigma, inv_gamma, seed, corr_strategy)
        self.message_bits = message_bits


    def binary_embed(self, cover: NDArray[uint8], stego: NDArray[uint8]):
        if self.message_bits is None or self.model is None: return
        message_len = len(self.message_bits)
        for i in range(self.model.count_pixels):
            if i < message_len:
                stego[i] = self.message_bits[i]
            else:
                stego[i] = cover[i]



class HugoAlgProbability(HugoAlgBase):

    def __init__(self, cover: NDArray[uint8], T: uint32, inv_sigma: float32, inv_gamma: float32, seed: uint32, corr_strategy: int):
        super().__init__(cover, T, inv_sigma, inv_gamma, seed, corr_strategy)


    @abstractmethod
    def binary_embed(self, cover: NDArray[uint8], stego: NDArray[uint8]):

        pass



class HugoAlgSimulator(HugoAlgProbability):

    rel_payload: float32 = float32(0)


    def __init__(self, cover: NDArray[uint8], rel_payload: float32, T: uint32, inv_sigma: float32, inv_gamma: float32, seed: uint32, corr_strategy: int):
        super().__init__(cover, T, inv_sigma, inv_gamma, seed, corr_strategy)
        self.rel_payload = rel_payload


    def binary_embed(self, cover: NDArray[uint8], stego: NDArray[uint8]):
        if self.model is None: return
        weights = self.dist_min
        message_len = uint32(self.rel_payload * self.model.count_pixels)
        lmbd: float32 = calc_lambda_from_payload(message_len, weights, self.model.count_pixels)
        for i in range(self.model.count_pixels):
            flip_prob = exp(-lmbd * weights[i]) / (1 + exp(-lmbd * weights[i]))
            # иммитация замены бита плоскости НЗБ на бит вложения
            stego[i] = cover[i] ^ 1 if self.generator.random() < flip_prob else cover[i]



class HugoAlgProbabilityEmbeder(HugoAlgProbability):

    message_bits: NDArray[uint8] | None = None
    extraction_key: list[int] = []


    def __init__(self, cover: NDArray[uint8], message_bits: NDArray[uint8], T: uint32, inv_sigma: float32, inv_gamma: float32, seed: uint32, corr_strategy: int):
        super().__init__(cover, T, inv_sigma, inv_gamma, seed, corr_strategy)
        self.message_bits = message_bits


    def binary_embed(self, cover: NDArray[uint8], stego: NDArray[uint8]):
        if self.message_bits is None or self.model is None: return

        hamming_message
        tmp = zeros(32, dtype=uint8)
        for i in range(len(self.message_bits)):
            tmp[i % 8] = self.message_bits[i]
            if not (i + 1) % 8:
                val = 0
                for b in tmp:
                    val = (val << 1) | b
                code = encode(int(val))
                

                

        # print()



        weights = self.dist_min
        message_len = uint32(len(self.message_bits))
        lmbd: float32 = calc_lambda_from_payload(message_len, weights, self.model.count_pixels)

        F = open('embeding.txt', 'w')
        n = 0
        for i in range(self.model.count_pixels):
            if n < message_len:
                r = self.generator.random()
                flip_prob = exp(-lmbd * weights[i]) / (1 + exp(-lmbd * weights[i]))

                F.write(f'{r}')

                if self.message_bits[n] != cover[i]:

                    if r < flip_prob:
                        F.write(f' < ')
                        F.write(f'{flip_prob} - bit\n')

                        stego[i] = cover[i] ^ 1
                        n += 1
                    else:
                        F.write(f' > ')
                        F.write(f'{flip_prob}\n')

                        stego[i] = cover[i]
                else:
                    self.extraction_key.append(i)  # список пикселей, в которых бит вложения совпадает с битом покрывающего объекта
                    stego[i] = cover[i]
                    n += 1
                    

                    F.write(f' - {flip_prob}\n')


        
        # F.write(str(self.message_bits))

        F.close()

class HugoAlgProbabilityExtractor(object):

    extraction_key: list[int] | None = None
    generator: MersenneTwister
    model: HugoModel | None = None


    def __init__(self, stego: NDArray[uint8], extraction_key: list[int], T: uint32, inv_sigma: float32, inv_gamma: float32, seed: uint32):
        # инициализация генератора случайных чисел на основе ключа
        self.generator = MersenneTwister(int(seed))
        # инициализируем модель стеганограммы
        self.model = HugoModel(stego, T, inv_sigma, inv_gamma)
        self.extraction_key = extraction_key


    def binary_extract(self) -> NDArray[uint8] | None:
        if not self.model or self.extraction_key is None: return None
        # резервируем место под битовую вектор-строку вложения
        message_bits = zeros(self.model.count_pixels, dtype=uint8)
        dist_min: NDArray[float64] = zeros(self.model.count_pixels, dtype=float64)
        dist_plus: NDArray[float32] = zeros(self.model.count_pixels, dtype=float32)
        dist_minus: NDArray[float32] = zeros(self.model.count_pixels, dtype=float32)
        stego: NDArray[uint8] = zeros(self.model.count_pixels, dtype=uint8)

        # генерируем перестановки
        pixel_perm = zeros(self.model.count_pixels, dtype=uint32)
        for i in range(self.model.count_pixels):
            pixel_perm[i] = i
        for i in range(self.model.count_pixels):
            j = self.generator.randint() % (self.model.count_pixels - i)
            pixel_perm[i + j], pixel_perm[i] = pixel_perm[i], pixel_perm[i + j]

        # расчет начального искажения
        for i in range(self.model.count_pixels):
            ip = pixel_perm[i]
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
            stego[i] = self.model.get_cover_pixel(x, y) % 2

        weights = dist_min

        message_len = uint32(7792) # TODO !!!

        lmbd: float32 = calc_lambda_from_payload(message_len, dist_min, self.model.count_pixels)
        n = 0

        F = open('extracting.txt', 'w')

        for i in range(self.model.count_pixels):
            r = self.generator.random()
            flip_prob = exp(-lmbd * weights[i]) / (1 + exp(-lmbd * weights[i]))

            F.write(f'{r}')

            if i in self.extraction_key:
                message_bits[n] = stego[i]
                n += 1
                F.write(f' - {flip_prob}\n')
            else:
                if r < flip_prob:
                    F.write(f' < {flip_prob} - bit\n')
                    message_bits[n] = stego[i]
                    n += 1
                else:
                    F.write(f' > {flip_prob}\n')



        F.close()
        
        
        return message_bits



class LSB_hugo_embedding(Embedder):
    """
    Реализация алгоритма погружения в НЗБ HUGO (High Undetectable steGO) (hugo)
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
        # получаем параметры работы алгоритма
        T = (self.params or {}).get('T', default_T)
        inv_sigma = (self.params or {}).get('inv_sigma', default_inv_sigma)
        inv_gamma = (self.params or {}).get('inv_gamma', default_inv_gamma)
        seed = (self.params or {}).get('seed', default_seed)
        corr_strategy = (self.params or {}).get('corr_strategy', default_corr_strategy)
        rel_payload = (self.params or {}).get('rel_payload', default_rel_payload)


        # соединяем двумерные цветовые плоскости в один двумерный массив
        # cover_arr = hstack((cover_object))  # TODO распространить на несколько цветовых плоскостей

        if self.cover_object is None: return
        # hugo = HugoAlgSimulator(self.cover_object, float32(rel_payload), uint32(T), float32(inv_sigma), float32(inv_gamma), uint32(seed), corr_strategy)

        if self.message_bits is None: return
        # hugo = HugoAlgEmbeder(self.cover_object, self.message_bits, uint32(T), float32(inv_sigma), float32(inv_gamma), uint32(seed), corr_strategy)
        hugo = HugoAlgProbabilityEmbeder(self.cover_object, self.message_bits, uint32(T), float32(inv_sigma), float32(inv_gamma), uint32(seed), corr_strategy)


        hugo.embed_message()

        print(hugo.extraction_key)

        self.stego_object = hugo.get_image()



class LSB_hugo_extracting(Extractor):
    """
    (hugo)
    """

    def extracting(self):
        if self.stego_object is None: return
        # получаем параметры работы алгоритма
        seed = (self.params or {}).get('seed', default_seed)
        end_label = (self.params or {}).get('end_label', default_end_label)
        probability = (self.params or {}).get('probability', False)
        extraction_key_file = (self.params or {}).get('extraction_key_file', None)
        T = (self.params or {}).get('T', default_T)
        inv_sigma = (self.params or {}).get('inv_sigma', default_inv_sigma)
        inv_gamma = (self.params or {}).get('inv_gamma', default_inv_gamma)

        if probability:
            if not extraction_key_file: return
            with open(extraction_key_file, 'r') as F:
                extraction_key = F.readline()
            extraction_key = [int(item) for item in extraction_key.split(',')]
            hugo_extractor = HugoAlgProbabilityExtractor(self.stego_object, extraction_key, T, inv_sigma, inv_gamma, seed)
            self.message_bits = hugo_extractor.binary_extract()


        else:
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
