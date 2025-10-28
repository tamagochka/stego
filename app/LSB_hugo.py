import sys
from math import sqrt, exp, log
from abc import ABC, abstractmethod

from numpy.typing import NDArray
from numpy import uint8, zeros, uint32, inf, float32, copy, int32, float64, finfo, int8

from .Embedder import Embedder
from .Extractor import Extractor
from .utils import MersenneTwister, img_arr_surfs_to_one_arr, one_arr_to_img_arr_surfs


# значения по умолчанию параметров уникальных для алгоритма
default_action_type: str = 'prob'
# тип погружения:
# 'det'  - детерминированное погружение,
# 'prob' - вероятностное погружение,
# 'sim'  - симуляция вероятностного погружения
default_T: int = 90  # размер окна для модели совместной встречаемости
default_inv_sigma: float = 1  # параметр модели, управляющий общими искажениями
default_inv_gamma: float = 1  # параметр модели, управляющий локальными искажениями
default_corr_strategy: int = 2  # стратегия коррекции модели после погружения
default_seed: int = 42  # зерно для инициализации ГСПЧ
default_rel_payload: float = 0.5  # процент заполнения покрывающего объекта полезной нагрузкой (битами вложения) при симуляции погружения


def binary_entropyf(x: float32) -> float32:
    """
    Рассчет двоичной энтропии.

    Parameters
    ----------
    x: float32
        вероятность

    Returns
    -------
    float32
        значение двоичной энтропии
    """

    LOG2 = log(2.0)
    EPS = finfo(float32).eps
    z = float32(0)
    if (x < EPS) or ((1 - x) < EPS):
        return float32(0)
    else:
        z = float32((-x * log(x) - (1 - x) * log(1 - x)) / LOG2)
        return z        


def calc_lambda_from_payload(message_length: uint32, weigts: NDArray[float64], n: uint32) -> float32:
    """
    Рассчет параметра лямбда для вероятностного погружения.

    Parameters
    ----------
    message_length: uint32
        длинна сообщения (в битах)
    weigts: NDArray[float64]
        веса пикселей покрывающего объекта
    n: uint32
        количество пикселей покрывающего объекта
    
    Returns
    -------
    float32
        значение лямбда
    """

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

    Attributes
    ----------
    width: uint32
        ширина изображения
    height: uint32
        высота изображения
    count_pixels: uint32
        количество пикселей в покрывающем объекте
    T: int32
        диапазон различий между соседними пикселями для построения Марковского процесса
    inv_sigma: float32
        параметр модели
    inv_gamma: float32
        параметр модели
    cover_pixel: NDArray[uint8]
        покрывающий объект
    stego_noise: NDArray[int8]
        модель шума НЗБ, используемая при погружении
    cooc_diff: NDArray
        модель совместной встречаемости различий между соседними пикселями
    distortion: float64
        искажение

    Methods
    -------
    CD(type: int, d1: int, d2: int, d3: int) -> int32
        вычисление индекса элемента массива модели совместной встречаемости
    weight(d1: int, d2: int, d3: int) -> float32
        рассчет влияния пикселя на искажение
    set_stego_noise(i: int32, j: int32, value: int8) -> float64
        устанавливает значение шума НЗБ для пикселя в заданную величину и расчитывает искажение
    get_stego_pixel(i: uint32, j: uint32) -> uint8
        получить значение пикселя покрывающего объекта с битом вложения (пикселя стеганограммы)
    get_cover_pixel(i: uint32, j: uint32) -> uint8
        получить значение пикселя покрывающего объекта
    """

    width: uint32
    height: uint32
    count_pixels: uint32
    T: int32
    inv_sigma: float32
    inv_gamma: float32
    cover_pixel: NDArray[uint8]
    stego_noise: NDArray[int8]
    cooc_diff: NDArray[int8]
    distortion: float64


    def __init__(
            self,
            cover: NDArray[uint8],
            T: uint32 = uint32(0),
            inv_sigma: float32 = float32(0),
            inv_gamma: float32 = float32(0)
    ):
        """
        Инициализация модели покрывающего объекта.

        Parameters
        ----------
        cover: NDArray[uint8]
            покрывающий объект
        T: uint32
            диапазон различий между соседними пикселями для построения Марковского процесса
        inv_sigma: float32
            параметр модели, управляющий общими искажениями
        inv_gamma: float32
            параметр модели, управляющий локальными искажениями
        """

        self.width = uint32(cover.shape[0])
        self.height = uint32(cover.shape[1])
        self.count_pixels = self.width * self.height
        self.T = int32(T)  # type: ignore
        self.inv_sigma = inv_sigma
        self.inv_gamma = inv_gamma
        self.cover_pixel = copy(cover)
        self.stego_noise = zeros((self.width, self.height), dtype=int8)
        self.cooc_diff = zeros(2 * (2 * T + 1) * (2 * T + 1) * (2 * T + 1), dtype=int)
        self.distortion = float64(0)


    def CD(self, type: int, d1: int, d2: int, d3: int) -> int32:
        """
        Вычисляет индекс элемента массива cooc_diff, моделирующего статистику совместных различий трех сосдедних пикселей.

        Parameters
        ----------
        type: int
            тип направления (горизонтальное/вертикальное)
        d1: int
            разность интенсивности между текущим и предыдущим пикселем
        d2: int
            разность интенсивности между предыдущим и предпредыдущим пикселем
        d3: int
            разность интенсивности между предпредыдущим и предпредпредыдущим пикселем
        
        Returns
        -------
        int32
            индекс элемента массива модели совместной встречаемости
        """

        assert d1 <= self.T
        assert d1 >= -self.T
        assert d2 <= self.T
        assert d2 >= -self.T
        assert d3 <= self.T
        assert d3 >= -self.T
        assert type >= 0
        assert type <= 1
        T2 = 2 * self.T + 1

        # в оригинальном алгоритме на C++ возвращает ссылку на элемент массива,
        # но в виду того, что python так не умеет ф. возрващает его индекс
        return type * T2 * T2 * T2 + (d1 + self.T) * T2 * T2 + (d2 + self.T) * T2 + (d3 + self.T)


    def weight(self, d1: int, d2: int, d3: int) -> float32:
        """
        Рассчет влияния пикселя на статистику локальных различий.

        Parameters
        ----------
        d1: int
            разность между текущим и предыдущим пикселем
        d2: int
            разность между предыдущим и предпредыдущим пикселем
        d3: int
            разность между предпредыдущим и предпредпредыдущим пикселем
        
        Returns
        -------
        float32
            влияние пикселя на искажение
        """

        y = float(d1 * d1 + d2 * d2 + d3 * d3)
        return float32(pow(sqrt(y) + float(self.inv_sigma), -self.inv_gamma))
    

    def set_stego_noise(self, i: int32, j: int32, value: int8) -> float64:
        """
        Устанавливает значение шума НЗБ для пикселя в заданную величину и расчитывает искажение.

        Parameters
        ----------
        i: int32
            координата i (строка) пикселя
        j: int32
            координата j (столбец) пикселя
        value: int8
            значение шума НЗБ для установки
        
        Returns
        -------
        float64
            искажение после установки шума НЗБ
        """

        cp = self.cover_pixel[j, i]
        assert (int(cp) + int(value) >= 0) and (int(cp) + int(value) <= 255)
        dirs = [0, 1, 0, -1, 1, 0, -1, 0, 1, 1, -1, -1, 1, -1, -1, 1]
        # расчет искажения до установки шума НЗБ
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

                in_range = \
                    (pix_i >= 0) and \
                    (pix_i < self.height) and \
                    (pix_j >= 0) and \
                    (pix_j < self.width)
                in_range &= \
                    (pix_i - 3 * dir1 >= 0) and \
                    (pix_i - 3 * dir1 < self.height) and \
                    (pix_j - 3 * dir2 >= 0) and \
                    (pix_j - 3 * dir2 < self.width)

                if in_range:
                    p0 = int(self.cover_pixel[pix_j, pix_i] + \
                             self.stego_noise[pix_j, pix_i])
                    p1 = int(self.cover_pixel[pix_j - 1 * dir2, pix_i - 1 * dir1] + \
                             self.stego_noise[pix_j - 1 * dir2, pix_i - 1 * dir1])
                    p2 = int(self.cover_pixel[pix_j - 2 * dir2, pix_i - 2 * dir1] + \
                             self.stego_noise[pix_j - 2 * dir2, pix_i - 2 * dir1])
                    p3 = int(self.cover_pixel[pix_j - 3 * dir2, pix_i - 3 * dir1] + \
                             self.stego_noise[pix_j - 3 * dir2, pix_i - 3 * dir1])
                    d1 = int(p0 - p1)
                    d2 = int(p1 - p2)
                    d3 = int(p2 - p3)

                    if (d1 >= -self.T) and (d1 <= self.T) and (d2 >= -self.T) and \
                        (d2 <= self.T) and (d3 >= -self.T) and (d3 <= self.T):
                        cd = self.CD(type, d1, d2, d3)
                        w = self.weight(d1, d2, d3)

                        if not self.cooc_diff[cd]:
                            self.distortion += w
                        elif self.cooc_diff[cd] < 0:
                            self.distortion -= sum_type * w
                        else:
                            self.distortion += sum_type * w

                        self.cooc_diff[cd] += sum_type

            self.stego_noise[j, i] = value

        return self.distortion


    def get_stego_pixel(self, i: uint32, j: uint32) -> uint8:
        """
        Получить значение пикселя покрывающего объекта с битом вложения (пикселя стеганограммы).
        
        Parameters
        ----------
        i: uint32
            координата i (строка) пикселя
        j: uint32
            координата j (столбец) пикселя
        
        Returns
        -------
        uint8
            значение пикселя покрывающего объекта с битом вложения (пикселя стеганограммы)
        """

        p = self.cover_pixel[i, j] + self.stego_noise[i, j]
        assert p >= 0 and p <= 255

        return uint8(p)


    def get_cover_pixel(self, i: uint32, j: uint32) -> uint8:
        """
        Получить значение пикселя покрывающего объекта.

        Parameters
        ----------
        i: uint32
            координата i (строка) пикселя
        j: uint32
            координата j (столбец) пикселя
        
        Returns
        -------
        uint8
            значение пикселя покрывающего объекта
        """

        return self.cover_pixel[j, i]
    


class HugoAlgBase(ABC):
    """
    Базовый абстрактный класс (скелет) для реализации вариантов HUGO.

    Attributes
    ----------
    model: HugoModel | None
        модель покрывающего объекта
    generator: MersenneTwister
        генератор случайных чисел
    pixel_perm: NDArray[uint32]
        случайные перестановки пикселей
    pixel_perm_inv: NDArray[uint32]
        обратные перестановки для реализации стратегии коррекции №1
    corr_strategy: int
        стратегия коррекции модели после погружения
    dist_min: NDArray[float64]
        минимальное влияние пикселя на искажение
    
    Methods
    -------
    get_image() -> NDArray[uint8] | None
        получить изображение с погруженным в него вложением
    embed_message()
        погружение вложения, и коррекция искажений, вызваынных погружением
    binary_embed(cover: NDArray[uint8], stego: NDArray[uint8])
        абстрактный метод, в котором реализуется алгоритм погружения вложения классами-потомками
    """

    model: HugoModel
    generator: MersenneTwister
    pixel_perm: NDArray[uint32]  # случайные перестановки пикселей
    pixel_perm_inv: NDArray[uint32]  # обратные перестановки для реализации стратегии коррекции №1
    corr_strategy: int
    dist_min: NDArray[float64]


    def __init__(
            self,
            cover: NDArray[uint8],
            T: uint32,
            inv_sigma: float32,
            inv_gamma: float32,
            seed: uint32,
            corr_strategy: int
    ):
        """
        Инициализация скелета алгоритма HUGO.

        Parameters
        ----------
        cover: NDArray[uint8]
            покрывающий объект
        T: uint32
            диапазон различий между соседними пикселями для построения Марковского процесса
        inv_sigma: float32
            параметр модели, управляющий общими искажениями
        inv_gamma: float32
            параметр модели, управляющий локальными искажениями
        seed: uint32
            зерно для инициализации ГСПЧ
        corr_strategy: int
            стратегия коррекции модели после погружения
        """
        
        # инициализация генератора случайных чисел на основе ключа
        self.generator = MersenneTwister(int(seed))
        # инициализируем модель покрывающего объекта
        self.model = HugoModel(cover, T, inv_sigma, inv_gamma)
        # стратегия коррекции модели после погружения
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
        Получить стеганограмму с погруженным в нее вложением.

        Returns
        -------
        NDArray[uint8] | None
            стеганограмма с погруженным в нее вложением
        """

        stego: NDArray[uint8] = zeros((self.model.width, self.model.height), dtype=uint8)
        
        for i in range(self.model.width):
            for j in range(self.model.height):
                stego[i, j] = self.model.get_stego_pixel(uint32(i), uint32(j))

        return stego


    def embed_message(self):
        """
        Погружение вложения, и коррекция искажений, вызваынных погружением.
        """

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
                dist_plus[i] = self.model.set_stego_noise(x, y, int8(+1))
            else:
                dist_plus[i] = inf

            if cp >= 1:
                dist_minus[i] = self.model.set_stego_noise(x, y, int8(-1))
            else:
                dist_minus[i] = inf

            assert (dist_plus[i] != inf) or (dist_minus[i] != inf)

            self.model.set_stego_noise(x, y, int8(0))
            self.dist_min[i] = dist_plus[i] if dist_plus[i] < dist_minus[i] else dist_minus[i]
            # отсекаем плоскость НЗБ, в cover теперь лежат биты НЗБ
            cover[i] = self.model.get_cover_pixel(x, y) % 2

        # погружение
        stego = copy(cover)
        self.binary_embed(stego)

        # коррекция искажений
        match self.corr_strategy:
            case 0:  # без коррекции модели, простая аддитивная апроксимация
                for i in range(self.model.count_pixels):
                    if cover[i] != stego[i]:
                        ip = self.pixel_perm[i]
                        x = ip % self.model.height
                        y = ip // self.model.height
                        if dist_plus[i] < dist_minus[i]:
                            self.model.set_stego_noise(x, y, int8(+1))
                        else:
                            self.model.set_stego_noise(x, y, int8(-1))

            case 1:  # стратегия коррекции начанющаяся от первого пикселя к последнему
                for i in range(self.model.count_pixels):
                    ip = self.pixel_perm_inv[i]
                    if cover[ip] != stego[ip]:
                        x = ip % self.model.height
                        y = ip // self.model.height
                        cp = self.model.get_cover_pixel(x, y)
                        d_plus, d_minus = inf, inf
                        if cp <= 254:
                            d_plus = self.model.set_stego_noise(x, y, int8(+1))
                        if cp >= 1:
                            d_minus = self.model.set_stego_noise(x, y, int8(-1))
                        if d_plus < d_minus:
                            self.model.set_stego_noise(x, y, int8(+1))

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
                    if cp <= 254:
                        d_plus = self.model.set_stego_noise(x, y, int8(+1))
                    if cp >= 1:
                        d_minus = self.model.set_stego_noise(x, y, int8(-1))
                    if d_plus < d_minus:
                        self.model.set_stego_noise(x, y, int8(+1))

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
                    if cp <= 254:
                        d_plus = self.model.set_stego_noise(x, y, int8(+1))
                    if cp >= 1:
                        d_minus = self.model.set_stego_noise(x, y, int8(-1))
                    if d_plus < d_minus:
                        self.model.set_stego_noise(x, y, int8(+1))

            case 4:  # случайная стратегия корекции
                for i in range(self.model.count_pixels):
                    if cover[i] != stego[i]:
                        ip = self.pixel_perm[i]
                        x = ip % self.model.height
                        y = ip // self.model.height
                        cp = self.model.get_cover_pixel(x, y)
                        d_plus, d_minus = inf, inf
                        if cp <= 254:
                            d_plus = self.model.set_stego_noise(x, y, int8(+1))
                        if cp >= 1:
                            d_minus = self.model.set_stego_noise(x, y, int8(-1))
                        if d_plus < d_minus:
                            self.model.set_stego_noise(x, y, int8(+1))

            case _:
                raise ValueError('This model correction strategy is not implemented.')


    @abstractmethod
    def binary_embed(self, stego: NDArray[uint8]):
        """
        Абстрактный метод, в котором реализуется алгоритм погружения вложения классами-потомками.
        Полученные в результате погружения биты стеганограммы должны быть помещены в свойство stego.
        """

        pass



class HugoAlgDet(HugoAlgBase):
    """
    Реализация HUGO с c погружением.

    Attributes
    ----------
    message_bits: NDArray[uint8] | None
        битовая вектор-строка вложения
    
    Methods
    -------
    binary_embed(cover: NDArray[uint8], stego: NDArray[uint8])
        реализация алгоритма погружения вложения
    """

    message_bits: NDArray[uint8]


    def __init__(
            self,
            cover: NDArray[uint8],
            message_bits: NDArray[uint8],
            T: uint32,
            inv_sigma: float32,
            inv_gamma: float32,
            seed: uint32,
            corr_strategy: int
    ):
        """
        Конструктор класса.

        Parameters
        ----------
        cover: NDArray[uint8]
            покрывающий объект
        message_bits: NDArray[uint8]
            битовая вектор-строка вложения
        T: uint32
            размер окна для модели совместной встречаемости
        inv_sigma: float32
            параметр модели, управляющий общими искажениями
        inv_gamma: float32
            параметр модели, управляющий локальными искажениями
        seed: uint32
            зерно для инициализации ГСПЧ
        corr_strategy: int
            стратегия коррекции модели после погружения
        """
        
        super().__init__(cover, T, inv_sigma, inv_gamma, seed, corr_strategy)
        self.message_bits = message_bits


    def binary_embed(self, stego: NDArray[uint8]):
        """
        Реализация алгоритма погружения вложения.

        Parameters
        ----------
        stego: NDArray[uint8]
            биты плоскости НЗБ стеганограммы
        """

        message_len = len(self.message_bits)
        
        for i in range(self.model.count_pixels):
            if i < message_len:
                stego[i] = self.message_bits[i]



class HugoAlgProb(HugoAlgBase):
    """
    Базовый абстарктный класс (скелет) для реализации вариантов HUGO с вероятностным встраиванием.

    Methods
    -------
    binary_embed(cover: NDArray[uint8], stego: NDArray[uint8])
        абстрактный метод, в котором реализуется алгоритм погружения вложения классами-потомками
    """

    def __init__(
            self,
            cover: NDArray[uint8],
            T: uint32,
            inv_sigma: float32,
            inv_gamma: float32,
            seed: uint32,
            corr_strategy: int
    ):
        
        super().__init__(cover, T, inv_sigma, inv_gamma, seed, corr_strategy)


    @abstractmethod
    def binary_embed(self, stego: NDArray[uint8]):
        """
        Абстрактный метод, в котором реализуется алгоритм погружения вложения классами-потомками.
        Полученные в результате погружения биты стеганограммы должны быть помещены в свойство stego.
        """

        pass



class HugoAlgSim(HugoAlgProb):
    """
    Реализация HUGO с симуляцией вероятностного встраивания.

    Attributes
    ----------
    rel_payload: float32
        процент заполнения покрывающего объекта полезной нагрузкой (битами вложения) при симуляции погружения
    
    Methods
    -------
    binary_embed(cover: NDArray[uint8], stego: NDArray[uint8])
        симуляция погружения вложения
    """

    rel_payload: float32


    def __init__(
            self,
            cover: NDArray[uint8],
            rel_payload: float32,
            T: uint32,
            inv_sigma: float32,
            inv_gamma: float32,
            seed: uint32,
            corr_strategy: int
    ):
        """
        Конструктор класса.

        Parameters
        ----------
        cover: NDArray[uint8]
            покрывающий объект
        rel_payload: float32
            процент заполнения покрывающего объекта полезной нагрузкой (битами вложения) при симуляции погружения
        T: uint32
            размер окна для модели совместной встречаемости
        inv_sigma: float32
            параметр модели, управляющий общими искажениями
        inv_gamma: float32
            параметр модели, управляющий локальными искажениями
        seed: uint32
            зерно для инициализации ГСПЧ
        corr_strategy: int
            стратегия коррекции модели после погружения
        """
        
        super().__init__(cover, T, inv_sigma, inv_gamma, seed, corr_strategy)
        self.rel_payload = rel_payload


    def binary_embed(self, stego: NDArray[uint8]):
        """
        Симуляция погружения вложения.

        Parameters
        ----------
        cover: NDArray[uint8]
            биты плоскости НЗБ покрывающего объекта
        stego: NDArray[uint8]
            биты плоскости НЗБ стеганограммы
        """

        weights = self.dist_min
        message_len = uint32(self.rel_payload * self.model.count_pixels)
        lmbd: float32 = calc_lambda_from_payload(message_len, weights, self.model.count_pixels)

        # симуляция погружения
        for i in range(self.model.count_pixels):
            flip_prob = exp(-lmbd * weights[i]) / (1 + exp(-lmbd * weights[i]))
            # иммитация замены бита плоскости НЗБ на бит вложения
            stego[i] = stego[i] ^ 1 if self.generator.random() < flip_prob else stego[i]



class HugoAlgProbEmbeder(HugoAlgProb):
    """
    Реализация HUGO с вероятностным погружением.

    Attributes
    ----------
    message_bits: NDArray[uint8] | None
        битовая вектор-строка вложения
    extraction_key: NDArray[uint8] | None
        список пикселей, в которых бит вложения не совпадает с битом покрывающего объекта (ключ извлечения)
    """

    message_bits: NDArray[uint8]
    extraction_key: NDArray[uint8]  # список пикселей покрывающего объекта, в которые погружен бит вложения (ключ извлечения)


    def __init__(
            self,
            cover: NDArray[uint8],
            message_bits: NDArray[uint8],
            T: uint32,
            inv_sigma: float32,
            inv_gamma: float32,
            seed: uint32,
            corr_strategy: int,
    ):
        """
        Конструктор класса.

        Parameters
        ----------
        cover: NDArray[uint8]
            покрывающий объект
        message_bits: NDArray[uint8]
            битовая вектор-строка вложения
        T: uint32
            размер окна для модели совместной встречаемости
        inv_sigma: float32
            параметр модели, управляющий общими искажениями
        inv_gamma: float32
            параметр модели, управляющий локальными искажениями
        seed: uint32
            зерно для инициализации ГСПЧ
        corr_strategy: int
            стратегия коррекции модели после погружения
        """
        
        super().__init__(cover, T, inv_sigma, inv_gamma, seed, corr_strategy)
        self.message_bits = message_bits


    def binary_embed(self, stego: NDArray[uint8]):
        """
        Реализация алгоритма погружения вложения.

        Parameters
        ----------
        cover: NDArray[uint8]
            биты плоскости НЗБ покрывающего объекта
        stego: NDArray[uint8]
            биты плоскости НЗБ стеганограммы
        """

        weights = self.dist_min
        message_len = uint32(len(self.message_bits))
        self.extraction_key = zeros(self.model.count_pixels, dtype=uint8)
        lmbd: float32 = calc_lambda_from_payload(message_len, weights, self.model.count_pixels)
        n = 0

        # погружение
        for i in range(self.model.count_pixels):
            if n < message_len:
                r = self.generator.random()
                flip_prob = exp(-lmbd * weights[i]) / (1 + exp(-lmbd * weights[i]))
                if self.message_bits[n] != stego[i]:
                    # в зависимости от влияния пикселя на искажение случайно решаем погружать бит вложения или нет
                    if r < flip_prob:
                        # погружаем бит вложения
                        stego[i] = stego[i] ^ 1
                        n += 1
                        # сохраняем ключ извлечения
                        self.extraction_key[i] = 1

                else:
                    # если бит вложения совпадает с битом покрывающего объекта, ничего не меняем
                    n += 1
                    # сохраняем ключ извлечения
                    self.extraction_key[i] = 1

            else:
                # запоминаем длину ключа извлечения
                n = i
                break

        self.extraction_key = self.extraction_key[:n]



class HugoAlgDetExtractor(object):
    """
    Реализация алгоритма извлечения вложения из НЗБ HUGO (High Undetectable steGO), использующего детерминированное встраивание

    Attributes
    ----------
    generator: MersenneTwister
        генератор случайных чисел
    model: HugoModel | None
        модель стеганограммы
    """

    generator: MersenneTwister
    model: HugoModel

    def __init__(
            self,
            stego: NDArray[uint8],
            seed: uint32
    ):
        """
        Конструктор класса.

        Parameters
        ----------
        stego: NDArray[uint8]
            стеганограмма с погруженным в нее вложением
        seed: uint32
            зерно для инициализации ГСПЧ
        """

        # инициализация генератора случайных чисел на основе ключа
        self.generator = MersenneTwister(int(seed))
        # инициализируем модель стеганограммы
        self.model = HugoModel(stego)

    def binary_extract(self) -> NDArray[uint8] | None:
        """
        Извлечение битовой вектор-строки вложения.

        Returns
        -------
        NDArray[uint8] | None
            битовая вектор-строка вложения
        """

        message_len = self.model.count_pixels
        message_bits: NDArray[uint8] = zeros(message_len, dtype=uint8)

        # генерируем перестановки
        pixel_perm = zeros(message_len, dtype=uint32)
        for i in range(message_len):
            pixel_perm[i] = i
        for i in range(message_len):
            j = self.generator.randint() % (message_len - i)
            pixel_perm[i + j], pixel_perm[i] = pixel_perm[i], pixel_perm[i + j]
        
        # извлечение
        for i in range(message_len):
            ip = pixel_perm[i]
            x = ip % self.model.height
            y = ip // self.model.height
            message_bits[i] = self.model.get_cover_pixel(x, y) % 2
        
        return message_bits



class HugoAlgProbExtractor(object):
    """
    Реализация алгоритма извлечения вложения из НЗБ HUGO (High Undetectable steGO), использующего вероятностное встраивание

    Attributes
    ----------
    generator: MersenneTwister
        генератор случайных чисел
    model: HugoModel | None
        модель стеганограммы
    extraction_key: NDArray[uint8] | None
        ключ извлечения вложения
    extraction_key_len: uint32
        длинна ключа извлечения вложения
    
    Methods
    -------
    binary_extract() -> NDArray[uint8] | None
        извлечение битовой вектор-строки вложения
    """

    generator: MersenneTwister
    model: HugoModel
    extraction_key: NDArray[uint8]
    extraction_key_len: uint32


    def __init__(
            self,
            stego: NDArray[uint8],
            extraction_key: NDArray[uint8],
            seed: uint32
    ):
        """
        Конструктор класса.

        Parameters
        ----------
        stego: NDArray[uint8]
            стеганограмма с погруженным в нее вложением
        extraction_key: NDArray[uint8]
            ключ извлечения вложения
        seed: uint32
            зерно для инициализации ГСПЧ
        """
        
        # инициализация генератора случайных чисел на основе ключа
        self.generator = MersenneTwister(int(seed))
        # инициализируем модель стеганограммы
        self.model = HugoModel(stego)
        # ключ извлечения вложения и его длинна
        self.extraction_key = extraction_key
        self.extraction_key_len = uint32(extraction_key.size)


    def binary_extract(self) -> NDArray[uint8] | None:
        """
        Извлечение битовой вектор-строки вложения.

        Returns
        -------
        NDArray[uint8] | None
            битовая вектор-строка вложения
        """

        # резервируем место под битовую вектор-строку вложения
        message_bits: NDArray[uint8] = zeros(self.extraction_key_len, dtype=uint8)
        stego: NDArray[uint8] = zeros(self.model.count_pixels, dtype=uint8)

        # генерируем перестановки
        pixel_perm = zeros(self.model.count_pixels, dtype=uint32)
        for i in range(self.model.count_pixels):
            pixel_perm[i] = i

        for i in range(self.model.count_pixels):
            j = self.generator.randint() % (self.model.count_pixels - i)
            pixel_perm[i + j], pixel_perm[i] = pixel_perm[i], pixel_perm[i + j]

        # переставляем пиксели изображения
        for i in range(self.model.count_pixels):
            ip = pixel_perm[i]
            x = ip % self.model.height
            y = ip // self.model.height
            # отсекаем плоскость НЗБ, в cover теперь лежат биты НЗБ
            stego[i] = self.model.get_cover_pixel(x, y) % 2
        
        # извлечение вложения
        n = 0
        for i in range(self.extraction_key_len):
            if self.extraction_key[i]:
                message_bits[n] = stego[i]
                n += 1

        return message_bits



class LSB_hugo_embedding(Embedder):
    """
    Реализация алгоритма погружения в НЗБ HUGO (High Undetectable steGO) (hugo)
    Получает из свойства родителя params параметр работы:
    {'action_type': 'sim' | 'prob' | 'det'}
        тип действия:
        'sim' - симуляция встраивания сообщения в покрывающий объект с использованием алгоритма HUGO
        'prob' - встраивание сообщения в покрывающий объект с использованием алгоритма HUGO с вероятностным встраиванием
        'det' - встраивание сообщения в покрывающий объект с использованием алгоритма HUGO с детерминированным встраиванием
    {'T': 90}
        диапазон разности между соседними пикселями
    {'inv_sigma': 1}
        управляет общими искажениями (насколько осторожно работать с покрывающим объектом)
    {'inv_gamma': 1}
        управляет локальными искажениями (как сильно штрафовать за изменения в данном месте)
    {'seed': 42}
        зерно для инициализации ГСПЧ
    {'corr_strategy'}
        стратегия коррекции модели после погружения
        0 - без коррекции модели, простая аддитивная апроксимация
        1 - стратегия коррекции начанющаяся от первого пикселя к последнему
        2 - стратегия коррекции начанющаяся от пикселя с максимальным искажением к пикселю с минимальным
        3 - стратегия коррекции начанющаяся от пикселя с минимальным искажением к пикселю с максимальным
        4 - случайная стратегия корекции
    {'rel_payload': 0.5}
        относительная полезная нагрузка для симуляции встраивания (только для симуляции)
    """

    def embedding(self):
        # получаем параметры работы алгоритма
        action_type = (self.params or {}).get('action_type', default_action_type)
        T = (self.params or {}).get('T', default_T)
        inv_sigma = (self.params or {}).get('inv_sigma', default_inv_sigma)
        inv_gamma = (self.params or {}).get('inv_gamma', default_inv_gamma)
        seed = (self.params or {}).get('seed', default_seed)
        corr_strategy = (self.params or {}).get('corr_strategy', default_corr_strategy)
        rel_payload = (self.params or {}).get('rel_payload', default_rel_payload)

        # соединяем двумерные цветовые плоскости в один двумерный массив
        cover_arr, *widths = img_arr_surfs_to_one_arr(self.cover_object)
        if cover_arr is None or self.message_bits is None: return

        match action_type:
            case 'sim':  # симуляция встраивания сообщения в покрывающий объект с использованием алгоритма HUGO
                hugo = HugoAlgSim(cover_arr, float32(rel_payload), uint32(T), float32(inv_sigma), float32(inv_gamma), uint32(seed), corr_strategy)
            case 'prob':  # встраивание сообщения в покрывающий объект с использованием алгоритма HUGO с вероятностным встраиванием
                hugo = HugoAlgProbEmbeder(cover_arr, self.message_bits, uint32(T), float32(inv_sigma), float32(inv_gamma), uint32(seed), corr_strategy)
            case 'det':  # встраивание сообщения в покрывающий объект с использованием алгоритма HUGO с детерминированным встраиванием
                hugo = HugoAlgDet(cover_arr, self.message_bits, uint32(T), float32(inv_sigma), float32(inv_gamma), uint32(seed), corr_strategy)
            case _:
                raise ValueError('This action type is not implemented.')

        hugo.embed_message()
        
        if hasattr(hugo, 'extraction_key'):
            if hugo.extraction_key is not None:  # type: ignore
                self.extraction_key = hugo.extraction_key  # type: ignore

        stego_arr = hugo.get_image()
        if stego_arr is None: return
        self.stego_object = one_arr_to_img_arr_surfs(stego_arr, *widths)



class LSB_hugo_extracting(Extractor):
    """
    Реализация алгоритма извлечения из НЗБ HUGO (High Undetectable steGO) (hugo)
    Получает из свойства родителя params параметр работы:
    {'action_type': 'prob' | 'det'}
        тип действия:
        'prob' - извлечение сообщения из стеганограммы с использованием алгоритма HUGO с вероятностным встраиванием
        'det' - извлечение сообщения из стеганограммы с использованием алгоритма HUGO с детерминированным встраиванием
    {'seed': 42}
        зерно для инициализации ГСПЧ (только для детерминированного извлечения)
    """

    def extracting(self):
        if self.stego_object is None: return
        # получаем параметры работы алгоритма
        seed = (self.params or {}).get('seed', default_seed)
        action_type = (self.params or {}).get('action_type', default_action_type)

        stego_arr = img_arr_surfs_to_one_arr(self.stego_object)[0]
        if stego_arr is None: return

        match action_type:
            case 'prob':  # вероятностное извлечение
                if self.extraction_key is None: return
                hugo_extractor = HugoAlgProbExtractor(stego_arr, self.extraction_key, seed)
                self.message_bits = hugo_extractor.binary_extract()
            case 'det':  # детерминированное извлечение
                hugo_extractor = HugoAlgDetExtractor(stego_arr, seed)
                self.message_bits = hugo_extractor.binary_extract()
            case _:
                raise ValueError('This action type is not implemented.')



if __name__ == '__main__':
    sys.exit()
