import os
import sys
from math import sqrt, exp, log

from PIL import Image
from numpy import asarray, uint8, fromfile, concatenate, zeros, uint32, inf, float32, copy, ndarray, int16, int32, float64, uint32, finfo

from .utils import chars2bytes, to_bit_vector, MersenneTwister


default_T: uint32 = 90
default_inv_sigma: float32 = 1
default_inv_gamma: float32 = 1
default_seed: uint32 = 42


class HugoModel(object):
    """
    
    """

    width: uint32 = 0
    height: uint32 = 0
    n: uint32 = 0
    T: uint32 = 0
    inv_sigma: float32 = 0
    inv_gamma: float32 = 0
    cover_pixel: ndarray
    stego_noise: ndarray
    cooc_diff: ndarray
    distortion: float64 = 0


    def __init__(self, cover: ndarray[uint8], T: uint32, inv_sigma: float32, inv_gamma: float32):
        self.width = cover.shape[0]
        self.height = cover.shape[1]
        self.n = self.width * self.height
        self.T = T
        self.inv_sigma = inv_sigma
        self.inv_gamma = inv_gamma
        self.cover_pixel = copy(cover)
        self.stego_noise = zeros((self.width, self.height), dtype=int16)
        self.cooc_diff = zeros(2 * (2 * T + 1) * (2 * T + 1) * (2 * T + 1), dtype=int)
        self.distortion = 0


    def CD(self, type: int, d1: int, d2: int, d3: int) -> int:
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


    def weight(self, d1, d2, d3):
        y = float(d1 * d1 + d2 * d2 + d3 * d3)
        return pow(sqrt(y) + self.inv_gamma, -self.inv_gamma)
    

    def set_stego_noise(self, i, j, value) -> float:
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
                for shift in range(4):
                    pix_i = int(i) + shift * dir1
                    pix_j = int(j) + shift * dir2
                in_range = (pix_i >= 0) and (pix_i < self.height) and (pix_j >= 0) and (pix_j < self.width)
                in_range &= (pix_i - 3 * dir1 >= 0) and (pix_i - 3 * dir1 < self.height) and (pix_j - 3 * dir2 >= 0) and(pix_j - 3 * dir2 < self.width)
                if in_range:
                    p0 = int(self.cover_pixel[pix_i, pix_j] + self.stego_noise[pix_i, pix_j])
                    p1 = int(self.cover_pixel[pix_i - 1 * dir1, pix_j - 1 * dir2] + self.stego_noise[pix_i - 1 * dir1, pix_j - 1 * dir2])
                    p2 = int(self.cover_pixel[pix_i - 2 * dir1, pix_j - 2 * dir2] + self.stego_noise[pix_i - 2 * dir1, pix_j - 1 * dir2])
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


    def get_stego_pixel(self, i: uint32, j: uint32):
        p = self.cover_pixel[i, j] + self.stego_noise[i, j]
        assert p >= 0 and p <= 255
        return uint8(p)


    def get_cover_pixel(self, i: uint32, j: uint32):
        return self.cover_pixel[i, j]
    

    def get_distortion(self):
        return self.distortion



class HugoAlgSimulator(object):
    """
    
    """

    m: HugoModel
    width: uint32
    height: uint32
    n: uint32
    generator: MersenneTwister
    pixel_perm: ndarray[uint32]
    pixel_perm_inv: ndarray[uint32]


    def __init__(self, cover: ndarray[uint8], T: uint32, inv_sigma: float32, inv_gamma: float32, seed: uint32):
        self.width = cover.shape[1]
        self.height = cover.shape[0]
        self.n = self.width * self.height
        # инициализация генератора случайных чисел на основе ключа
        self.generator = MersenneTwister(seed)
        self.m = HugoModel(cover, T, inv_sigma, inv_gamma)
        self.pixel_perm = zeros(self.n, dtype=uint32)

        # генерируем перестановки
        for i in range(self.n):
            self.pixel_perm[i] = i
        for i in range(self.n):
            j = self.generator.randint() % (self.n - i)
            self.pixel_perm[i + j], self.pixel_perm[i] = self.pixel_perm[i], self.pixel_perm[i + j]
        # генерируем обратные перестановки
        self.pixel_perm_inv = zeros(self.n, dtype=uint32)
        for i in range(self.n):
            self.pixel_perm_inv[self.pixel_perm[i]] = i


    def embed_random_message(self, rel_payload: list[float32], corr_strategy: uint32, n_changes: list[uint32], distortion: list[float32]):
        dist_plus: ndarray[float32] = zeros(self.n, dtype=float32)
        dist_minus: ndarray[float32] = zeros(self.n, dtype=float32)
        dist_dir: ndarray[int32] = zeros(self.n, dtype=int32)
        dist_min: ndarray[float64] = zeros(self.n, dtype=float64)
        cover: ndarray[uint8] = zeros(self.n, dtype=uint8)
        stego: ndarray[uint8] = zeros(self.n, dtype=uint8)
        message_length: uint32 = uint32(rel_payload[0] * self.n)
        v: list[tuple[uint32, float32]] = []

        n_changes[0] = 0
        for i in range(self.n):
            ip = self.pixel_perm[i]
            cp = self.m.get_cover_pixel(ip % self.height, ip // self.height)  # ip % height == y, ip // height == x
            if cp <= 254:
                dist_plus[i] = self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
            else:
                dist_plus[i] = inf
            if cp >= 1:
                dist_minus[i] = self.m.set_stego_noise(ip % self.height, ip // self.height, -1)
            else:
                dist_minus[i] = inf
            assert (dist_plus[i] != inf) or (dist_minus[i] != inf)
            self.m.set_stego_noise(ip % self.height, ip // self.height, 0)
            dist_dir[i] = +1 if dist_plus[i] < dist_minus[i] else -1
            dist_min[i] = dist_plus[i] if dist_plus[i] < dist_minus[i] else dist_minus[i]
            # отсекаем плоскость НЗБ, в cover теперь лежат биты НЗБ
            cover[i] = self.m.get_cover_pixel(ip % self.height, ip // self.height) % 2
            
        self.binary_embed(message_length, cover, stego, dist_min, rel_payload)

        d_plus: float32 = 0
        d_minus: float32 = 0

        match corr_strategy:
            case 0:
                for i in range(self.n):
                    if cover[i] != stego[i]:
                        ip = self.pixel_perm[i]
                        n_changes[0] += 1
                        if dist_plus[i] < dist_minus[i]:
                            self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
                        else:
                            self.m.set_stego_noise(ip % self.height, ip // self.height, -1)
            case 1:
                for i in range(self.n):
                    ip = self.pixel_perm_inv[i]
                    if cover[ip] != stego[ip]:
                        n_changes[0] += 1
                        cp = self.m.get_cover_pixel(ip % self.height, ip // self.height)
                        d_plus = inf
                        d_minus = inf
                        if cp <= 254:
                            d_plus = self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
                        if cp >= 1:
                            d_minus = self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
                        if d_plus < d_minus:
                            self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
            case 2:
                for i in range(self.n):
                    if cover[i] != stego[i]:
                        v.append((i, dist_min[i]))
                v.sort(key=lambda pair: pair[1], reverse=True)
                for i in range(len(v)):
                    ip = self.pixel_perm[v[i][0]]
                    n_changes[0] += 1
                    cp = self.m.get_cover_pixel(ip % self.height, ip // self.height)
                    d_plus = inf
                    d_minus = inf
                    if cp <= 254:
                        d_plus = self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
                    if cp >= 1:
                        d_minus = self.m.set_stego_noise(ip % self.height, ip // self.height, -1)
                    if d_plus < d_minus:
                            self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
            case 3:
                for i in range(self.n):
                    if cover[i] != stego[i]:
                        v.append((i, float32(dist_min[i])))
                v.sort(key=lambda pair: pair[1])
                for i in range(len(v)):
                    ip = self.pixel_perm[v[i][0]]
                    n_changes[0] += 1
                    cp = self.m.get_cover_pixel(ip % self.height, ip // self.height)
                    d_plus = inf
                    d_minus = inf
                    if cp <= 254:
                        d_plus = self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
                    if cp >= 1:
                        d_minus = self.m.set_stego_noise(ip % self.height, ip // self.height, -1)
                    if d_plus < d_minus:
                            self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
            case 4:
                for i in range(n):
                    if cover[i] != cover[i]:
                        ip = self.pixel_perm[i]
                        n_changes[0] += 1
                        cp = self.m.get_cover_pixel(ip % self.height, ip // self.height)
                        d_plus = inf
                        d_minus = inf
                        if cp <= 254:
                            d_plus = self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
                        if cp >= 1:
                            d_minus = self.m.set_stego_noise(ip % self.height, ip // self.height, -1)
                        if d_plus < d_minus:
                                self.m.set_stego_noise(ip % self.height, ip // self.height, +1)
            case _:
                raise ValueError('This model correction strategy is not implemented.')
            
        distortion[0] = self.m.get_distortion()


    def binary_embed(self, message_length: uint32, cover: ndarray[uint8], stego: ndarray[uint8], weights: ndarray[float64], rel_payload: list[float32]):
        rel_payload[0] = 0
        lmbd: float32 = self.calc_lambda_from_payload(message_length, weights, self.n)
        for i in range(self.n):
            flip_prob = exp(-lmbd * weights[i]) / (1 + exp(-lmbd * weights[i]))
            # иммитация замены бита плоскости НЗБ на бит вложения
            # TODO сделать на замену битами реального сообщения
            stego[i] = cover[i] ^ 1 if self.generator.random() < flip_prob else cover[i]
            rel_payload[0] += self.binary_entropyf(flip_prob)
        rel_payload[0] /= self.n


    def calc_lambda_from_payload(self, message_length: uint32, weigts: ndarray[float64], n: uint32) -> float32:
        
        l1: float32 = 0
        l2: float32 = 0
        l3: float32 = 1000.0
        m1: float32 = float32(n)
        m2: float32 = 0
        m3: float32 = float32(message_length + 1)
        j: int32 = 0
        iterations: uint32 = 0

        while m3 > message_length:
            l3 *= 2
            m3 = 0
            for i in range(n):
                m3 += self.binary_entropyf(1 / (1 + exp(-l3 * weigts[i])))
            j += 1
            if j > 10:
                return l3
            iterations += 1
        alpha: float32 = message_length / n
        while ((m1 - m3) / n > alpha / 1000.0) and (iterations < 30):
            l2 = l1 + (l3 - l1) / 2
            m2 = 0
            for i in range(n):
                m2 += self.binary_entropyf(1 / (1 + exp(-l2 * weigts[i])))
            if m2 < message_length:
                l3 = l2
                m3 = m2
            else:
                l1 = l2
                m1 = m2
            iterations += 1
        return l2


    def binary_entropyf(self, x: float32) -> float32:
        LOG2 = log(2.0)
        EPS = finfo(float32).eps
        z: float32 = 0
        if (x < EPS) or ((1 - x) < EPS):
            return 0
        else:
            z = (-x * log(x) - (1 - x) * log(1 - x)) / LOG2
            return z
        

    def get_image(self) -> ndarray[uint8]:
        stego: ndarray[uint8] = zeros((self.width, self.height), dtype=uint8)
        for i in range(self.width):
            for j in range(self.height):
                stego[i, j] = self.m.get_stego_pixel(i, j)
        return stego
    


def LSB_hugo_embedding(
        cover_file_path: str,
        stego_file_path: str,
        message_file_path: str,
        T: int = default_T,
        inv_sigma: float = default_inv_sigma,
        inv_gamma: float = default_inv_gamma,
        seed: int = default_seed
    ):
    """
    
    
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
    message_file_name_bytes = chars2bytes(message_file_name)
    message_file_name_bytes_len = asarray([len(message_file_name_bytes)])
    # собираем все в одину вектор-строку байт
    message_bytes = concatenate((
        message_file_name_bytes_len,
        message_file_name_bytes,
        message_object))
    # преобразуем вктор-строку байт в вектор-строку бит
    message_bits = to_bit_vector(message_bytes)
    message_len = len(message_bits)

    # соединяем двумерные цветовые плоскости в один двумерный массив
    # cover_arr = hstack((cover_object))  # TODO распространить на несколько цветовых плоскостей

    rel_payload: list[float32] = [1]
    corr_strategy: uint32 = 2
    n_changes: list[uint32] = [0]
    distortion: list[float32] = [0]



    hugo = HugoAlgSimulator(cover_object, T, inv_sigma, inv_gamma, seed)
    hugo.embed_random_message(rel_payload=rel_payload, corr_strategy=corr_strategy, n_changes=n_changes, distortion=distortion)

    stego_object = hugo.get_image()
    

    with Image.fromarray(stego_object) as F:
        F.save(stego_file_path)


if __name__ == '__main__':
    sys.exit()