# извлечение сообщения из НЗБ пикселей после встраивания с псевдослучайной перестановкой

from PIL import Image
from numpy import asarray, set_printoptions, inf, uint8, hstack, empty
from math import floor
from utils import D2B, B2D, keypairgen

set_printoptions(threshold=inf)
set_printoptions(linewidth=inf)


def LSB_PRP_extracting(S, Ko, K):
    # цветовые компоненты изображения
    Sr = S[:, :, 0]
    Sg = S[:, :, 1]
    Sb = S[:, :, 2]

    # соединяем матрицы цветов бок в бок друг за другом
    S_ = hstack((Sb, Sg, Sr))

    X, Y = S_.shape
    N = X * Y
    KT = keypairgen(Ko, K)  # генерируем пары ключей

    Mb = empty(N, dtype=uint8)

    for i in range(N):
        x = floor(i / Y)
        y = i % Y
        for s in range(K):
            x = (x + B2D(D2B(KT[2 * s - 1]) ^ D2B(y))) % X
            y = (y + B2D(D2B(KT[2 * s]) ^ D2B(x))) % Y
        P = D2B(S_[x, y])
        Mb[i] = P[0]

    M = empty(N // 8, dtype=uint8)
    for i in range(N // 8):
        M[i] = B2D(Mb[i * 8: i * 8 + 8])

    M = ''.join([chr(m) for m in M])

    return M


if __name__ == '__main__':
    stg = 'assets/stego.bmp'  # изображение, в которое встроено сообщение
    Ko = 125  # ключ
    K = 10  # количество генерируемых пар ключей

    with Image.open(stg) as F:
        S = asarray(F, dtype=uint8)

    M = LSB_PRP_extracting(S, Ko, K)

    print(M)