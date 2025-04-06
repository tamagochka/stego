# извлечение сообщения из НЗБ пикселей после блочного встраивания
import numpy
from PIL import Image
from numpy import asarray, set_printoptions, inf, uint8, hstack, empty
from utils import D2B, B2D
from math import floor

set_printoptions(threshold=inf)
set_printoptions(linewidth=inf)

def block_LSB_extracting(S):
    # цветовые компоненты изображения
    Sr = S[:, :, 0]
    Sg = S[:, :, 1]
    Sb = S[:, :, 2]

    S_ = hstack((Sb, Sg, Sr))

    X, Y = S_.shape
    Xa = S_[S_.shape[0] - 1, S_.shape[1] - 1]
    print('Xa: ', Xa)

    Nm = Y * Xa

    Mb = empty(Nm, dtype=uint8)

    for i in range(Xa):
        r1 = i * floor(X / Xa)
        r2 = (i + 1) * floor(X / Xa)
        for y in range(Y):
            d = S_[r1:r2, y]
            b = 0
            for x in range(len(d)):
                P = D2B(d[x])
                LSB = P[0]
                b = b ^ LSB
            Mb[y + i * Y] = b

    M = empty(Nm // 8, dtype=uint8)
    for i in range(Nm // 8):
        M[i] = B2D(Mb[i * 8: i * 8 + 8])
    M = ''.join([chr(m) for m in M])

    return M


if __name__ == '__main__':
    stg = 'assets/stego.bmp'

    with Image.open(stg) as F:
        S = asarray(F, dtype=uint8)
    M = block_LSB_extracting(S)

    print(M)