# извлечение сообщения из НЗБ пикселей изображения после НЗБ встраивания с псевдослучайным интервалом (LSB PRI)

from PIL import Image
from numpy import asarray, uint8, hstack, concatenate, empty, array, set_printoptions, inf
from utils import step, D2B, B2D

set_printoptions(threshold=inf)
set_printoptions(linewidth=inf)

def LSB_PRI_extracting(S, Ms, Me, k):
    # цветовые компоненты изображения
    Sr = S[:, :, 0]
    Sg = S[:, :, 1]
    Sb = S[:, :, 2]

    # соединяем матрицы цветов бок в бок друг за другом
    S_ = hstack((Sb, Sg, Sr))
    # выстраиваем получившуюся матрицу в вектор построчно
    Sv = concatenate(S_)

    z = Ms
    b = empty(8, dtype=uint8)
    i = 0
    M = []
    while z <= len(Sv):
        b.fill(0)
        for j in range(8):
            P = D2B(Sv[z])
            b[j] = P[0]
            z += step(D2B(z), k)
            if z > len(Sv):
                break
        M.append(B2D(b))
        i += 1

    M = ''.join([chr(m) for m in M])
    M = M[:M.find(Me)]

    return M


if __name__ == '__main__':
    stg = 'assets/stego.bmp'  # изображение, в которое встроено сообщение
    # ограничивающие метки начала и конца сообщения в файле изображения
    Ms = 54  # стартовая позиция с которой начинается встраивание (номер пикселя)
    Me = 'KiHeu,6'  # маркер окончания встроенного сообщения
    k = 10  # масштабирующий коэффициент для шага псевдослучайного интервала

    with Image.open(stg) as F:
        S = asarray(F, dtype=uint8)


    # извллекаем сообщение из изображения
    M = LSB_PRI_extracting(S, Ms, Me, k)


    print(M)
