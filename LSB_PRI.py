from random import seed, randint
from numpy import copy, empty, uint8
from utils import D2B, B2D


def LSB_PRI_embedding(cover, message_bits, key, min_step=1, max_step=8):
    """

    :param cover:
    :param message_bits:
    :param key:
    :param min_step:
    :param max_step:
    :return:
    """

    message_len = len(message_bits)
    stego = copy(cover)
    cover_len = len(cover)

    seed(key)

    pointer = randint(min_step, max_step)
    i = 0
    while pointer <= cover_len:
        b = D2B(cover[pointer])
        b[0] = message_bits[i]
        stego[pointer] = B2D(b)
        i += 1
        pointer += randint(min_step, max_step)

    return stego


def LSB_PRI_extracting(stego, key, min_step=1, max_step=8):
    """

    :param stego:
    :param key:
    :param min_step:
    :param max_step:
    :return:
    """

    stego_len = len(stego)
    message_bits = empty(stego_len, dtype=uint8
    seed(key)

    pointer = randint(min_step, max_step)
    i = 0
    while pointer <= stego_len:
        b = D2B(stego[pointer])
        message_bits[i] = b[0]
        i += 1
        pointer += randint(min_step, max_step)

    return message_bits



