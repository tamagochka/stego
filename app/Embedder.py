import os
import sys
from typing import Any
from abc import ABC, abstractmethod


from PIL import Image
from numpy.typing import NDArray
from numpy import asarray, uint8, fromfile, concatenate, packbits

from .utils import chars2bytes, to_bit_vector
from .config import default_start_label, default_end_label


class Embedder(ABC):
    """
    Абстрактный класс для погружения вложения в покрывающий объект.
    """
    
    cover_object: NDArray[uint8] | None = None
    message_object: NDArray[uint8] | None = None
    stego_object: NDArray[uint8] | None = None
    extraction_key: NDArray[uint8] | None = None
    message_file_name: str | None = None
    message_bits: NDArray[uint8] | None = None
    params: dict[str, Any] | None = None


    def set_params(self, **params: dict[str, Any]):
        """
        Устанавливает параметры погружения вложения.

        Parameters
        ----------
        **params: dict[str, Any]
            словарь с параметрами. Абстрактный класс Embedder обрабатывает только параметры:
            - {'start_label': 'H@4@l0'} - метка начала места погружения;
            - {'end_label': 'k0HEU'} - метка конца места погружения.
            Другие возможные параметры обрабатываются классами потомками, реализующими конкретный алгоритм погружения.
        """

        self.params = params


    def load_cover_file(self, cover_file_path: str):
        """
        Загружает файл покрывающего объекта и помещает в свойство cover_object.

        Parameters
        ----------
        cover_file_path: str
            имя/путь к покрывающему объекту
        """

        with Image.open(cover_file_path) as F:
            self.cover_object = asarray(F, dtype=uint8)


    def load_message_file(self, message_file_path: str):
        """
        Загружает файл вложения и помещает в свойство message_object.

        Parameters
        ----------
        message_file_path: str
            имя/путь к файлу вложения
        """

        with open(message_file_path, 'rb') as F:
            self.message_object = fromfile(F, dtype=uint8)
            self.message_file_name = os.path.basename(F.name)

    
    def prepare_message_object(self) -> bool:
        """
        Добавляет к объекту вложения (свойство message_object) имя файла вложения (свойство message_file_name) и длинну имени файла,
        Также сначала и сконца полученной строки добавляет метки начала и конца места погружения.
        Затем преобразует полученную байтовую вектор-строку в битовую вектор-строку и помещает в message_bits.
        message_bits затем методом embedding погружается в покрывающий объект.
        """

        if self.message_file_name is None or self.message_object is None:
            return False
        # преобразуем метки начала и конца места погружения,
        # а также имя файла вложения и длинну его имени в байтовые вектор-строки
        start_label = (self.params or {}).get('start_label', default_start_label)
        start_label_bytes = chars2bytes(start_label)
        end_label = (self.params or {}).get('end_label', default_end_label)
        end_label_bytes = chars2bytes(end_label)
        message_file_name_bytes = chars2bytes(self.message_file_name)
        message_file_name_bytes_len = asarray([len(message_file_name_bytes)])
        # собираем все в одину вектор-строку байт
        message_bytes = concatenate((
            start_label_bytes,
            message_file_name_bytes_len,
            message_file_name_bytes,
            self.message_object,
            end_label_bytes
        ))
        # преобразуем вктор-строку байт в вектор-строку бит
        self.message_bits = to_bit_vector(message_bytes)
        return True


    @abstractmethod
    def embedding(self):
        """
        Абстрактный метод, в котором реализуется алгоритм погружения вложения классами-потомками.
        Полученная в результате работы метода стаганограмма должна быть помещена в свойство stego_object.
        """

        pass


    def save_stego_file(self, stego_file_path: str):
        """
        Сохраняет полученную стеганограмму из свойства stego_object в файл.

        Parameters
        ----------
        stego_file_path: str
            имя/путь к файлу стеганограммы
        """

        if self.stego_object is not None:
            with Image.fromarray(self.stego_object) as F:
                F.save(stego_file_path)


    def save_key_file(self, key_file_path: str):
        """
        Сохраняет ключ, необходимый для извлечения вложения из стеганограммы в файл.

        Parameters
        ----------
        key_file_path: str
            имя/путь к файлу ключа
        """
        
        # сохранение ключа в бинарный файл
        if self.extraction_key is not None:
            packbits(self.extraction_key).tofile(key_file_path)


    def process_one_file(
            self,
            cover_file_path: str,
            stego_file_path: str,
            message_file_path: str | None,
            key_file_path: str | None = None,
            **params: dict[str, Any]
    ):
        """
        Погрузить одно вложение в один файл.

        Parameters
        ----------
        cover_file_path: str
            имя/путь к покрывающему объекту, в который будет погружено вложения
        stego_file_path: str
            имя/путь к стеганограмме, результату погружения вложения в покрывающий объект
        message_file_path: str
            имя/путь к файлу вложения
        key_file_path: str
            имя/путь к файлу с ключем, необходимым для извлечения вложения из стеганограммы
        **params: dict[str, Any]
            параметры погружения вложения, зависят от используемого алгоритма
            У всех алгоритмов есть параметры:
            - start_label: str = 'H@4@l0' - метка начала места погружения;
            - end_label: str = 'k0HEU' - метка конца места погружения.
        """

        self.set_params(**params)
        self.load_cover_file(cover_file_path)
        if message_file_path is not None:
            self.load_message_file(message_file_path)
            if not self.prepare_message_object(): return
        self.embedding()
        self.save_stego_file(stego_file_path)
        if self.extraction_key is not None and key_file_path is not None:
            self.save_key_file(key_file_path)


if __name__ == '__main__':
    sys.exit()
