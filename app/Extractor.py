import os
import sys
from typing import Any
from abc import ABC, abstractmethod

from PIL import Image
from numpy.typing import NDArray
from numpy import uint8, asarray

from .utils import from_bit_vector, bytes2chars, chars2bytes


# метки начала и конца места погружения вложения в покрывающий объект
default_start_label: str = 'H@4@l0'
default_end_label: str = 'k0HEU'


class Extractor(ABC):
    """
    Абстрактный класс для извлечения вложения из стеганограммы.
    """

    stego_object: NDArray[uint8] | None = None
    message_object: NDArray[uint8] | None = None
    message_bits: NDArray[uint8] | None = None
    message_file_name: str | None = None
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


    def load_stego_file(self, stego_file_path: str):
        """
        Загружает файл стеганограммы и помещает в свойство stego_object.

        Parameters
        ----------
        stego_file_path: str
            имя/путь к стеганограмме
        """

        with Image.open(stego_file_path, 'r') as F:
            self.stego_object = asarray(F, dtype=uint8)


    @abstractmethod
    def extracting(self):
        """
        Абстрактный метод, в котором реализуется алгоритм извлечения вложения классами-потомками.
        Полученные в результате извлечения биты вложения должны быть помещены в свойство message_bits.
        """

        pass


    def extract_message_object(self) -> bool:
        """
        Преобразует битовую вектор-строку message_bits с битами вложения в байтовую вектор-строку,
        Затем обрезает метки начала и конца места погружения. Извлекает имя файла вложения,
        сохраняет его в свойство message_file_name. Сохраняет байты вложения в message_object.
        """

        start_label = default_start_label
        if self.params and 'start_label' in self.params:
            start_label = self.params['start_label']
        end_label = default_end_label
        if self.params and 'end_label' in self.params:
            end_label = self.params['end_label']
        if self.message_bits is None:
            return False
        message_bytes = from_bit_vector(self.message_bits)
        message = bytes2chars(message_bytes)
        message = message[message.find(start_label) + len(start_label):message.find(end_label)]
        message_file_name_len = chars2bytes(message[0])[0]
        self.message_file_name = message[1:message_file_name_len + 1]
        self.message_object = chars2bytes(message[message_file_name_len + 1:])
        return True
    

    def save_message_file(self, extract_file_path: str='.') -> bool:
        """
        Сохраняет вложение из свойства message_object в файл с именем message_file_name.
        В директории extract_file_path.

        Parameters
        ----------
        extract_file_path: str
            путь к директории для сохранения вложений
        """

        if not self.message_file_name:
            return False
        message_file_path = os.path.join(extract_file_path, self.message_file_name)
        if self.message_object is None:
            return False
        with open(message_file_path, 'bw') as F:
            F.write(self.message_object)
        return True
    

    def process_one_file(self, strgo_file_path: str, extract_file_path: str, **params: dict[str, Any]):
        """
        Извлечь одно вложение из одной стеганограммы.

        Parameters
        ----------
        strgo_file_path: str
            имя/путь к файлу стеганограммы
        extract_file_path: str
            путь к файлу вложения (только директория)
        **params: dict[str, Any]
            параметры извлечения вложения, зависят от используемого алгоритма.
            У всех алгоритмов есть параметры:
            - start_label: str = 'H@4@l0' - метка начала места погружения;
            - end_label: str = 'k0HEU' - метка конца места погружения.
        """

        self.set_params(**params)
        self.load_stego_file(strgo_file_path)
        self.extracting()
        if not self.extract_message_object(): return
        self.save_message_file(extract_file_path)


if __name__ == '__main__':
    sys.exit()
