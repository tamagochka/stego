import os
from typing import Any
from abc import ABC, abstractmethod


from PIL import Image
from numpy.typing import NDArray
from numpy import asarray, uint8, fromfile, concatenate

from .utils import chars2bytes, to_bit_vector


# метки начала и конца места погружения вложения в покрывающий объект
default_start_label: str = 'H@4@l0'
defualt_end_label: str = 'k0HEU'


class Embedder(ABC):
    
    cover_object: NDArray[uint8] | None = None
    message_object: NDArray[uint8] | None = None
    stego_object: NDArray[uint8] | None = None

    message_file_name: str | None = None
    message_bits: NDArray[uint8] | None = None
    message_len: int | None = None

    params: dict[str, Any] | None = None


    def load_cover_file(self, cover_file_path: str):
        with Image.open(cover_file_path) as F:
            self.cover_object = asarray(F, dtype=uint8)


    def load_message_file(self, message_file_path: str):
        with open(message_file_path, 'rb') as F:
            self.message_object = fromfile(F, dtype=uint8)
            self.message_file_name = os.path.basename(F.name)

    
    def set_params(self, **params):
        self.params = params


    def prepare_message_object(self) -> bool:
        if self.message_file_name is None or self.message_object is None:
            return False
        start_label = default_start_label
        if self.params and 'start_label' in self.params:
            start_label = self.params['start_label']
        start_label_bytes = chars2bytes(start_label)
        end_label = defualt_end_label
        if self.params and 'end_label' in self.params:
            end_label = self.params['end_label']
        end_label_bytes = chars2bytes(end_label)
        message_file_name_bytes = chars2bytes(self.message_file_name)
        message_file_name_bytes_len = asarray([len(message_file_name_bytes)])
        message_bytes = concatenate((
            start_label_bytes,
            message_file_name_bytes_len,
            message_file_name_bytes,
            self.message_object,
            end_label_bytes
        ))
        self.message_bits = to_bit_vector(message_bytes)
        self.message_len = len(self.message_bits)
        return True


    @abstractmethod
    def embeding(self):
        pass


    def save_stego_file(self, stego_file_path: str):
        if self.stego_object is not None:
            with Image.fromarray(self.stego_object) as F:
                F.save(stego_file_path)


    def process_one_file(self, cover_file_path, stego_file_path, message_file_path, **params):
        self.set_params(**params)
        self.load_cover_file(cover_file_path)
        self.load_message_file(message_file_path)
        if not self.prepare_message_object(): return
        self.embeding()
        self.save_stego_file(stego_file_path)


