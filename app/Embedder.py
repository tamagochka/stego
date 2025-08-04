import os
from abc import ABC, abstractmethod

from PIL import Image
from numpy.typing import NDArray
from numpy import asarray, uint8, fromfile, concatenate

from .utils import chars2bytes, to_bit_vector


class Embedder(ABC):
    
    cover_object: NDArray[uint8] | None = None
    message_object: NDArray[uint8] | None = None
    stego_object: NDArray[uint8] | None = None

    message_file_name: str | None = None
    message_bits: NDArray[uint8]
    message_len: int


    def load_cover_file(self, cover_file_path: str):
        with Image.open(cover_file_path) as F:
            self.cover_object = asarray(F, dtype=uint8)


    def load_message_file(self, message_file_path: str):
        with open(message_file_path, 'rb') as F:
            self.message_object = fromfile(F, dtype=uint8)
            self.message_file_name = os.path.basename(F.name)


    def prepare_message_object(self, **params) -> bool:
        if not self.message_file_name or not self.message_object:
            return False
        start_label = params['start_label']
        start_label_bytes = ''  # TODO проверить работу если в метод не будет передан start_label
        if start_label:
            start_label_bytes = chars2bytes(start_label)
        end_label = params['end_label']
        end_label_bytes = ''
        if end_label:
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
        if self.stego_object:
            with Image.fromarray(self.stego_object) as F:
                F.save(stego_file_path)
