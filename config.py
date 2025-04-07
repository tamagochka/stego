import os
import sys
import logging
from logging import handlers
from dataclasses import dataclass

import toml


def check_dir(dir_name: str):
    if dir_name:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

def get_file_path(folder_name: str, file_name: str):
    file_path = file_name
    if folder_name:
        file_path = os.path.join(folder_name, file_name)
    return file_path


@dataclass
class AppCongig(object):
    """
    Класс для хранения конфигурации приложения
    
    Attributes
    ----------
        config_file: str = None
            путь к файлу конфигурации
        log_level: int = logging.WARN
            уровень логирования
        log_format: str = '%(asctime)s - [%(levelname)s] - %(funcName)s - %(message)s'
            формат логов
        log_to_console: bool = True
            выводить лог в консоль
        log_to_file: bool = False
            выводить лог в файл
        log_file_dir: str = '.'
            путь к лог-файлам
        log_file_name: str = 'stego.log'
            шаблон имени лог-файлов
        log_file_max_bytes: int = 5242880
            максимальный размер лог-файла, после которого создается новый
        log_files_count: int = 10
            максимальное количество лог-файлов, хранящихся на компьютере
        
        директории для хранения данных:
        extracts_folder: str = None
            директория с извлеченными вложениями
        covers_folder: str = None
            дирректория с покрывающими объектами
        stegos_folder: str = None
            директория с полученными стеганограммами
        messages_folder: str = None
            дирктороия с вложениями
        analysis_folder: str = None
            директория с результатами стегоанализа
        если заданы директории для хранения данных, то программа будет искать и сохранять файлы в них,
        иначе будет использоваться путь, заданный пользователем

    Methods
    -------
        apply_config()
            применить конфигурацию, если в конфигурации заданы директории для хранения данных,
            то они будут созданы при применении конфигурации
    """

    config_file: str = None

    log_level: int = logging.WARN
    log_format: str = '%(asctime)s - [%(levelname)s] - %(funcName)s - %(message)s'
    log_to_console: bool = True
    log_to_file: bool = False
    log_file_dir: str = '.'
    log_file_name: str = 'stego.log'
    log_file_max_bytes: int = 5242880
    log_files_count: int = 10

    extracts_folder: str = None
    covers_folder: str = None
    stegos_folder: str = None
    messages_folder: str = None
    analysis_folder: str = None


    def __init__(self, config_file: str | None):
        if not config_file:
            return
        if os.path.exists(config_file):
            self.config_file = config_file
            conf = toml.load(self.config_file)
            logging_conf = conf.get('logging', None)
            if logging_conf:
                log_level = logging_conf.get('log_level', None)
                if log_level:
                    self.log_level = logging.getLevelNamesMapping().get(log_level, logging.WARN)
                self.log_format = logging_conf.get('log_format', self.log_format)
                self.log_to_console = logging_conf.get('log_to_console', self.log_to_console)
                self.log_to_file = logging_conf.get('log_to_file', self.log_to_file)
                self.log_file_name = logging_conf.get('log_file_name', self.log_file_name)
                self.log_file_dir = logging_conf.get('log_file_dir', self.log_file_dir)
                self.log_file_max_bytes = logging_conf.get('log_file_max_bytes', self.log_file_max_bytes)
                self.log_files_count = logging_conf.get('log_files_count', self.log_files_count)
            files_conf = conf.get('files', None)
            if files_conf:
                self.extracts_folder = files_conf.get('extracts_folder', self.extracts_folder)
                self.covers_folder = files_conf.get('covers_folder', self.covers_folder)
                self.stegos_folder = files_conf.get('stegos_folder', self.stegos_folder)
                self.messages_folder = files_conf.get('messages_folder', self.messages_folder)
                self.analysis_folder = files_conf.get('analysis_folder', self.analysis_folder)


    def __str__(self):
        s = f'''
            config_file: {self.config_file}
            log_level: {self.log_level}
            log_format: {self.log_format}
            log_to_console: {self.log_to_console}
            log_to_file: {self.log_to_file}
            log_file_dir: {self.log_file_dir}
            log_file_name: {self.log_file_name}
            log_file_max_bytes: {self.log_file_max_bytes}
            log_files_count: {self.log_files_count}
            extract_folder: {self.extracts_folder}
            covers_folder: {self.covers_folder}
            stegos_folder: {self.stegos_folder}
            messages_folder: {self.messages_folder}
            analysis_folder: {self.analysis_folder}
        '''
        return s


    def apply_config(self):
        if self.log_to_file:
            if not os.path.isdir(self.log_file_dir):
                os.makedirs(self.log_file_dir)
        handlers_list = []
        if self.log_to_console:
            handlers_list.append(logging.StreamHandler(stream=sys.stdout))
        else:
            handlers_list.append(logging.NullHandler())
        if self.log_to_file:
            handlers_list.append(
                    handlers.RotatingFileHandler(
                    os.path.join(self.log_file_dir, self.log_file_name),
                    maxBytes=self.log_file_max_bytes,
                    backupCount=self.log_files_count
                )
            )
        logging.basicConfig(
            level=self.log_level,
            format=self.log_format,
            handlers=handlers_list
        )
        check_dir(self.extracts_folder)
        check_dir(self.covers_folder)
        check_dir(self.stegos_folder)
        check_dir(self.messages_folder)
        check_dir(self.analysis_folder)
   

    def get_extracts_file_path(self, file_name: str) -> str:
        return get_file_path(self.extracts_folder, file_name)
    
    def get_covers_file_path(self, file_name: str) -> str:
        return get_file_path(self.covers_folder, file_name)

    def get_stegos_file_path(self, file_name: str) -> str:
        return get_file_path(self.stegos_folder, file_name)

    def get_messages_file_path(self, file_name: str) -> str:
        return get_file_path(self.messages_folder, file_name)

    def get_analysis_file_path(self, file_name: str) -> str:
        return get_file_path(self.analysis_folder, file_name)

