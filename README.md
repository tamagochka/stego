## Запуск с использованием docker-compose
``` bash
ARGS='embedding -a lsb -p "fill_rest=False" -c test5.bmp -m message1.txt -s test5_lsb.bmp' ASSETS='~/assets' CONF=./config.toml docker-compose up
```

``` bash
ARGS='extracting -a lsb -s test5_lsb.bmp' docker-compose up
```

``` bash
ARGS='analysing -a visual -s test5_lsb.bmp -r test5_lsb' docker-compose up
```

**ARGS** - аргументы командной строки передаваемые программе, доступные аргументы описаны в справке
**CONF** - путь к файлу конфигурации по умолчанию в рабочей директории *./config.toml*
**ASSETS** - путь к директориям с наборами данных по умолчанию в рабочей директории *./assets*
имена директорий задаются в файле конфигурации

## Нативный запуск под linux
``` bash
python -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
```
``` bash
python run.py -cfg config.toml embedding -a lsb -p "fill_rest=False" -c test5.bmp -m message1.txt -s test5_lsb.bmp
```

``` bash
python run.py -cfg config.toml extracting -a lsb -s test5_lsb.bmp
```

``` bash
python run.py -cfg config.toml analysing -a visual -s test5_lsb.bmp -r test5_lsb
```
