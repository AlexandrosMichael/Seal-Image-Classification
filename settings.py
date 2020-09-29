import pathlib

BASE_DIR = pathlib.Path().absolute()

BINARY_DIR = pathlib.Path().joinpath(BASE_DIR, 'data/binary')

MULTI_DIR = pathlib.Path().joinpath(BASE_DIR, 'data/multi')


