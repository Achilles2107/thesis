import pathlib
import os
from pathlib import Path

cwd = pathlib.Path.cwd()

cwd / '/Thesis/'

print('Current Directory: \n', cwd)

path = Path(cwd)

iris_no_fl = os.system(str(path / 'iris_model/iris_model.py'))
