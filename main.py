import pathlib
import pandas as pd
from pathlib import Path

cwd = pathlib.Path.cwd()

print(cwd)

path = Path(cwd / 'datasets/iris_classification')

df = pd.read_csv(path / 'iris_training.csv')
print(df.head)

print(path)
