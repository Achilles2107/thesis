import csv
import random

records = 120
print("Making %d records\n" % records)

fieldnames = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
writer = csv.DictWriter(open("/iris_random01.csv", "w", newline=''), fieldnames=fieldnames)

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']


def rd_float():
    x = random.uniform(0.1, 10.0)
    return x


def rd_int():
    y = random.randint(0, 2)
    return y


writer.writerow(dict(zip(fieldnames, fieldnames)))
for i in range(0, records):
  writer.writerow(dict([
    ('sepal_length', round(rd_float(), 1)),
    ('sepal_width', round(rd_float(), 1)),
    ('petal_length', round(rd_float(), 1)),
    ('petal_width', round(rd_float(), 1)),
    ('species', rd_int())]))

print('done')
