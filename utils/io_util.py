# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from numpy import array
from csv import DictWriter
from pandas import DataFrame, read_csv
from pathlib import Path


def save_to_csv(data:list, header:list, filename=None, pathsave=None):
    ## Check the parent directories
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    # Convert data and header to dictionary
    mydict = {}
    for idx, h in enumerate(header):
        mydict[h] = array(data[idx]).reshape(-1)
    df = DataFrame(mydict, columns=header)
    df.to_csv(f"{pathsave}/{filename}.csv", index=False, header=True)
    return None


def save_to_csv_dict(data:dict, filename=None, pathsave=None):
    ## Check the parent directories
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    ## Reshape data
    data_shaped = {}
    for key, value in data.items():
        data_shaped[key] = array(value).reshape(-1)
    df = DataFrame(data_shaped, columns=data_shaped.keys())
    df.to_csv(f"{pathsave}/{filename}.csv", index=False, header=True)
    return None


def save_results_to_csv(data:dict, filename=None, pathsave=None):
    ## Check the parent directories
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    with open(f"{pathsave}/{filename}.csv", 'a') as file:
        w = DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=data.keys())
        if file.tell() == 0:
            w.writeheader()
        w.writerow(data)
    return None


def load_dataset(name_folder=None, name_file=None, name_input=None, name_output=None):
    dataset = read_csv(f"{name_folder}/{name_file}")
    X = dataset[name_input].values                      # Input variables
    if name_output is not None:
        Y = dataset[name_output].values.reshape(-1, 1)  # Output variable
        return X, Y
    return X, None