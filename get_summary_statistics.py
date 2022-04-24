# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from pathlib import Path
from pandas import read_csv, DataFrame
from sklearn.model_selection import ParameterGrid
from numpy import array
from config import Config, MhaConfig
from utils.io_util import save_to_csv_dict


def save_fast_to_csv(list_results, list_paths, columns):
    for idx, results in enumerate(list_results):
        df = DataFrame(results, columns=columns)
        df.to_csv(list_paths[idx], index=False)
    return True


def read_results_from_files(models, data):
    matrix_results = []
    for idx_network, network in enumerate(Config.LIST_NETWORKS):
        for obj in Config.OBJ_FUNCS:
            for trial in range(0, Config.N_TRIALS):
                for model in models:
                    path_model = f"{Config.DATA_RESULTS}/{data['name_data']}/{idx_network}-{obj}-{trial}/{model['name']}"

                    parameters_grid = list(ParameterGrid(model["param_grid"]))
                    keys = model["param_grid"].keys()
                    for mha_paras in parameters_grid:
                        filename = "".join([f"-{mha_paras[key]}" for key in keys]) + ".csv"

                        # Load metrics
                        filepath = f"{path_model}/{Config.FILENAME_METRICS}{filename}"
                        df = read_csv(filepath, usecols=Config.FILE_METRIC_CSV_HEADER)
                        values = df.values.tolist()[0]
                        results = [idx_network, obj, trial, model['name']] + values
                        matrix_results.append(array(results))
    matrix_results = array(matrix_results)
    matrix_dict = {}
    for idx, key in enumerate(Config.FILE_METRIC_CSV_HEADER_FULL):
        matrix_dict[key] = matrix_results[:, idx]
    ## Save final file to csv
    save_to_csv_dict(matrix_dict, Config.FILENAME_STATISTICS_FINAL, f"{Config.DATA_RESULTS}/{data['name_data']}")
    # savetxt(f"{Config.DATA_RESULTS}/statistics_final.csv", matrix_results, delimiter=",")
    df = read_csv(f"{Config.DATA_RESULTS}/{data['name_data']}/{Config.FILENAME_STATISTICS_FINAL}.csv", usecols=Config.FILE_METRIC_CSV_HEADER_FULL)
    return df



## Read the final csv file and calculate min,max,mean,std,cv. for each: test_size | m_rule | obj | model | paras | trial 1 -> n

for idx_data, data in enumerate(Config.DATA):

    df_results = read_results_from_files(MhaConfig.models, data)
    print(df_results.info())

    for idx_network, network in enumerate(Config.LIST_NETWORKS):
        for obj in Config.OBJ_FUNCS:
            pathsave = f"{Config.DATA_RESULTS}/{data['name_data']}/{idx_network}-{obj}-{Config.FOLDERNAME_STATISTICS}"
            Path(pathsave).mkdir(parents=True, exist_ok=True)
            min_results, mean_results, max_results, std_results, cv_results = [], [], [], [], []
            for model in MhaConfig.models:
                parameters_grid = list(ParameterGrid(model["param_grid"]))
                keys = model["param_grid"].keys()

                for mha_paras in parameters_grid:
                    model_paras = "".join([f"-{mha_paras[key]}" for key in keys])
                    model_paras = model_paras[1:]
                    df_result = df_results[(df_results["network"] == idx_network) &
                                           (df_results["obj"] == obj) & (df_results["model"] == model["name"]) &
                                           (df_results["model_paras"] == model_paras)][Config.FILE_METRIC_CSV_HEADER_CALCULATE]

                    t1 = df_result.min(axis=0).to_numpy()
                    t2 = df_result.mean(axis=0).to_numpy()
                    t3 = df_result.max(axis=0).to_numpy()
                    t4 = df_result.std(axis=0).to_numpy()
                    t5 = t4 / t2

                    t1 = [idx_network, obj, model["name"], model_paras] + t1.tolist()
                    t2 = [idx_network, obj, model["name"], model_paras] + t2.tolist()
                    t3 = [idx_network, obj, model["name"], model_paras] + t3.tolist()
                    t4 = [idx_network, obj, model["name"], model_paras] + t4.tolist()
                    t5 = [idx_network, obj, model["name"], model_paras] + t5.tolist()

                    min_results.append(t1)
                    mean_results.append(t2)
                    max_results.append(t3)
                    std_results.append(t4)
                    cv_results.append(t5)
            save_fast_to_csv([min_results, mean_results, max_results, std_results, cv_results],
                             [f"{pathsave}/{Config.FILE_MIN}", f"{pathsave}/{Config.FILE_MEAN}",
                              f"{pathsave}/{Config.FILE_MAX}", f"{pathsave}/{Config.FILE_STD}",
                              f"{pathsave}/{Config.FILE_CV}"], columns=Config.FILE_METRIC_HEADER_STATISTICS)


