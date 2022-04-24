# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from pandas import read_csv, Series
from sklearn.model_selection import ParameterGrid
import numpy as np
from config import Config
from utils.io_util import save_to_csv_dict
from keras.models import load_model as keras_load_model
import pickle


def save_model_latest(obj, pathfile=None, mlp=False):
    obj.model.save(f'{pathfile}.h5')
    del obj.model
    if not mlp:
        del obj.optimizer.history.list_population
    name_obj = open(f'{pathfile}.pkl', 'wb')
    pickle.dump(obj, name_obj)
    name_obj.close()
    return 0


def load_model_latest(pathfile=None):
    name_path = f"{pathfile}.h5"
    model = keras_load_model(name_path, compile=False)
    obj = pickle.load(open(f"{pathfile}.pkl", 'rb'))
    obj.model = model
    return obj


def save_all_results_to_csv(models):
    matrix_results = []
    for test_size in Config.TEST_SIZE:
        for idx_network, network in enumerate(Config.LIST_NETWORKS):
            for obj in Config.OBJ_FUNCS:
                for trial in range(0, Config.N_TRIALS):
                    for model in models:
                        path_general = f"{Config.DATA_RESULTS}/{test_size}-{idx_network}/{obj}/trial-{trial}/{model['name']}"
                        path_model = f"{path_general}/{Config.RESULTS_FOLDER_MODEL}"

                        parameters_grid = list(ParameterGrid(model["param_grid"]))
                        keys = model["param_grid"].keys()
                        for mha_paras in parameters_grid:
                            filename = "".join([f"-{mha_paras[key]}" for key in keys]) + ".csv"

                            # Load metrics
                            filepath = f"{path_model}/{Config.FILENAME_METRICS}{filename}"
                            df = read_csv(filepath, usecols=Config.FILE_METRIC_CSV_HEADER)
                            values = df.values.tolist()[0]
                            results = [test_size, idx_network, obj, trial, model['name']] + values
                            matrix_results.append(np.array(results))
    matrix_results = np.array(matrix_results)
    matrix_dict = {}
    for idx, key in enumerate(Config.FILE_METRIC_CSV_HEADER_FULL):
        matrix_dict[key] = matrix_results[:, idx]
    ## Save final file to csv
    save_to_csv_dict(matrix_dict, Config.FILENAME_STATISTICS_FINAL, f"{Config.DATA_RESULTS}")
    # savetxt(f"{Config.DATA_RESULTS}/statistics_final.csv", matrix_results, delimiter=",")
    df = read_csv(f"{Config.DATA_RESULTS}/{Config.FILENAME_STATISTICS_FINAL}.csv", usecols=Config.FILE_METRIC_CSV_HEADER_FULL)
    return df


def get_best_model(dataframe):
    def fitness(cols):
        return Series([ np.sum(cols * Config.PHASE1_BEST_WEIGHTS) ])
    dataframe['fitness'] = dataframe[Config.PHASE1_BEST_METRICS].apply(fitness, axis=1)

    ## Get the best model based on the min fitness
    minvalueIndexLabel = dataframe['fitness'].idxmin()
    best_data = dataframe.iloc[[minvalueIndexLabel]]

# test_size,network,obj,trial,model,model_paras,time_train,time_total,MAE_train,RMSE_train,R_train,R2s_train,MAPE_train,NSE_train,KGE_train,PCD_train,KLD_train,VAF_train,A10_train,A20_train,MAE_test,RMSE_test,R_test,R2s_test,MAPE_test,NSE_test,KGE_test,PCD_test,KLD_test,VAF_test,A10_test,A20_test
# 0.3,0,RMSE,0,MLP,5-sgd-0.1-16-0.2,3.885,3.926,2.2640000000000002,3.105,0.9570000000000001,0.9159999999999999,0.107,0.904,0.934,0.91,10.222000000000001,91.61,0.6,0.825,2.4619999999999997,3.2680000000000002,0.953,0.9079999999999999,0.121,0.898,0.932,0.8740000000000001,8.217,90.807,0.511,0.7879999999999999

    name_folder = f"{Config.DATA_RESULTS}/{best_data['test_size'].iat[0]}-{best_data['network'].iat[0]}/" + \
                  f"{best_data['obj'].iat[0]}/trial-{best_data['trial'].iat[0]}/{best_data['model'].iat[0]}"
    name_file = f"model-{best_data['model_paras'].iat[0]}"
    model = {
        "folder": name_folder,
        "file": name_file
    }
    # model = load_model_latest(name_folder, name_file)
    return model, best_data









