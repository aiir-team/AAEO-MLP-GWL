# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from sklearn.model_selection import ParameterGrid
from models.base_mlp import SimpleMlp
from config import Config, MhaConfig
from time import time
from utils.io_util import load_dataset
from utils.preprocessor_util import TimeSeries


if __name__ == '__main__':
    starttime = time()

    for idx_data, data in enumerate(Config.DATA):
        # load dataset and select feature for training and testing
        X_data, Y_data = load_dataset(data["name_folder"], data["name_file"], data["name_input"], data["name_output"])
        TS = TimeSeries(data=[X_data, Y_data], scale_type=data["scaling"])
        x_train, x_test, y_train, y_test = TS.train_test_split(X_data, Y_data, train_size=data["train_size"])
        X_train, X_test = TS.scale(scale_type=data["scaling"], train_data=x_train, test_data=x_test, fit_data=x_train, fit_name="x")
        Y_train, Y_test = TS.scale(scale_type=data["scaling"], train_data=y_train, test_data=y_test, fit_data=y_train, fit_name="y")
        dataset_final = [X_train, Y_train, None, None, X_test, Y_test]
        n_inputs = len(X_train[0])
        for idx_network, network in enumerate(Config.LIST_NETWORKS):
            for obj in Config.OBJ_FUNCS:
                for trial in range(0, Config.N_TRIALS):
                    base_paras = {
                        "obj": obj,
                        "n_inputs": n_inputs,
                        "list_layers": network,
                        "dataset": dataset_final,
                        "verbose": Config.VERBOSE_SIMPLE_MLP,
                        "validation_used": data["validation_used"],
                        "pathsave": f"{Config.DATA_RESULTS}/{data['name_data']}/{idx_network}-{obj}-{trial}/MLP"
                    }
                    parameters_grid = list(ParameterGrid(MhaConfig.mlp))
                    for mlp_paras in parameters_grid:
                        md = SimpleMlp(base_paras, mlp_paras)
                        md.processing(TS)
    print('That took: {} seconds'.format(time() - starttime))