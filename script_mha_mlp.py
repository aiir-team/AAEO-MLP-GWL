# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

# import warnings
# from tensorflow.python.util import deprecation
# warnings.simplefilter(action='ignore', category=FutureWarning)
# deprecation._PRINT_DEPRECATION_WARNINGS = False

from sklearn.model_selection import ParameterGrid
import concurrent.futures as parallel
from time import time
from models import mha_mlp
from utils.io_util import load_dataset
from utils.preprocessor_util import TimeSeries
from config import Config, MhaConfig


def phase_one(algorithm):
    print(f"Start running: {algorithm['name']}")

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
                        "verbose": Config.VERBOSE_LATEST,
                        "validation_used": data["validation_used"],
                        "pathsave": f"{Config.DATA_RESULTS}/{data['name_data']}/{idx_network}-{obj}-{trial}/{algorithm['name']}"
                    }
                    hybrid_paras = {
                        "lb": Config.MHA_LB,
                        "ub": Config.MHA_UB,
                    }
                    parameters_grid = list(ParameterGrid(algorithm["param_grid"]))
                    for mha_paras in parameters_grid:
                        md = getattr(mha_mlp, algorithm["class"])(base_paras, hybrid_paras, mha_paras)
                        md.processing(TS)


if __name__ == '__main__':
    start_phase1 = time()
    print(f"Phase 1 Start!!!")

    # phase_one(MhaConfig.models[0])
    with parallel.ProcessPoolExecutor(max_workers=6) as executor:
        results = executor.map(phase_one, MhaConfig.models)
    print(f"Phase 1 DONE: {time() - start_phase1} seconds")

