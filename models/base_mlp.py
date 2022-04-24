# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from time import time
from utils.math_util import round_function
from utils.io_util import save_results_to_csv, save_to_csv_dict
from utils.model_util import save_model_latest
from utils.visual.line import draw_predict_line_with_error
from utils import math_util
from utils.preprocessor_util import TimeSeries
from permetrics.regression import RegressionMetric
from config import Config
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
import numpy as np


class BaseClass:

    def __init__(self, base_paras=None):
        self.obj = base_paras["obj"]
        self.n_inputs = base_paras["n_inputs"]
        self.list_layers = base_paras["list_layers"]
        self.n_layers = len(self.list_layers)
        dataset = base_paras["dataset"]
        self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test = \
            dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5]
        self.verbose = base_paras["verbose"]
        self.pathsave = base_paras["pathsave"]
        self.validation_used = base_paras["validation_used"]

        ## Below variables will be created by child class of this class
        self.traditional_mlp = False
        self.filename, self.model = None, None
        self.solution, self.best_fit, self.loss_train = None, None, None

    def decode_solution(self, solution):
        pass

    def prediction(self, data=None, solution=None):
        self.decode_solution(solution)
        return self.model.predict(data)

    def training(self):  # Depend the child class of this class. They will implement their training function
        pass

    def save_results(self, results: dict):
        ## Save prediction results_paper of training set and testing set to csv file
        data = {key: results[key] for key in Config.HEADER_TRUTH_PREDICTED_TRAIN_FILE}
        save_to_csv_dict(data, f"{Config.FILENAME_PRED_TRAIN}-{self.filename}", self.pathsave)

        if self.validation_used:
            data = {key: results[key] for key in Config.HEADER_TRUTH_PREDICTED_VALID_FILE}
            save_to_csv_dict(data, f"{Config.FILENAME_PRED_VALID}-{self.filename}", self.pathsave)

        data = {key: results[key] for key in Config.HEADER_TRUTH_PREDICTED_TEST_FILE}
        save_to_csv_dict(data, f"{Config.FILENAME_PRED_TEST}-{self.filename}", self.pathsave)

        ## Save loss train to csv file
        epoch = list(range(1, len(self.loss_train['loss']) + 1))
        if self.validation_used:
            data = {"epoch": epoch, "loss": self.loss_train['loss'], "val_loss": self.loss_train['val_loss']}
        else:
            data = {"epoch": epoch, "loss": self.loss_train['loss']}
        save_to_csv_dict(data, f"{Config.FILENAME_LOSS_TRAIN}-{self.filename}", self.pathsave)

        ## Calculate performance metrics and save it to csv file
        RM1 = RegressionMetric(results[Config.Y_TRAIN_TRUE_UNSCALED].flatten(), results[Config.Y_TRAIN_PRED_UNSCALED].flatten(), decimal=3)
        mm1 = RM1.get_metrics_by_list_names(Config.METRICS_FOR_TESTING_PHASE)
        mm2 = {}
        if self.validation_used:
            RM2 = RegressionMetric(results[Config.Y_VALID_TRUE_UNSCALED].flatten(), results[Config.Y_VALID_PRED_UNSCALED].flatten(), decimal=3)
            mm2 = RM2.get_metrics_by_list_names(Config.METRICS_FOR_TESTING_PHASE)
        RM3 = RegressionMetric(results[Config.Y_TEST_TRUE_UNSCALED].flatten(), results[Config.Y_TEST_PRED_UNSCALED].flatten(), decimal=3)
        mm3 = RM3.get_metrics_by_list_names(Config.METRICS_FOR_TESTING_PHASE)
        item = {'model_paras': self.filename, 'time_train': self.time_train, 'time_total': self.time_total}
        for metric_name, value in mm1.items():
            item[metric_name + "_train"] = value
        if self.validation_used:
            for metric_name, value in mm2.items():
                item[metric_name + "_valid"] = value
        for metric_name, value in mm3.items():
            item[metric_name + "_test"] = value
        filename_metrics = f"{Config.FILENAME_METRICS}-{self.filename}"
        save_results_to_csv(item, filename_metrics, self.pathsave)

        ## Save models
        path_file = f"{self.pathsave}/model-{self.filename}"
        save_model_latest(self, pathfile=path_file, mlp=self.traditional_mlp)

        ## Visualization
        draw_predict_line_with_error([results[Config.Y_TEST_TRUE_UNSCALED].flatten(), results[Config.Y_TEST_PRED_UNSCALED].flatten()],
                                     [item["R_test"], item["RMSE_test"]], f"{self.filename}-{Config.FILENAME_VISUAL_PERFORMANCE}",
                                     self.pathsave, Config.FILE_FIGURE_TYPES)

    ## Processing all tasks
    def processing(self, TS: TimeSeries):
        self.time_total = time()
        self.TS = TS
        ## Pre-processing dataset
        ## Training
        self.time_train = time()
        self.training()
        self.time_train = round_function(time() - self.time_train, 3)

        ## Get the prediction for training and testing set
        Y_train_pred = self.prediction(self.X_train, self.solution).reshape((-1, 1))
        Y_valid_pred = None
        if self.validation_used:
            Y_valid_pred = self.prediction(self.X_valid, self.solution).reshape((-1, 1))
        Y_test_pred = self.prediction(self.X_test, self.solution).reshape((-1, 1))

        ## Unscaling the predictions to calculate the errors
        results = {
            Config.Y_TRAIN_TRUE_SCALED: self.Y_train,
            Config.Y_TRAIN_TRUE_UNSCALED: TS.fit_data["y"],
            Config.Y_TRAIN_PRED_SCALED: Y_train_pred,
            Config.Y_TRAIN_PRED_UNSCALED: TS.inverse_scale(TS.scale_type, Y_train_pred, TS.fit_data["y"]),

            Config.Y_TEST_TRUE_SCALED: self.Y_test,
            Config.Y_TEST_TRUE_UNSCALED: TS.inverse_scale(TS.scale_type, self.Y_test, TS.fit_data["y"]),
            Config.Y_TEST_PRED_SCALED: Y_test_pred,
            Config.Y_TEST_PRED_UNSCALED: TS.inverse_scale(TS.scale_type, Y_test_pred, TS.fit_data["y"]),
        }
        if self.validation_used:
            results[Config.Y_VALID_TRUE_SCALED] = self.Y_valid
            results[Config.Y_VALID_TRUE_UNSCALED] = TS.inverse_scale(TS.scale_type, self.Y_valid, TS.fit_data["y"])
            results[Config.Y_VALID_PRED_SCALED] = Y_valid_pred
            results[Config.Y_VALID_PRED_UNSCALED] = TS.inverse_scale(TS.scale_type, Y_valid_pred, TS.fit_data["y"]),

        self.time_total = round(time() - self.time_total, 3)
        self.save_results(results)


class SimpleMlp(BaseClass):
    def __init__(self, base_paras=None, mlp_paras=None):
        super().__init__(base_paras)
        self.optimizer = mlp_paras["optimizer"]
        self.learning_rate = mlp_paras["learning_rate"]
        self.epoch = mlp_paras["epoch"]
        self.batch_size = mlp_paras["batch_size"]
        self.filename = f"{self.epoch}-{self.optimizer}-{self.learning_rate}-{self.batch_size}"
        self.traditional_mlp = True

    def training(self):  # Depend the child class of this class. They will implement their training function
        self.model = Sequential()
        for idx, layer in enumerate(self.list_layers):
            if idx == 0:
                self.model.add(Dense(layer["n_nodes"], input_dim=self.n_inputs, activation=layer["activation"]))
            else:
                self.model.add(Dense(layer["n_nodes"], activation=layer["activation"]))
            if idx != (self.n_layers - 1):
                self.model.add(Dropout(layer["dropout"]))
        # Configure the model and start training
        # opt = getattr(optimizers, self.optimizer)(learning_rate=self.learning_rate, momentum=0.9)
        # self.model.compile(optimizer=opt, loss=getattr(math_util, self.obj))
        self.model.compile(optimizer=self.optimizer, loss=getattr(math_util, self.obj))
        if self.validation_used:
            ml = self.model.fit(self.X_train, self.Y_train, epochs=self.epoch, batch_size=self.batch_size,
                                verbose=int(self.verbose), validation_data=(self.X_valid, self.Y_valid))
        else:
            ml = self.model.fit(self.X_train, self.Y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=int(self.verbose))
        self.loss_train = ml.history            # dictionary: ['loss': [], 'val_loss': [] ]


class HybridMlp(BaseClass):
    def __init__(self, base_paras=None, hybrid_paras=None):
        super().__init__(base_paras)
        self.model = self.create_network()
        self.problem_size = self.calculate_problem_size(self.model)
        self.lb = hybrid_paras["lb"] * self.problem_size
        self.ub = hybrid_paras["ub"] * self.problem_size
        self.problem = {
            "fit_func": self.objective_function,
            "lb": self.lb,
            "ub": self.ub,
            "minmax": "min",
            "save_population": False,
            "log_to": self.verbose
        }
        if self.validation_used:
            self.problem["obj_weight"] = Config.TRAIN_TEST_OBJ_WEIGHTS_FOR_METRICS  # Define it or default value will be [1, 1]

    def create_network(self):
        model = Sequential()
        for idx, layer in enumerate(self.list_layers):
            if idx == 0:
                model.add(Dense(layer["n_nodes"], input_dim=self.n_inputs, activation=layer["activation"]))
            else:
                model.add(Dense(layer["n_nodes"], activation=layer["activation"]))
            if idx != (self.n_layers - 1):
                model.add(Dropout(layer["dropout"]))
        return model

    def calculate_problem_size(self, model: Sequential):
        return np.sum([np.size(w) for w in model.get_weights()])

    def decode_solution(self, solution):
        weight_sizes = [(w.shape, np.size(w)) for w in self.model.get_weights()]
        weights = []
        cut_points = 0
        for ws in weight_sizes:
            weights.append(np.reshape(solution[cut_points:cut_points + ws[1]], ws[0]))
            cut_points += ws[1]
        self.model.set_weights(weights)

    # Evaluates the objective function
    def objective_function(self, solution=None):
        y_train_pred = self.prediction(self.X_train, solution)
        obj_train = RegressionMetric(self.Y_train.flatten(), y_train_pred.flatten(), decimal=8)
        loss = obj_train.get_metric_by_name(self.obj)[self.obj]
        if self.validation_used:
            y_valid_pred = self.model.predict(self.X_valid)
            obj_valid = RegressionMetric(self.Y_valid.flatten(), y_valid_pred.flatten(), decimal=8)
            val_loss = obj_valid.get_metric_by_name(self.obj)[self.obj]
            return [loss, val_loss]
        return loss

    def get_history_loss(self, list_global_best=None):
        # 2D array / matrix 2D
        global_obj_list = np.array([agent[1][-1] for agent in list_global_best])
        global_obj_list = global_obj_list[1:]
        # Make each obj_list as a element in array for drawing
        if self.validation_used:
            return {
                "loss": global_obj_list[:, 0],
                "val_loss": global_obj_list[:, 1]
            }
        return {"loss": global_obj_list}
