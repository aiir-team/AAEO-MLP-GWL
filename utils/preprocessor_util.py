# !/usr/bin/env python
# Created by "Thieu" at 17:22, 08/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from sklearn.preprocessing import MinMaxScaler
from config import Config
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from utils.io_util import load_dataset


def get_scaler(mode:str, X_data:None):
    """
    mode = "dataset" --> Get scaler based on input X
    mode = "lbub" --> get scaler based on lower bound, upper bound in phase 2
    """
    scaler = MinMaxScaler()  # Data scaling using the MinMax method
    if mode == "dataset":
        scaler.fit(X_data)
    elif mode == "lbub":
        X_data = np.array([Config.MHA_LB_SCALING, Config.MHA_UB_SCALING])
        scaler.fit(X_data)
    else:
        print("Please select the scaling mode again!")
        exit(0)
    return scaler


class CheckDataset:
    def __init__(self):
        pass

    def check_consecutive(self, df, time_name="timestamp", time_different=300):
        """
        :param df: Type of this must be dataframe
        :param time_name: the column name of date time
        :param time_different by seconds: 300 = 5 minutes
            https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.Timedelta.html
        :return:
        """
        consecutive = True
        for i in range(df.shape[0] - 1):
            diff = (df[time_name].iloc[i + 1] - df[time_name].iloc[i]).seconds
            if time_different != diff:
                print("===========Not consecutive at: {}, different: {} ====================".format(i + 3, diff))
                consecutive = False
        return consecutive


class TimeSeries:
    def __init__(self, data=None, scale_type="minmax"):
        self.data_original = data
        self.scale_type = scale_type
        self.fit_data = {}

    def train_test_split(self, x_data, y_data, train_size):
        if 0 < train_size < 1:
            train_idx = int(train_size * x_data.shape[0])
        else:
            train_idx = int(train_size)
        x_train, x_test, y_train, y_test = x_data[:train_idx], x_data[train_idx:], y_data[:train_idx], y_data[train_idx:]
        return x_train, x_test, y_train, y_test

    def scale(self, scale_type="std", train_data=None, test_data=None, fit_data=None, fit_name="x"):
        self.fit_data[fit_name] = fit_data
        data_mean, data_std = fit_data.mean(axis=0), fit_data.std(axis=0)
        data_min, data_max = fit_data.min(axis=0), fit_data.max(axis=0)

        if scale_type == "std":
            train_scaled = (train_data - data_mean) / data_std
            test_scaled = (test_data - data_mean) / data_std
        elif scale_type == "minmax":
            train_scaled = (train_data - data_min) / (data_max - data_min)
            test_scaled = (test_data - data_min) / (data_max - data_min)
        elif scale_type == "kurtosis":
            train_scaled = np.sign(train_data - data_mean) * np.power(np.abs(train_data - data_mean), 1.0 / 3)
            test_scaled = np.sign(test_data - data_mean) * np.power(np.abs(test_data - data_mean), 1.0 / 3)
        elif scale_type == "kurtosis_std":
            train_kurtosis = np.sign(train_data - data_mean) * np.power(np.abs(train_data - data_mean), 1.0 / 3)
            test_kurtosis = np.sign(test_data - data_mean) * np.power(np.abs(test_data - data_mean), 1.0 / 3)
            data_mean_kur, data_std_kur = train_kurtosis.mean(axis=0), train_kurtosis.std(axis=0)
            train_scaled = (train_kurtosis - data_mean_kur) / data_std_kur
            test_scaled = (test_kurtosis - data_mean_kur) / data_std_kur
        elif scale_type == "kurtosis_minmax":
            train_kurtosis = np.sign(train_data - data_mean) * np.power(np.abs(train_data - data_mean), 1.0 / 3)
            test_kurtosis = np.sign(test_data - data_mean) * np.power(np.abs(test_data - data_mean), 1.0 / 3)
            data_min_kur, data_max_kur = train_kurtosis.min(axis=0), train_kurtosis.max(axis=0)
            train_scaled = (train_kurtosis - data_min_kur) / (data_max_kur - data_min_kur)
            test_scaled = (test_kurtosis - data_min_kur) / (data_max_kur - data_min_kur)
        elif scale_type == "boxcox":
            self.boxcox_lamdas = {}
            train_data = np.squeeze(train_data)
            if len(train_data.shape) == 1:
                train_scaled, lamda = boxcox(train_data.ravel())
                test_scaled = boxcox(test_data.ravel(), lmbda=lamda)
                self.boxcox_lamdas[fit_name] = lamda
                train_scaled = np.reshape(train_scaled, (-1, 1))
                test_scaled = np.reshape(test_scaled, (-1, 1))
            else:
                train_scaled = []
                test_scaled = []
                lamdas_list = []
                for idx in range(train_data.shape[1]):
                    train_new, lamda = boxcox(train_data[:, idx].ravel())
                    test_new = boxcox(test_data[:, idx].ravel(), lmbda=lamda)
                    train_scaled.append(train_new)
                    test_scaled.append(test_new)
                    lamdas_list.append(lamda)
                self.boxcox_lamdas[fit_name] = lamdas_list
                train_scaled = np.array(train_scaled).transpose()
                test_scaled = np.array(test_scaled).transpose()
        else: # scale_type == "loge":
            train_scaled = np.log(train_data)
            test_scaled = np.log(test_data)
        return train_scaled, test_scaled

    def inverse_scale(self, scale_type="std", data=None, fit_data=None):
        data_mean, data_std = fit_data.mean(axis=0), fit_data.std(axis=0)
        data_min, data_max = fit_data.min(axis=0), fit_data.max(axis=0)

        if scale_type == "std":
            return data_std * data + data_mean
        elif scale_type == "minmax":
            return data * (data_max - data_min) + data_min
        elif scale_type == "kurtosis":
            return np.power(data, 3) + data_mean
        elif scale_type == "kurtosis_std":
            fit_kurtosis_data = np.sign(fit_data - data_mean) * np.power(np.abs(fit_data - data_mean), 1.0 / 3)
            data_mean_kur, data_std_kur = fit_kurtosis_data.mean(axis=0), fit_kurtosis_data.std(axis=0)
            temp = data_std_kur * data + data_mean_kur
            return np.power(temp, 3) + data_mean
        elif scale_type == "kurtosis_minmax":
            fit_kurtosis_data = np.sign(fit_data - data_mean) * np.power(np.abs(fit_data - data_mean), 1.0 / 3)
            data_min_kur, data_max_kur = fit_kurtosis_data.min(axis=0), fit_kurtosis_data.max(axis=0)
            temp = data * (data_max_kur - data_min_kur) + data_min_kur
            return np.power(temp, 3) + data_mean
        elif scale_type == "boxcox":
            data = np.squeeze(data)
            if len(data.shape) == 1:
                return inv_boxcox(data, self.boxcox_lamdas["y"])
            else:
                data_results = []
                for idx in range(data.shape[1]):
                    data_results.append(inv_boxcox(data[:, idx], self.boxcox_lamdas["y"][idx]))
                return np.array(data_results).transpose()
        elif scale_type == "loge":
            return np.exp(data)


    def scaling(self, scale_type="std", separate=True):
        if separate:
            self.data_mean, self.data_std = self.data_original[:self.train_idx].mean(axis=0), self.data_original[:self.train_idx].std(axis=0)
            self.data_min, self.data_max = self.data_original[:self.train_idx].min(axis=0), self.data_original[:self.train_idx].max(axis=0)
        else:
            self.data_mean, self.data_std = self.data_original.mean(axis=0), self.data_original.std(axis=0)
            self.data_min, self.data_max = self.data_original.min(axis=0), self.data_original.max(axis=0)

        if scale_type == "std":
            self.data_new = (self.data_original - self.data_mean) / self.data_std
        elif scale_type == "minmax":
            self.data_new = (self.data_original - self.data_min) / (self.data_max - self.data_min)
        elif scale_type == "loge":
            self.data_new = np.log(self.data_original)

        elif scale_type == "kurtosis":
            self.data_new = np.sign(self.data_original - self.data_mean) * np.power(np.abs(self.data_original - self.data_mean), 1.0 / 3)
        elif scale_type == "kurtosis_std":
            self.data_kurtosis = np.sign(self.data_original - self.data_mean) * np.power(np.abs(self.data_original - self.data_mean), 1.0 / 3)
            self.data_mean_kur, self.data_std_kur = self.data_original[:self.train_idx].mean(axis=0), self.data_original[:self.train_idx].std(axis=0)
            self.data_new = (self.data_kurtosis - self.data_mean_kur) / self.data_std_kur

        elif scale_type == "boxcox":
            self.data_new, self.lamda_boxcox = boxcox(self.data_original.flatten())
        elif scale_type == "boxcox_std":
            self.data_boxcox, self.lamda_boxcox = boxcox(self.data_original.flatten())
            self.data_boxcox = self.data_boxcox.reshape(-1, 1)
            self.data_mean, self.data_std = self.data_boxcox[:self.train_idx].mean(axis=0), self.data_boxcox[:self.train_idx].std(axis=0)
            self.data_new = (self.data_boxcox - self.data_mean) / self.data_std
        return self.data_new

    def multi_scaling(self, scale_type="std", separate=True):
        """
        :param dataset: 2D numpy array
        :param scale_type: std / minmax
        :return:
        """
        self.data_new = []
        self.data_mean = []
        self.data_std = []
        self.data_min = []
        self.data_max = []

        self.data_kurtosis = []
        self.data_mean_kur = []
        self.data_std_kur = []
        self.lamda_boxcox = []
        self.data_boxcox = []

        self.multi_size = len(self.data_original[0])
        for i in range(self.multi_size):
            col_data_original = self.data_original[:, i]

            if separate:
                col_data_mean, col_data_std = col_data_original[:self.train_idx].mean(axis=0), col_data_original[:self.train_idx].std(axis=0)
                col_data_min, col_data_max = col_data_original[:self.train_idx].min(axis=0), col_data_original[:self.train_idx].max(axis=0)
            else:
                col_data_mean, col_data_std = col_data_original.mean(axis=0), col_data_original.std(axis=0)
                col_data_min, col_data_max = col_data_original.min(axis=0), col_data_original.max(axis=0)

            if scale_type == "std":
                col_data_new = (col_data_original - col_data_mean) / col_data_std
            elif scale_type == "minmax":
                col_data_new = (col_data_original - col_data_min) / (col_data_max - col_data_min)
            elif scale_type == "loge":
                col_data_new = np.log(col_data_original)

            elif scale_type == "kurtosis":
                col_data_new = np.sign(col_data_original - col_data_mean) * np.power(np.abs(col_data_original - col_data_mean), 1.0 / 3)
            elif scale_type == "kurtosis_std":
                col_data_kurtosis = np.sign(col_data_original - col_data_mean) * np.power(np.abs(col_data_original - col_data_mean), 1.0 / 3)
                col_data_mean_kur, col_data_std_kur = col_data_original[:self.train_idx].mean(axis=0), col_data_original[:self.train_idx].std(axis=0)
                col_data_new = (col_data_kurtosis - col_data_mean_kur) / col_data_std_kur
                self.data_kurtosis.append(col_data_kurtosis)
                self.data_mean_kur.append(col_data_mean_kur)
                self.data_std_kur.append(col_data_std_kur)
            elif scale_type == "boxcox":
                col_data_new, col_lamda_boxcox = boxcox(col_data_original.flatten())
            elif scale_type == "boxcox_minmax":
                col_data_boxcox, col_lamda_boxcox = boxcox(col_data_original.flatten())
                col_data_boxcox = col_data_boxcox.reshape(-1, 1)
                col_data_min, col_data_max = col_data_boxcox[:self.train_idx].min(axis=0), col_data_boxcox[:self.train_idx].max(axis=0)
                col_data_new = (col_data_boxcox - col_data_min) / (col_data_max - col_data_min)
                self.lamda_boxcox.append(col_lamda_boxcox)
                self.data_boxcox.append(col_data_boxcox)
            elif scale_type == "boxcox_std":
                col_data_boxcox, col_lamda_boxcox = boxcox(col_data_original.flatten())
                col_data_boxcox = col_data_boxcox.reshape(-1, 1)
                col_data_mean, col_data_std = col_data_boxcox[:self.train_idx].mean(axis=0), col_data_boxcox[:self.train_idx].std(axis=0)
                col_data_new = (col_data_boxcox - col_data_mean) / col_data_std
                self.lamda_boxcox.append(col_lamda_boxcox)
                self.data_boxcox.append(col_data_boxcox)

            self.data_mean.append(col_data_mean)
            self.data_std.append(col_data_std)
            self.data_min.append(col_data_min)
            self.data_max.append(col_data_max)
            self.data_new.append(col_data_new)

        return np.array(self.data_new)

    def inverse_scaling(self, data=None, scale_type="std"):

        if scale_type == "std":
            return self.data_std[self.multi_size - 1] * data + self.data_mean[self.multi_size - 1]
        elif scale_type == "minmax":
            return data * (self.data_max[self.multi_size - 1] - self.data_min[self.multi_size - 1]) + self.data_min[self.multi_size - 1]
        elif scale_type == "loge":
            return np.exp(data)

        elif scale_type == "kurtosis":
            return np.power(data, 3) + self.data_mean[self.multi_size - 1]
        elif scale_type == "kurtosis_std":
            temp = self.data_std_kur[self.multi_size - 1] * data + self.data_mean_kur[self.multi_size - 1]
            return np.power(temp, 3) + self.data_mean[self.multi_size - 1]

        elif scale_type == "boxcox":
            return inv_boxcox(data, self.lamda_boxcox[self.multi_size - 1])
        elif scale_type == "boxcox_minmax":
            boxcox_invert = data * (self.data_max[self.multi_size - 1] - self.data_min[self.multi_size - 1]) + self.data_min[self.multi_size - 1]
            return inv_boxcox(boxcox_invert, self.lamda_boxcox[self.multi_size - 1])
        elif scale_type == "boxcox_std":
            boxcox_invert = self.data_std[self.multi_size - 1] * data + self.data_mean[self.multi_size - 1]
            return inv_boxcox(boxcox_invert, self.lamda_boxcox[self.multi_size - 1])

    def univariate_data(self, dataset, history_column=None, start_index=0, end_index=None, pre_type="2D"):
        """
        :param dataset: 2-D numpy array
        :param history_column: python list time in the past you want to use. (1, 2, 5) means (t-1, t-2, t-5) predict time t
        :param start_index: 0- training set, N- valid or testing set
        :param end_index: N-training or valid set, None-testing set
        :param pre_type: 3D for RNN-based, 2D for normal neural network like MLP, FFLN,..
        :return:
        """
        data = []
        labels = []

        history_size = len(history_column)
        if end_index is None:
            end_index = len(dataset) - history_column[-1] - 1  # for time t, such as: t-1, t-4, t-7 and finally t
        else:
            end_index = end_index - history_column[-1] - 1

        for i in range(start_index, end_index):
            indices = i - 1 + np.array(history_column)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i + history_column[-1]])
        if pre_type == "3D":
            return np.array(data), np.array(labels)
        return np.reshape(np.array(data), (-1, history_size)), np.array(labels)

    # def inverse_scaling(self, data=None, scale_type="std"):
    #     if scale_type == "std":
    #         return self.data_std * data + self.data_mean
    #     elif scale_type == "minmax":
    #         return data * (self.data_max - self.data_min) + self.data_min
    #     elif scale_type == "loge":
    #         return np.exp(data)
    #
    #     elif scale_type == "kurtosis":
    #         return np.power(data, 3) + self.data_mean
    #     elif scale_type == "kurtosis_std":
    #         temp = self.data_std_kur * data + self.data_mean_kur
    #         return np.power(temp, 3) + self.data_mean
    #
    #     elif scale_type == "boxcox":
    #         return inv_boxcox(data, self.lamda_boxcox)
    #     elif scale_type == "boxcox_std":
    #         boxcox_invert = self.data_std * data + self.data_mean
    #         return inv_boxcox(boxcox_invert, self.lamda_boxcox)

    def multivariate_data(self, dataset, history_column=None, start_index=0, end_index=None, pre_type="2D"):
        """
        :param dataset: 2-D numpy array
        :param history_column: python list time in the past you want to use. (1, 2, 5) means (t-1, t-2, t-5) predict time t
        :param start_index: 0- training set, N- valid or testing set
        :param end_index: N-training or valid set, None-testing set
        :param pre_type: 3D for RNN-based, 2D for normal neural network like MLP, FFLN,..
        :return:
        """
        data = []
        labels = []

        history_size = len(history_column)
        if end_index is None:
            end_index = len(dataset[self.multi_size - 1]) - history_column[-1] - 1  # for time t, such as: t-1, t-4, t-7 and finally t
        else:
            end_index = end_index - history_column[-1] - 1

        for i in range(start_index, end_index):
            indices = i - 1 + np.array(history_column)
            # Reshape data from (history_size,) to (history_size, 1)
            data.append([])
            for j in range(self.multi_size):
                for vl in np.reshape(dataset[j][indices], (history_size, 1)):
                    data[i - start_index].append(vl)
            labels.append(dataset[self.multi_size - 1][i + history_column[-1]])
        if pre_type == "3D":
            return np.array(data), np.array(labels)
        return np.reshape(np.array(data), (-1, history_size * self.multi_size)), np.reshape(np.array(labels), (-1, 1))

    # def multivariate_data(self,dataset, history_column=None, start_index=0, end_index=None, pre_type="2D"):
