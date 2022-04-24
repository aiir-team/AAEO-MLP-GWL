#!/usr/bin/env python
# Created by "Thieu" at 08:13, 16/04/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(color_codes=True)


def get_loss_train_file(default_path, model_name):
    working_dir = Path(f"{default_path}/{model_name}")
    list_path = []
    for path in working_dir.glob("pred_test*"):
        # print(path)         # Relative path
        # print(path.absolute())  # OR if you need absolute paths
        # print(path.stem)  # OR if you need only filenames without extension for further parsing
        list_path.append(path)
    return list_path[0]


## https://datagy.io/seaborn-line-plot/
# df_long = pd.read_csv('https://raw.githubusercontent.com/datagy/mediumdata/master/stocks.csv', parse_dates=['Date'])
# sns.lineplot(data=df_long, x='Date', y='Open', hue='Symbol')
# https://stackoverflow.com/questions/55916061/no-legends-seaborn-lineplot
# https://stackoverflow.com/questions/51963725/how-to-plot-a-dashed-line-on-seaborn-lineplot


## Basic model

# models = ["AAEO-MLP", "AEO-MLP", "CGO-MLP"]
# markers = [".", "x", None]
# default_path = "../data/results_new_MI_new/Bantwal_well_MIR/0-MSE-0"
#
# for idx, model in enumerate(models):
#     file_path = get_loss_train_file(default_path, model)
#     df1 = pd.read_csv(file_path)
#     # sns.lineplot(data=df1, x='epoch', y='loss', label=model, marker=markers[idx], linestyle='--')
#     data_points = range(1, len(df1['y_test_true_unscaled'])+1)
#     sns.lineplot(data=df1, x=data_points, y='y_test_true_unscaled', label="Observed")
#     sns.lineplot(data=df1, x=data_points, y='y_test_pred_unscaled', label="Predicted", linestyle='--')
#
#     plt.ylabel('GWL')
#     plt.xlabel('Testing dataset')
#     plt.title('Observed verse predicted values')
#
#     plt.tight_layout()
#     # plt.savefig("surwell.pdf", bbox_inches="tight")
#     # plt.savefig("AAEO-MLP_linear_relationship.pdf", bbox_inches="tight")
#     plt.show()


## Full model

models = ["GA-MLP", "DE-MLP", "PSO-MLP", "HHO-MLP", "SSA-MLP", "HGS-MLP", "MVO-MLP", "EFO-MLP", "EO-MLP",
          "CHIO-MLP", "FBIO-MLP", "SMA-MLP", "CGO-MLP", "AEO-MLP", "IAEO-MLP", "EAEO-MLP", "MAEO-MLP", "AAEO-MLP"]

data_files = ["Ganjimutt_MIR_05", "Surathkal_MIR_04"]
data_titles = ["Ganjimutt well", "Surathkal well"]
data_names = ["ganjimutt", "surathkal"]
trials = 10

for idx_data, data_file in enumerate(data_files):

    for trial in range(0, trials):
        default_path = f"../data/results_paper_minmax/{data_file}/0-MSE-{trial}"

        for idx, model in enumerate(models):
            file_path = get_loss_train_file(default_path, model)
            df1 = pd.read_csv(file_path)
            data_points = range(1, len(df1['y_test_true_unscaled'])+1)
            sns.lineplot(data=df1, x=data_points, y='y_test_true_unscaled', label="Observed")
            sns.lineplot(data=df1, x=data_points, y='y_test_pred_unscaled', label="Predicted", linestyle='--')

            plt.ylabel('GWL (m)')
            plt.xlabel('Time (month)')
            plt.title(f'{model} (Testing dataset of {data_titles[idx_data]})')

            # plt.tight_layout()
            plt.savefig(f"prediction/prediction-{model}-{trial}-{data_names[idx_data]}.pdf", bbox_inches="tight")
            # plt.show()
            plt.close()

