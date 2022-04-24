#!/usr/bin/env python
# Created by "Thieu" at 15:18, 06/04/2022 ----------%                                                                               
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
    for path in working_dir.glob("loss_train*"):
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
#     sns.lineplot(data=df1, x='epoch', y='loss', label=model, linestyle='--')
#
# plt.ylabel('MSE')
# plt.xlabel('Epoch')
# plt.title('Convergence of training process')
#
# plt.tight_layout()
# # plt.savefig("surwell.pdf", bbox_inches="tight")
# # plt.savefig("AAEO-MLP_linear_relationship.pdf", bbox_inches="tight")
# plt.show()


## Full model

models = ["GA-MLP", "DE-MLP", "PSO-MLP", "HHO-MLP", "SSA-MLP", "HGS-MLP", "MVO-MLP", "EFO-MLP", "EO-MLP",
          "CHIO-MLP", "FBIO-MLP", "SMA-MLP", "CGO-MLP", "AEO-MLP", "IAEO-MLP", "EAEO-MLP", "MAEO-MLP", "AAEO-MLP"]
markers = [".", "x", None]

data_files = ["Ganjimutt_MIR_05", "Surathkal_MIR_04"]
data_names = ["ganjimutt", "surathkal"]
trials = 10

for idx_data, data_file in enumerate(data_files):

    for trial in range(0, trials):
        default_path = f"../data/results_paper_minmax/{data_file}/0-MSE-{trial}"

        for idx, model in enumerate(models):
            file_path = get_loss_train_file(default_path, model)
            df1 = pd.read_csv(file_path)
            # sns.lineplot(data=df1, x='epoch', y='loss', label=model, marker=markers[idx], linestyle='--')
            sns.lineplot(data=df1, x='epoch', y='loss', label=model, linestyle='--')

        plt.legend(loc=(1.04, -0.05))

        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.title('Convergence of training process')

        plt.tight_layout()
        plt.savefig(f"convergence/convergence-{trial}-{data_names[idx_data]}.pdf", bbox_inches="tight")
        plt.show()

