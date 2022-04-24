#!/usr/bin/env python
# Created by "Thieu" at 21:23, 14/04/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# https://stackabuse.com/seaborn-box-plot-tutorial-and-examples/
# https://www.youtube.com/watch?v=Vo-bfTqEFQk
# https://stackoverflow.com/questions/36092363/seaborn-boxplots-changes-narrows-width-of-boxes-when-a-hue-is-chosen-how-migh
# https://www.geeksforgeeks.org/how-to-use-seaborn-color-palette-to-color-boxplot/


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

## Basic model

# plt.figure(0, (10, 5))
# df = pd.read_excel(f'gan_statistics_final.xlsx', sheet_name="clean")
#
# sns.boxplot(x="model", y="RMSE_test", hue='model', data=df,
#             width=0.5, dodge=False, palette="RdYlGn_r")
# plt.xticks(rotation=45, ha='right')
# plt.legend(loc=(1.04, -0.085))
#
# plt.ylabel('RMSE')
# plt.xlabel('Model')
# plt.title('Boxplot of tested models on RMSE metric')
#
# plt.tight_layout()
# # plt.savefig("surwell.pdf", bbox_inches="tight")
# # plt.savefig("AAEO-MLP_linear_relationship.pdf", bbox_inches="tight")
# plt.show()



## Full model

files = ["gan_statistics_final", "sur_statistics_final"]
well_names = ["Ganjimutt well", "Surathkal well"]
save_names = ["ganjimutt", "surathkal"]

cols_excel = ["MAE_test", "RMSE_test", "R_test", "NNSE_test", "KGE_test", "A20_test"]
cols_name = ["MAE", "RMSE", "R", "NNSE", "KGE", "A20"]

for idx_well, filename in enumerate(files):
    df = pd.read_excel(f'{filename}.xlsx', sheet_name="clean")

    for idx_metric, metric in enumerate(cols_excel):
        plt.figure(0, (10, 5))

        sns.boxplot(x="model", y=metric, hue='model', data=df,
                    width=0.5, palette ="RdYlGn_r", dodge=False)
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc=(1.04, -0.085))

        plt.ylabel(cols_name[idx_metric])
        plt.xlabel(None)
        plt.title(f'Boxplot of implemented models on {cols_name[idx_metric]} metric at {well_names[idx_well]} ')

        plt.tight_layout()
        # plt.savefig("surwell.pdf", bbox_inches="tight")
        plt.savefig(f"boxplot/boxplot-{cols_name[idx_metric]}-{save_names[idx_well]}.pdf", bbox_inches="tight")
        plt.show()

        # plt.tight_layout(pad=1)
        # plt.show()

