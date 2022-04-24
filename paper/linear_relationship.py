#!/usr/bin/env python
# Created by "Thieu" at 15:02, 06/04/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(color_codes=True)

df = pd.read_csv("../data/results_new_MI_new/Surathkal_well_MIR/0-MSE-0/AAEO-MLP/pred_test-1000-50.csv")

sns.regplot(x="y_test_true_unscaled", y="y_test_pred_unscaled", data=df, ci=None)

# sns.pointplot(x="#Lag", y="value", hue='variable', data=pd.melt(df, '#Lag'), kind='point')

# plt.legend(bbox_to_anchor=(-0.1, -0.25, 1, 0.2), loc='lower left', borderaxespad=0, ncol=4)

plt.ylabel('Predicted data')
plt.xlabel('Observed data')

plt.title("AAEO-MLP")

plt.tight_layout()
# plt.savefig("surwell.pdf", bbox_inches="tight")
# plt.savefig("AAEO-MLP_linear_relationship.pdf", bbox_inches="tight")
plt.show()

