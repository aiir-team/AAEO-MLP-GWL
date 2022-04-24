#!/usr/bin/env python
# Created by "Thieu" at 17:47, 05/04/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


# df = pd.read_csv("bantwell.csv")
# df = pd.read_csv("surathkal_well.csv")
df = pd.read_csv("ganjimutt.csv")

# 1. category
# sns.scatterplot(x="thieu", y="lam", data=dataset)

# 2. multiple categories
# sns.catplot(x="#Lag", y=["Rainfall", "Tidal-height", "Mean-temperature", "DGWL"], data=dataset, kind='point')

sns.pointplot(x="#Lag", y="value", hue='variable', data=pd.melt(df, '#Lag'), kind='point')

plt.legend(bbox_to_anchor=(-0.1, -0.25, 1, 0.2), loc='lower left', borderaxespad=0, ncol=4)
plt.ylabel('MI value')
# plt.title("Bantwal dataset")
# plt.title("Surathkal well")
plt.title("Ganjimutt well")
plt.tight_layout()
# plt.savefig("surwell.pdf", bbox_inches="tight")
# plt.savefig("bantwell.pdf", bbox_inches="tight")
plt.savefig("genjimutt.pdf", bbox_inches="tight")
plt.show()



