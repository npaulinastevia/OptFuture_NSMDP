import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
#sns.barplot(data=df, x="data budgets", y="bug", hue="algorithms")
li = pd.read_csv(r'C:\Users\panou\PycharmProjects\OptFuture_NSMDP\Src\results_cart1706042594.txt', index_col=None, header=0)
sns.lineplot(data=li, y="x1", x="steps", hue="n_actions")
# plt.legend()
#plt.ylabel("Average cumulative reward", fontsize=15)
#plt.xlabel("Steps", fontsize=15)
#plt.yticks(fontsize=15)
#plt.xticks(fontsize=11)

plt.legend(fontsize=15, loc='upper left')  # (bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()
