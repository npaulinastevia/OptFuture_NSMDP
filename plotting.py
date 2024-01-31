import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
#sns.barplot(data=df, x="data budgets", y="bug", hue="algorithms")
li = pd.read_csv(r'C:\Users\panou\PycharmProjects\Sequoia\results_cart_testCRL1706181975.txt', index_col=None, header=0)

#new=li.groupby('episodes')['reward'].sum() Cartpolev0  NSCartpolev0  NSCartpolev1  NSCartpolev2
#new=pd.DataFrame(new)
#print(li[li.Environments=='Cartpolev0'].groupby('episodes')['reward'].sum().to_list())
x=li[li.Environments=='Cartpolev0'].groupby('episodes')['reward'].sum().to_list()[:92]
x1=li[li.Environments=='NSCartpolev0'].groupby('episodes')['reward'].sum().to_list()[:92]
x2=li[li.Environments=='NSCartpolev1'].groupby('episodes')['reward'].sum().to_list()[:92]
x3=li[li.Environments=='NSCartpolev2'].groupby('episodes')['reward'].sum().to_list()[:92]

a1=['Cartpolev0']*len(x)
a2=['NSCartpolev0']*len(x)
a3=['NSCartpolev1']*len(x)
a4=['NSCartpolev2']*len(x)
# 'episodes':[i for i in range(len(x))]+[i for i in range(len(x))] +[i for i in range(len(x))] +[i for i in range(len(x))]
data={'Environments':a1+a2+a3+a4,
      'episodes':[i for i in range(len(x))]+[i for i in range(len(x))] +[i for i in range(len(x))] +[i for i in range(len(x))],
      'reward':x+x1+x2+x3}

li=pd.DataFrame(data)
sns.lineplot(data=li, y="reward", x="episodes", hue="Environments")
# plt.legend()
#plt.ylabel("Average cumulative reward", fontsize=15)
#plt.xlabel("Steps", fontsize=15)
#plt.yticks(fontsize=15)
#plt.xticks(fontsize=11)

plt.legend(fontsize=15, loc='upper left')  # (bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()
