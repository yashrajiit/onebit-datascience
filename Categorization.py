# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 13:42:48 2021

@author: yashr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xticks

df = pd.read_csv("D:\\Transactions-Last365days - job1 (1).csv")

df.category.describe()



df.head()

df.columns

sorted(df.category_id.unique())


initc = sorted(df.category.unique())


df0=df.groupby(by = 'category').count()

len(sorted(df.category_id.unique()))

plot1 = df.category.value_counts()
ax = plot1.plot.bar(x='Index', y='category', rot=90)



df['Category Summarised'] = pd.cut(df.category_id,bins=[9999999,
    10999999,12999999, 13999999, 14999999, 15999999, 16999999, 17999999, 
    18999999, 19999999, 20999999, 21999999, 22999999 ],labels=['Bank Fees','Bills and Education',
                                                      'Food and Drinks',
                                                      'Healthcare','Interest Earned',
                                                      'Payment', 'Recreation',
                                                      'Services', 'Shops','Tax',
                                                      'Transfer', 'Travel'])
                                                               
plot2 = df["Category Summarised"].value_counts()                                                        
ax = plot2.plot.bar(x='Index', y='category', rot=90)

finalc = sorted(df['Category Summarised'].unique())

df.to_csv('D:\\Transactions categories summarised.csv')
