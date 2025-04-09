# check if match_1/Home.csv has null values
import pandas as pd

df = pd.read_csv("data/match_3/Home.csv")

print(df.isnull().sum())
print(len(df))
print(len([col for col in df.columns if col.endswith('_x') ]))
#plot the ball_x values distribution
import matplotlib.pyplot as plt

plt.hist(df['ball_x'].fillna(10000), bins=100)        
plt.show()

