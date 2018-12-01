import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split


data_path = "../source_data/spamham.csv"
data = read_csv(data_path)
# print(data.shape) 5572

data.loc[data["Category"] == 'ham', "Category"] = 1
data.loc[data["Category"] == 'spam', "Category"] = 0

X = data['Message']
Y = data['Category']

# 以8:2的比例分割原始数据为训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=4)

# 保存训练集与测试集
train_data = pd.concat([x_train, y_train], axis=1)
train_data.to_csv("../process_data/train_data.csv", index=False)
test_data = pd.concat([x_test, y_test], axis=1)
test_data.to_csv("../process_data/test_data.csv", index=False)

# print(train_data.shape) 4457
# print(test_data.shape) 1115
