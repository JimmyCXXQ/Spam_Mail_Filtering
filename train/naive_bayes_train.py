from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

train_data_path = "../process_data/train_data.csv"
train_data = read_csv(train_data_path)
x_train = train_data["Message"]
y_train = train_data["Category"]

# test_data_path = "../process_data/test_data.csv"
# test_data = read_csv(test_data_path)
# x_test = test_data["Message"]
# y_test = test_data["Category"]

tf_idf = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_matrix = tf_idf.fit_transform(x_train)
# x_test_matrix = tf_idf.transform(x_test)

# 训练朴素贝叶斯
y_train = y_train.astype('int')
m_naive_bayes = MultinomialNB()
m_naive_bayes.fit(x_train_matrix, y_train)

# 性能评分展示
# predResult = m_naive_bayes.predict(x_test_matrix)
# y_test = y_test.astype('int')
# actual_Y = y_test.values

# print("朴素贝叶斯垃圾邮件分类模型准确率: {0:.4f}".format(accuracy_score(actual_Y, predResult) * 100), "\n")
# print("朴素贝叶斯垃圾邮件分类模型F1评分:{0: .4f}".format(f1_score(actual_Y, predResult, average='macro') * 100), "\n")
# cmMNb = confusion_matrix(actual_Y, predResult)
# print("朴素贝叶斯垃圾邮件分类模型混淆矩阵:\n", cmMNb)

# 保存tf_idf模型、朴素贝叶斯模型
tf_idf_model_save_path = "../models/tf_idf.pkl"
naive_bayes_save_path = "../models/m_naive_bayes.pkl"
joblib.dump(tf_idf, tf_idf_model_save_path)
joblib.dump(m_naive_bayes, naive_bayes_save_path)

