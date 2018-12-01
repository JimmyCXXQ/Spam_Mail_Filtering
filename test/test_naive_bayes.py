from pandas import read_csv
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

test_data_path = "../process_data/test_data.csv"
test_data = read_csv(test_data_path)
x_test = test_data["Message"]
y_test = test_data["Category"]

tf_idf_model_save_path = "../models/tf_idf.pkl"
naive_bayes_save_path = "../models/m_naive_bayes.pkl"

tf_idf = joblib.load(tf_idf_model_save_path)
m_naive_bayes = joblib.load(naive_bayes_save_path)

x_test_matrix = tf_idf.transform(x_test)
predResult = m_naive_bayes.predict(x_test_matrix)
y_test = y_test.astype('int')
actual_Y = y_test.values

print("朴素贝叶斯垃圾邮件分类模型准确率: {0:.4f}".format(accuracy_score(actual_Y, predResult) * 100), "\n")
print("朴素贝叶斯垃圾邮件分类模型F1评分:{0: .4f}".format(f1_score(actual_Y, predResult, average='macro') * 100), "\n")
cmMNb = confusion_matrix(actual_Y, predResult)
print("朴素贝叶斯垃圾邮件分类模型混淆矩阵:\n", cmMNb)
