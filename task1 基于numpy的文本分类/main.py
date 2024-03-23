import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # 读入训练集，取出输入和标签
    train = pd.read_csv('./data/train.tsv',sep='\t')
    X_data, y_data = train["Phrase"].values, train["Sentiment"].values

    # 用于测试,只取前20000个样本
    just_test = 1
    if just_test == 1:
        X_data = X_data[:20000]
        y_data = y_data[:20000]

    # y = np.array(y_data).reshape((-1,1))

    # 用词袋模型表示文本特征
    vectorizer1 = CountVectorizer()
    X_bow = vectorizer1.fit_transform(X_data)
    # 用N-gram表示文本特征
    vectorizer2 = CountVectorizer(ngram_range=(1, 2))
    X_ngram = vectorizer2.fit_transform(X_data)

    # 划分训练集和测试集
    X_train_bow, X_test_bow,y_train_bow,y_test_bow = train_test_split(X_bow,y_data,test_size=0.2,random_state=42,stratify=y_data)
    X_train_ngram, X_test_ngram,y_train_ngram,y_test_ngram = train_test_split(X_ngram,y_data,test_size=0.2,random_state=42,stratify=y_data)

    # softmax回归
    model1 = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    # 训练模型（词袋模型）
    model1.fit(X_train_bow,y_train_bow)
    # 预测
    y_pred_bow = model1.predict(X_test_bow)
    # 计算准确率
    accuracy_bow = accuracy_score(y_test_bow, y_pred_bow)
    print(f'Bow test set accuracy: {accuracy_bow}')

    model2 = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    # 训练模型（词袋模型）
    model2.fit(X_train_ngram,y_train_ngram)
    # 预测
    y_pred_ngram = model2.predict(X_test_ngram)
    # 计算准确率
    accuracy_ngram = accuracy_score(y_test_ngram, y_pred_ngram)
    print(f'N-gram test set accuracy: {accuracy_ngram}')
