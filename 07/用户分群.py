
import os

from sklearn.preprocessing import label_binarize

os.environ['OMP_NUM_THREADS'] = '4'
import pandas as pd
from sklearn.cluster import k_means, KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score, silhouette_score  # ch轮廓系数法
# 1.定义函数，找聚类的质心数k
def dm01_find_k():
     # 1.加载数据集
    df = pd.read_csv('./data/customers.csv')
    df.info()
    print(df.head())
    print(df.describe())
    # 2.定义sse_list.sc_list. 记录不同k值的评估效果
    sse_list = []    # sse只考虑簇内，越小越好
    sc_list = []       # sc考虑簇间和簇内，越大越好
    #  抽取特征
    x =df.iloc[:,3:5]
    print(x)
    # 3.定义for循环训练，测试不同k值的评估效果
    for k in range(2, 20):
        # 4.创建KMeans模型对象
        estimator = KMeans(n_clusters=k,max_iter=100,random_state=23)
        # 5.模型训练
        estimator.fit(x)
        # 6.模型预测
        y_pred = estimator.predict(x)
        # 7.分别把评分记录到列表中
        sse_list.append(estimator.inertia_)
        sc_list.append(silhouette_score(x,y_pred))
    # 8.绘制折线图，看最佳k在哪
    plt.figure(figsize=(20,10))
    plt.plot(range(2, 20), sse_list,  label='sse')
    plt.show()


    plt.figure(figsize=(20,10))
    plt.plot(range(2, 20), sc_list,  label='sc')
    plt.show()


# 2.创建模型对象
def dm02_train_predict_evalate():
    df = pd.read_csv('./data/customers.csv')
    df.info()
    x = df.iloc[:,3:5]
    print(x.head())
    print(x.describe())
    estimator = KMeans(n_clusters=5,max_iter=100,random_state=23)
    estimator.fit(x)
    y_pred = estimator.predict(x)
    print(y_pred)
    plt.scatter(x.values[y_pred == 0, 0], x.values[y_pred == 0, 1],c = 'red', label = 'standard')
    plt.scatter(x.values[y_pred == 1, 0], x.values[y_pred == 1, 1] )
    plt.scatter(x.values[y_pred == 2, 0], x.values[y_pred == 2, 1] )
    plt.scatter(x.values[y_pred == 3, 0], x.values[y_pred == 3, 1] )
    plt.scatter(x.values[y_pred == 4, 0], x.values[y_pred == 4, 1] )

    # print(estimator.cluster_centers_) 五个簇的质心坐标
    # print(estimator.labels_)
    # 绘制五个簇的质心，
    plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1])
    plt.title('cluster of customers')
    plt.xlabel('Annual Income k ')
    plt.ylabel('spending score 1-100')
    plt.legend(loc = 7)
    plt.show()
# 3.测试
if __name__ == '__main__':
    # dm01_find_k()
    dm02_train_predict_evalate()
8