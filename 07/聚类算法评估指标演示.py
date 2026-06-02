
import os
os.environ['OMP_NUM_THREADS'] = '4'
import pandas as pd
from sklearn.cluster import k_means, KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score, silhouette_score  # ch轮廓系数法

# 1.定义函数，演示:SSE+肘部法
def dm01_sse():
# 1.1定义SSE列表，记录每个K的SSE值
    sse_list=[]
# 1.2生成数据集   参1:样本数量 参2:特征数量 参3:4个质心 参4:4个簇的std标准差 参5: 固定随机种子
    x,y = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=[[-1,-1],[0,0],[1,1],[2,2]],
    cluster_std=[0.4,0.2,0.2,0.2],
    random_state=23
    )
# 1.3 for循环遍历每个K值，计算其对应的SSE值，并添加到列表中
    for k in range(1,100):
        # 1.3.1 创建kmeans对象，参1：簇数量，参2：最大迭代次数，参3：固定随机种子
        estimator = KMeans(n_clusters=k, max_iter=100,random_state=23)
        # 1.3.2训练模型
        estimator.fit(x)
        # 1.3.3模型预测
        # 1.3.4获取到每个簇的sse值
        sse_value = estimator.inertia_
        # 1.3.5将每一个sse值添加到列表中
        sse_list.append(sse_value)
# 1.4 绘制sse曲线，SSE值可视化
    print(sse_list)
# 1.4.1 创建画布，指定画布的尺寸
# 1.4.2 设置标题
    plt.title("K-Means SSE")
# 1.4.3 设置x轴的刻度
    plt.xticks(range(1,100,3))
# 1.4.4 添加x轴，y轴的标签
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
# 1.4.5 绘制网格
    plt.grid(True)
# 1.4.6 绘制折线图
    plt.plot(range(1,100),sse_list)
# 1.4.7 显示图形
    plt.show()





# 2.定义函数，演示:SC轮廓系数法
def dm02_sc():
# 1.1定义sc列表，记录每个K的sc值
    sc_list=[]
# 1.2生成数据集   参1:样本数量 参2:特征数量 参3:4个质心 参4:4个簇的std标准差 参5: 固定随机种子
    x,y = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=[[-1,-1],[0,0],[1,1],[2,2]],
    cluster_std=[0.4,0.2,0.2,0.2],
    random_state=23
    )
# 1.3 for循环遍历每个K值，计算其对应的sc值，并添加到列表中
    for k in range(2,100):   # 考虑簇外，至少2个簇
        # 1.3.1 创建kmeans对象，参1：簇数量，参2：最大迭代次数，参3：固定随机种子
        estimator = KMeans(n_clusters=k, max_iter=100,random_state=23)
        # 1.3.2训练模型
        estimator.fit(x)
        # 1.3.3模型预测
        y_pred = estimator.predict(x)
        # 1.3.4获取到每个簇的sc值
        sc_value = silhouette_score(x, y_pred)
        # 1.3.5将每一个sc值添加到列表中
        sc_list.append(sc_value)
# 1.4 绘制sc曲线，sc值可视化
    print(sc_list)
# 1.4.1 创建画布，指定画布的尺寸
# 1.4.2 设置标题
    plt.title("K-Means sc")
# 1.4.3 设置x轴的刻度
    plt.xticks(range(1,100,3))
# 1.4.4 添加x轴，y轴的标签
    plt.xlabel("Number of clusters")
    plt.ylabel("sc")
# 1.4.5 绘制网格
    plt.grid(True)
# 1.4.6 绘制折线图
    plt.plot(range(2,100),sc_list)
# 1.4.7 显示图形
    plt.show()




# 3.定义函数，演示:CH轮廓系数法
def dm03_ch():
# 1.1定义ch列表，记录每个K的ch值
    ch_list=[]
# 1.2生成数据集   参1:样本数量 参2:特征数量 参3:4个质心 参4:4个簇的std标准差 参5: 固定随机种子
    x,y = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=[[-1,-1],[0,0],[1,1],[2,2]],
    cluster_std=[0.4,0.2,0.2,0.2],
    random_state=23
    )
# 1.3 for循环遍历每个K值，计算其对应的ch值，并添加到列表中
    for k in range(2,100):   # 考虑簇外，至少2个簇
        # 1.3.1 创建kmeans对象，参1：簇数量，参2：最大迭代次数，参3：固定随机种子
        estimator = KMeans(n_clusters=k, max_iter=100,random_state=23)
        # 1.3.2训练模型
        estimator.fit(x)
        # 1.3.3模型预测
        y_pred = estimator.predict(x)
        # 1.3.4获取到每个簇的ch值
        ch_value = calinski_harabasz_score(x, y_pred)
        # 1.3.5将每一个ch值添加到列表中
        ch_list.append(ch_value)
# 1.4 绘制ch曲线，ch值可视化
    print(ch_list)
# 1.4.1 创建画布，指定画布的尺寸
# 1.4.2 设置标题
    plt.title("K-Means ch")
# 1.4.3 设置x轴的刻度
    plt.xticks(range(1,100,3))
# 1.4.4 添加x轴，y轴的标签
    plt.xlabel("Number of clusters")
    plt.ylabel("ch")
# 1.4.5 绘制网格
    plt.grid(True)
# 1.4.6 绘制折线图
    plt.plot(range(2,100),ch_list)
# 1.4.7 显示图形
    plt.show()
# 4.测试
if __name__ == '__main__':
    # dm01_sse()
    # dm02_sc()
    dm03_ch()