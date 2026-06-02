# 1.导包
from sklearn.neighbors import KNeighborsClassifier

# 2.准备数据集（测试集 和 数据集）
x_train = [[0],[0],[1],[3]]   # 训练集的特征数据
y_train = [0,0,1,1]          # 训练集的标签数据
x_test = [[5]]              # 测试集的特征数据
"""
特征需要[]包裹，因为特征可以有多个特征，是一个二维数组，就算只有一个特征，也要写成二维的；
而标签不需要，因为标签是离散的，标签只有一列
"""
# 3.创建模型对象--KNN模型
model = KNeighborsClassifier(n_neighbors=2, algorithm='kd_tree')
model.fit(x_train, y_train)
"""
kd_tree（k-dimensional tree，k维树）是一种用于高效最近邻搜索的数据结构：
    数据结构为二叉树结构，递归地将k维数据空间划分
工作原理：
每次选择一个维度进行划分,
选择该维度的中位数作为分割点,
递归地在左右子树重复该过程
另外还有
          algorithm = 'auto'      # 自动选择（默认）
          algorithm = 'ball_tree' # 球树，适用于高维数据
          algorithm = 'brute'     # 暴力搜索，小数据集时简单有效
          algorithm = 'kd_tree'   # kd树，中低维度（通常D<20）效率高
"""
# 4.模型训练
model.fit(x_train, y_train)
# 5.模型预测
y_predict = model.predict(x_test)
# 6. 打印预测结果
print(f"预测值为{y_predict}")