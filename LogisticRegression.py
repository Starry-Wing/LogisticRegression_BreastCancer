# By-StarWing
# 2023.4.10

import csv
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import copy

# 引入乳腺癌数据集
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

# 设定训练集大小
DATA_NUM = 500
# 梯度下降步长
A = 0.01
# 梯度下降次数(迭代次数)
ITER_NUM = 1000

w_vector = np.array(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
     21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0])  # 初始模型w
b = 0  # 初始模型b


# 读取保存的模型
def read_model():
    save_w_vector = []
    save_b = 0
    save_model_info1 = []
    with open('model.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            save_model_info1.append(row)
    for i in range(30):
        save_w_vector.append(float(save_model_info1[0][i]))
    save_b = float(save_model_info1[0][30])
    return np.array(save_w_vector), save_b


# 读取数据集
def read_csv():
    info = []
    with open("boston_house_prices.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            info.append(row)
    return info


# 获取数据集大小
def get_data_num(data):
    return len(data)


# 获取模型特征数量
def get_vector_num(data):
    return len(data[0])


# 获取输入特征集
def get_x_vector_list(data):
    return np.array(data)


# 获取输出目标集
def get_y_list(target):
    return np.array(target)


# 返回一个随机样本
def get_rand(data, target):
    i = random.randint(0, len(data) - 1)
    return np.array(data[i]), target[i]


# 模型
def model(x_vector):
    Z = np.dot(w_vector, x_vector) + b
    F = 1 / (1 + np.exp(-Z / 5000))
    return F


# 损失函数
def loss_function(x_vector, y):
    L = -y * math.log(model(x_vector), math.e) - (1 - y) * math.log(1 - model(x_vector), math.e)
    return L


# 代价函数
def cost_function(x_vector_list, y_list):
    sum = 0
    for i in range(DATA_NUM):
        sum = sum + loss_function(x_vector_list[i], y_list[i])
    sum = sum / DATA_NUM
    return sum


# 代价函数对w的偏导数:
def cost_w(x_vector_list, y_list, j):
    sum = 0
    for i in range(DATA_NUM):
        sum = sum + (model(x_vector_list[i]) - y_list[i]) * x_vector_list[i][j]
    sum = sum / DATA_NUM
    return sum


# 代价函数对b的偏导数:
def cost_b(x_vector_list, y_list):
    sum = 0
    for i in range(DATA_NUM):
        sum = sum + (model(x_vector_list[i]) - y_list[i])
    sum = sum / DATA_NUM
    return sum


# 梯度下降算法(输入特征x, 输出目标y)
def gradient_descent(x_vector_list, y_list):
    global b, w_vector
    # cost_x = []
    # cost_y = []
    last_cost = 9999999999
    cost = cost_function(x_vector_list, y_list)
    for i in range(ITER_NUM):
        b_temp = b - A * cost_b(x_vector_list, y_list)
        w_vector_temp = copy.deepcopy(w_vector)
        for j in range(w_vector.size):
            w_vector_temp[j] = w_vector[j] - A * cost_w(x_vector_list, y_list, j)
        w_vector = w_vector_temp
        b = b_temp
        last_cost = cost
        cost = cost_function(x_vector_list, y_list)
        # if cost < 10000:
        #     cost_x.append(i)
        #     cost_y.append(cost)
        print("第%d次迭代cost： %f" % (i, cost))
        if last_cost - cost < 0.00000001:
            break
    # plt.plot(cost_x, cost_y)
    # plt.show()
    return w_vector, b


# 进行预测
def get_result(x_vector, y):
    # print("------------随机测试------------")
    cal_y = model(x_vector)
    # print("预期输出: ", y)
    # print("计算结果： ", cal_y)
    # print("误差： ", y - cal_y)
    print("恶性肿瘤概率为: {:.2%}".format(cal_y))
    if cal_y >= 0.5:
        print("鉴定为癌症")
    else:
        print("鉴定为良性肿瘤")

    # print("------------------------------")
    if y == 0:
        if cal_y >= 0.5:
            return False
        else:
            return True
    else:
        if cal_y >= 0.5:
            return True
        else:
            return False


# 保存模型至csv文件
def save_model():
    m = np.append(w_vector, [b])
    with open('model.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(m)
    print('已保存至model.csv')


# 训练
def train():
    global w_vector, b
    data = cancer_data['data']
    target = cancer_data['target']
    w_vector, b = read_model()
    x_vector_list = get_x_vector_list(data)
    y_list = get_y_list(target)
    gradient_descent(x_vector_list, y_list)
    print("模型w: ", w_vector)
    print("模型b:", b)
    save_model()


# 批量测试
def test():
    global w_vector, b
    w_vector, b = read_model()
    allnum = 1000
    truenum = 0
    for i in range(allnum):
        x_vector, y = get_rand(cancer_data['data'], cancer_data['target'])
        print("--------------------------")
        print("测试样本编号: ", i)
        if get_result(x_vector, y):
            truenum = truenum + 1
            print("预测正确")
        else:
            print("预测错误")
        print("------------------------")
    print("----------------测试结束-----------------")
    print("测试样本个数: ", allnum)
    print("预测正确数: ", truenum)
    print("正确率: {:.2%}".format(truenum / allnum))


# train()
test()
