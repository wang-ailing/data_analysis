from insight_analysis import *
import pandas as pd

def test():
    import numpy as np
    from scipy import stats
    from sklearn.linear_model import LinearRegression

    # 假设 x 是你的数据列表
    x = np.array([1, 1, 3, 4, 1])

    # 步骤1: 数据排序
    x_sorted = np.sort(x)[::-1]

    # 步骤2: 幂律回归分析
    # 创建指数 i（从1开始，因为这里是降序的）
    i = np.arange(1, len(x_sorted) + 1)
    # 构建幂律模型的输入，这里 β 被固定为 0.7
    X = i ** (-0.7)
    # 使用线性回归模型拟合数据
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), x_sorted)

    # 步骤3: 高斯模型训练
    # 计算残差
    residuals = x_sorted - model.predict(X.reshape(-1, 1))
    # 计算残差的均值和标准差
    mu, sigma = np.mean(residuals), np.std(residuals)

    # 步骤4: 预测 x_max 和计算残差 R
    # 预测 x_max，这里假设 x_max 是排序后的第一个元素
    x_max_prediction = model.predict(np.array([1]).reshape(-1, 1))
    R = x_sorted[0] - x_max_prediction[0]

    # 步骤5: 计算 p 值
    # 使用正态分布的累积分布函数（CDF）来计算 p 值
    p_value = 1 - stats.norm.cdf(R, mu, sigma)

    # 输出结果
    print(f"The p-value is: {p_value}")

def test2():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # 假设 data 是你的数据列表
    data = np.array([1, 1, 3, 4, 1])

    # 对数据进行排序
    sorted_data = np.sort(data)[::-1]
    ranks = np.arange(1, len(sorted_data) + 1)

    # 计算Zipf's Law的拟合参数
    params, _ = curve_fit(lambda r, a, b: a / (r ** b), ranks, sorted_data)

    # 绘制原始数据的累积频率图
    plt.figure(figsize=(10, 6))
    plt.step(ranks, sorted_data, where='post', label='Data')
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Frequency')

    # 绘制Zipf's Law拟合曲线
    zipf_law = lambda r: params[0] / (r ** params[1])
    plt.step(ranks, zipf_law(ranks), where='post', label='Zipf\'s Law Fit')

    plt.legend()
    plt.title('Cumulative Frequency vs. Rank')
    plt.show()

if __name__ == '__main__':

    pass

    # 根据不同{data.iloc[:, 0].name}的{data.iloc[:, 1].name}数据，

    data = Series([2, 4, 6, 8, 500])
    print(data)
    print(single_point_schema_check(data))
    print(attribution_detection(data))
    print(outstanding_1_detection(data, threshold=1/5))
    print(outstanding_2_detection(data, threshold=2/5))
    print(outstanding_last_detection(data))
    print(evenness_detection(data))
    print("-----------------------------------------------------------")
    data = Series([2, 4, 6, 30, 50])
    print(data)
    print(single_point_schema_check(data))
    print(attribution_detection(data))
    print(outstanding_1_detection(data, threshold=1/5))
    print(outstanding_2_detection(data, threshold=2/5))
    print(outstanding_last_detection(data))
    print(evenness_detection(data))
    print("-----------------------------------------------------------")
    data = Series([5, 4, 4, 5, 5])
    print(data)
    print(single_point_schema_check(data))
    print(attribution_detection(data))
    print(outstanding_1_detection(data, threshold=1/5))
    print(outstanding_2_detection(data, threshold=2/5))
    print(outstanding_last_detection(data))
    print(evenness_detection(data))
    print("-----------------------------------------------------------")
    data = Series([5, -2, -4, -6, -8, -100])
    print(data)
    print(single_point_schema_check(data))
    print(attribution_detection(data))
    print(outstanding_1_detection(data, threshold=1/5))
    print(outstanding_2_detection(data, threshold=2/5))
    print(outstanding_last_detection(data))
    print(evenness_detection(data))
    print("-----------------------------------------------------------")
    # Test the evenness function
    # df = pd.DataFrame({'model': [1, 2, 3, 4, 5], 'sales': [996, 997, 996, 998, 997]})
    # print(df)
    # print(evenness_detection(df))


    # df = pd.DataFrame({'date': ['2021.01.01', '2021.01.02', '2021.01.03', '2021.01.04', '2021.01.05'], 'sales': [2.4, 4, 6, 8, 10]})
    # print(single_shape_schema_check(df))
