import pandas as pd
from pandas import DataFrame
from typing import Tuple, List, Union, Dict
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import ruptures as rpt

def single_shape_schema_check(data: DataFrame) -> bool:
    """
    Check if the data has the correct schema for a single shape.

    Args:
        data (DataFrame): DataFrame with columns 'date' and 'value'

    Returns:
        bool: True if the data has the correct schema, False otherwise.
    """

    try:
        data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d') 
        # date example: '2021-01-01'
        if data.iloc[:, 1].dtype != 'float64':
            data.iloc[:, 1] = data.iloc[:, 1].astype(float)
    except Exception :
        return False

    return True


def change_point_detection(data: DataFrame, method: str = 'rbf', jump: int = 1, penalty: float = 1.0) -> Tuple[List[int], List[float]]:
    values = data.iloc[:, 1].values
    # change point detection
    model = "l2"  # "l1", "l2", "rbf"
    algo = rpt.Pelt(model=model, min_size=7, jump=1).fit(values)
    my_bkps = algo.predict(pen=5)
    # show results
    print(my_bkps)
    result = my_bkps[:-1]
    # 显示结果
    plt.plot(values)
    for bkp in result:
        plt.axvline(x=bkp, color='r', linestyle='--')
    plt.show()

def change_point_detection_rbf(data: DataFrame, method: str = 'rbf', jump: int = 1, penalty: float = 1.0) -> Tuple[List[int], List[float]]:
    values = data.iloc[:, 1].values
    # change point detection
    model = "rbf"  # "l2", "rbf"
    algo = rpt.Pelt(model=model, min_size=1, jump=1).fit(values)
    my_bkps = algo.predict(pen=1)
    # show results
    print(my_bkps)
    result = my_bkps[:-1]
    # 显示结果
    plt.plot(values)
    for bkp in result:
        plt.axvline(x=bkp, color='r', linestyle='--')
    plt.savefig('change_point_detection_model_rbf_1.png', dpi=800, bbox_inches='tight')
    plt.show()

def change_point_detection_test(data: DataFrame, detection_window_length: int = 3, threshold: float = 0.05) -> List:

    length = len(data.iloc[:, 1])
    data_list = data.iloc[:, 1].values

    for i in range(length):
        left_window = data_list[max(i-detection_window_length,0):i]
        right_window = data_list[i+1:min(i+detection_window_length+1,length)]
        if len(left_window) == 0 or len(right_window) == 0 or len(right_window) < detection_window_length or len(left_window) < detection_window_length:
            continue
        print(left_window, right_window)
        whole_window = np.concatenate((left_window, right_window))
        # print(whole_window)
        # whole_window = data_list[i-detection_window_length:i+detection_window_length+1]
        print(whole_window)

        Y_left = np.mean(left_window)
        Y_right = np.mean(right_window)
        n = detection_window_length
        y = whole_window
        # sigma_Y = np.sqrt(abs((np.sum(y**2)/(2*n)) - (np.sum(y)/(2*n))**2))
        sigma_Y = np.sqrt((np.sum(y**2)/(2*n)) - (np.sum(y)/(2*n))**2)
        print((np.sum(y**2)/(2*n)), (np.sum(y)/(2*n))**2)
        sigma_miuY = sigma_Y / np.sqrt(n)
        k_mean = np.abs(Y_left - Y_right) / sigma_miuY

        print("Y_left:", Y_left, "Y_right:", Y_right, "sigma_Y:", sigma_Y, "sigma_miuY:", sigma_miuY)
        print("k_mean:", k_mean)
        p_value = stats.norm.pdf(k_mean, 0, 1)
        # p_value = 1 - stats.norm.cdf(k_mean, 0, 1)
        print("p_value:", p_value)
        threshold = 0.35
        if p_value > threshold:
            print("Change point detected at index:", data.iloc[i, 0])
            print("i:", i, "k_mean:", k_mean, "p_value:", p_value)
            print("left_window:", left_window, "right_window:", right_window)

    pass

    

def outlier_detection(data: DataFrame) -> Tuple[Dict, str]:
    values = data.iloc[:, 1].values
    # change point detection
    model = "l2"  # "l2", "rbf"
    algo = rpt.Pelt(model=model, min_size=1, jump=3).fit(values)
    my_bkps = algo.predict(pen=3) #惩罚值越大，要求变化越大
    # show results
    print(my_bkps)
    result = my_bkps[:-1]
    # 显示结果
    plt.plot(values)
    for bkp in result:
        plt.axvline(x=bkp, color='r', linestyle='--')
    plt.show()

def seasonality_detection(data: DataFrame) -> Tuple[Dict, str]:
    pass


def trend_detection(data: DataFrame) -> Tuple[Dict, str]:
    pass

if __name__ == '__main__':
    api_return="""{\"content\":{\"names\":[\"\\u5fae\\u535a\"],\"plot_type\":\"line\",\"title\":\"\\u58f0\\u91cf\\u8d8b\\u52bf\",\"x_title\":\"\\u6c7d\\u8f66\",\"xvalues\":[\"2024-01-01\",\"2024-01-02\",\"2024-01-03\",\"2024-01-04\",\"2024-01-05\",\"2024-01-06\",\"2024-01-07\",\"2024-01-08\",\"2024-01-09\",\"2024-01-10\",\"2024-01-11\",\"2024-01-12\",\"2024-01-13\",\"2024-01-14\",\"2024-01-15\",\"2024-01-16\",\"2024-01-17\",\"2024-01-18\",\"2024-01-19\",\"2024-01-20\",\"2024-01-21\",\"2024-01-22\",\"2024-01-23\",\"2024-01-24\",\"2024-01-25\",\"2024-01-26\",\"2024-01-27\",\"2024-01-28\",\"2024-01-29\",\"2024-01-30\",\"2024-01-31\",\"2024-02-01\",\"2024-02-02\",\"2024-02-03\",\"2024-02-04\",\"2024-02-05\",\"2024-02-06\",\"2024-02-07\",\"2024-02-08\",\"2024-02-09\",\"2024-02-10\",\"2024-02-11\",\"2024-02-12\",\"2024-02-13\",\"2024-02-14\",\"2024-02-15\",\"2024-02-16\",\"2024-02-17\",\"2024-02-18\",\"2024-02-19\",\"2024-02-20\",\"2024-02-21\",\"2024-02-22\",\"2024-02-23\",\"2024-02-24\",\"2024-02-25\",\"2024-02-26\",\"2024-02-27\"],\"y_title\":\"\\u5e74\\u4efd\",\"yvalues\":[[4683,6346,5198,3815,5484,3100,5649,777,2025,740,2781,3511,464,535,2265,3917,441,513,1479,1931,425,1144,2401,1424,2697,1663,4292,1919,2860,7074,3034,4745,2298,3840,885,1119,755,642,663,780,255,376,484,552,392,2266,6393,341,1163,7990,3199,1262,833,891,1059,810,1352,0]]},\"isTab\":true,\"type\":\"chart\"}"""
    api_return = json.loads(api_return)
    api_return['content']['xvalues'].pop()
    api_return['content']['yvalues'][0].pop()
    data = DataFrame({'date': api_return['content']['xvalues'], 'value': api_return['content']['yvalues'][0]})
    print(data)

    # print(outlier_detection(data))
    print(change_point_detection_rbf(data))



    # 绘图
    # plt.plot(data['date'], data['value'])
    # plt.show()

    # data = DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-09', '2021-01-10', '2021-01-11', '2021-01-12', '2021-01-13', '2021-01-14', '2021-01-15', '2021-01-16'], 'value': [1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1]})

    # print(change_point_detection(data))
    # print(change_point_detection(data, threshold=0.01, detection_window_length=3))
    # plt.figure(figsize=(12, 6))
    # plt.plot(data['date'], data['value'])
    # plt.rcParams.update({'font.size': 1})
    # plt.gcf().autofmt_xdate()
    # plt.show()