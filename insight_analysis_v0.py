from insight_analysis import *
import pandas as pd


if __name__ == '__main__':

    pass

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
    print(attribution_detection(data=data))
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
