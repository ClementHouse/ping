#coding:utf-8
import lightgbm as lgb
import numpy as np
import pandas as pd
import datetime
from math import radians,cos,sin,asin,sqrt

"""
按照每个用户来抽特征：（这个用比例很好，直接归一化了，省事）
1.每个用户有多少记录
2.每个用户有多少行程
3.每个用户每个电话状态占所有状态的比例
4.每个用户第一个经纬度和固定点的距离
5.每个用户的记录中，每个小时所占的比例（8点出现的频率，9点出现的频率巴拉巴拉）
6.速度的均值，均方差
7.高度的均值（我觉得应该加上高度的均方差）

baseline用的xgboost，我还是先用原来的lightgbm，如果还是负分，可能就是模型的参数有毛病了
后续：
经纬度信息，这个只用了每个用户第一条记录的经纬度，感觉应该用每个行程的
方向感觉不管也行，往北往南开关系不大；但是觉得后续也可以加上方向的均方差，方向变动大可能是那种拐弯大的地方，可能会危险......

"""


path_train = "/data/dm/train.csv"  # 训练文件路径
path_test = "/data/dm/test.csv"  # 测试文件路径

# path_train = "train.csv"  # 训练文件
# path_test = "test.csv"  # 测试文件

path_test_out = "model/"


def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b=pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b=pandas_obj.memory_usage(deep=True)
    usage_mb=usage_b/1024**2
    return ("{:03.2f} MB".format(usage_mb))


def haversine1(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2=map(radians,[lon1,lat1,lon2,lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000


def feature_process(path):
    data = pd.read_csv(path)
    #这样test和train可以一起处理
    if "Y" not in list(data.columns):
        data["Y"]=-1

    for user in data["TERMINALNO"].unique():
        datatemp=data.loc[data["TERMINALNO"]==user,:]     #iloc只能通过行号索引，但是loc可以用标签。用的时候，选择优先级：iloc>loc>ix。







if __name__=="__main__":



