# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import pickle
from math import radians, cos, sin, asin, sqrt

#wt:这个函数用来计算，两个经纬度之间的距离，单位是米。就是两个坐标之间有多少米。用haversine算。返回的是相距多少米
def haversine1(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])    #map,根据提供的函数对指定的序列做映射
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

start_all = datetime.datetime.now()

# path
path_train = "/data/dm/train.csv"  # 训练文件路径
path_test = "/data/dm/test.csv"  # 测试文件路径

# read train data
data = pd.read_csv(path_train)
train1 = []
#wt:这一句的意思是数有多少个用户（线上的没看，给的demo是100个，可能线上人很多吧）
alluser = data['TERMINALNO'].nunique()

# Feature Engineer, 对每一个用户生成特征:
# trip特征, record特征(数量,state等),
# 地理位置特征(location,海拔,经纬度等), 时间特征(星期,小时等), 驾驶行为特征(速度统计特征等)

#wt:对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
for item in data['TERMINALNO'].unique():

    print('user NO:',item)
    temp = data.loc[data['TERMINALNO'] == item,:]
    temp.index = range(len(temp))     #wt:temp是这一用户的所有数据，然后把index重新弄成了

    # trip 特征
    #wt 每个用户有多少个行程
    num_of_trips = temp['TRIP_ID'].nunique()

    # record 特征
    # wt:shape返回这个dataframe的维度信息，0就是数有多少条记录
    num_of_records = temp.shape[0]

    #wt：这个是算一个用户中，每种电话状态占的比例（算的是百分比）
    num_of_state = temp[['TERMINALNO','CALLSTATE']]
    nsh = num_of_state.shape[0]
    num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE']==0].shape[0]/float(nsh)
    num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE']==1].shape[0]/float(nsh)
    num_of_state_2 = num_of_state.loc[num_of_state['CALLSTATE']==2].shape[0]/float(nsh)
    num_of_state_3 = num_of_state.loc[num_of_state['CALLSTATE']==3].shape[0]/float(nsh)
    num_of_state_4 = num_of_state.loc[num_of_state['CALLSTATE']==4].shape[0]/float(nsh)
    del num_of_state


    ### 地点特征
    #wt:只取了这个用户第一个点的经纬度啊，觉得这里有改善空间（比如，一个用户的不同trip起点的平均？）
    startlong = temp.loc[0, 'LONGITUDE']
    startlat  = temp.loc[0, 'LATITUDE']
    hdis1 = haversine1(startlong, startlat, 113.9177317,22.54334333)  # 距离某一点的距离

    # 时间特征
    #wt:算一个用户的信息中，每个时间占的比例（有多少是8点的，有多少是9点的巴拉巴拉......）
    # temp['weekday'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).weekday())
    temp['hour'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
    hour_state = np.zeros([24,1])
    for i in range(24):
        hour_state[i] = temp.loc[temp['hour']==i].shape[0]/float(nsh)

    # 驾驶行为特征
    mean_speed = temp['SPEED'].mean()      #wt:速度的均值
    var_speed = temp['SPEED'].var()        #wt:速度的无偏方差（均方差）
    mean_height = temp['HEIGHT'].mean()    #wt:高度的均值（觉得这里也可以改进，高度应该算方差的，因为路况可能起起伏伏......）

    # 添加label
    target = temp.loc[0, 'Y']

    # 所有特征
    feature = [item, num_of_trips, num_of_records,num_of_state_0,num_of_state_1,num_of_state_2,num_of_state_3,num_of_state_4,\
               mean_speed,var_speed,mean_height\
        ,float(hour_state[0]),float(hour_state[1]),float(hour_state[2]),float(hour_state[3]),float(hour_state[4]),float(hour_state[5])
        ,float(hour_state[6]),float(hour_state[7]),float(hour_state[8]),float(hour_state[9]),float(hour_state[10]),float(hour_state[11])
        ,float(hour_state[12]),float(hour_state[13]),float(hour_state[14]),float(hour_state[15]),float(hour_state[16]),float(hour_state[17])
        ,float(hour_state[18]),float(hour_state[19]),float(hour_state[20]),float(hour_state[21]),float(hour_state[22]),float(hour_state[23])
        ,hdis1
        ,target]

    train1.append(feature)

train1 = pd.DataFrame(train1)

# 特征命名
featurename = ['item', 'num_of_trips', 'num_of_records','num_of_state_0','num_of_state_1','num_of_state_2','num_of_state_3','num_of_state_4',\
              'mean_speed','var_speed','mean_height'
    ,'h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11'
    ,'h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23'
    ,'dis'
    ,'target']
train1.columns = featurename

print("train data process time:",(datetime.datetime.now()-start_all).seconds)

# Train model
feature_use = ['item', 'num_of_trips', 'num_of_records','num_of_state_0','num_of_state_1','num_of_state_2','num_of_state_3','num_of_state_4',\
               'mean_speed','var_speed','mean_height'
    ,'h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11'
    ,'h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23'
    ,'dis']

params = {
    "objective": 'reg:linear',
    "eval_metric":'rmse',
    "seed":1123,
    "booster": "gbtree",
    "min_child_weight":5,
    "gamma":0.1,
    "max_depth": 3,
    "eta": 0.009,
    "silent": 1,
    "subsample":0.65,
    "colsample_bytree":.35,
    "scale_pos_weight":0.9
    # "nthread":16
}

df_train = xgb.DMatrix(train1[feature_use].fillna(-1), train1['target'])
gbm = xgb.train(params,df_train,num_boost_round=800)

# save model to file
pickle.dump(gbm, open("pima.pickle.dat", "wb"))
print("training end:",(datetime.datetime.now()-start_all).seconds)


# The same process for the test set
data = pd.read_csv(path_test)
test1 = []

for item in data['TERMINALNO'].unique():

    print('user NO:',item)

    temp = data.loc[data['TERMINALNO'] == item,:]
    temp.index = range(len(temp))

    # trip 特征
    num_of_trips = temp['TRIP_ID'].nunique()

    # record 特征
    num_of_records = temp.shape[0]
    num_of_state = temp[['TERMINALNO','CALLSTATE']]
    nsh = num_of_state.shape[0]
    num_of_state_0 = num_of_state.loc[num_of_state['CALLSTATE']==0].shape[0]/float(nsh)
    num_of_state_1 = num_of_state.loc[num_of_state['CALLSTATE']==1].shape[0]/float(nsh)
    num_of_state_2 = num_of_state.loc[num_of_state['CALLSTATE']==2].shape[0]/float(nsh)
    num_of_state_3 = num_of_state.loc[num_of_state['CALLSTATE']==3].shape[0]/float(nsh)
    num_of_state_4 = num_of_state.loc[num_of_state['CALLSTATE']==4].shape[0]/float(nsh)
    del num_of_state


    ### 地点特征
    startlong = temp.loc[0, 'LONGITUDE']
    startlat  = temp.loc[0, 'LATITUDE']

    hdis1 = haversine1(startlong, startlat, 113.9177317,22.54334333)

    # 时间特征
    # temp['weekday'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).weekday())
    temp['hour'] = temp['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
    hour_state = np.zeros([24,1])
    for i in range(24):
        hour_state[i] = temp.loc[temp['hour']==i].shape[0]/float(nsh)

    # 驾驶行为特征
    mean_speed = temp['SPEED'].mean()
    var_speed = temp['SPEED'].var()
    mean_height = temp['HEIGHT'].mean()

    # test标签设为-1
    target = -1.0

    feature = [item, num_of_trips, num_of_records,num_of_state_0,num_of_state_1,num_of_state_2,num_of_state_3,num_of_state_4,\
               mean_speed,var_speed,mean_height\
        ,float(hour_state[0]),float(hour_state[1]),float(hour_state[2]),float(hour_state[3]),float(hour_state[4]),float(hour_state[5])
        ,float(hour_state[6]),float(hour_state[7]),float(hour_state[8]),float(hour_state[9]),float(hour_state[10]),float(hour_state[11])
        ,float(hour_state[12]),float(hour_state[13]),float(hour_state[14]),float(hour_state[15]),float(hour_state[16]),float(hour_state[17])
        ,float(hour_state[18]),float(hour_state[19]),float(hour_state[20]),float(hour_state[21]),float(hour_state[22]),float(hour_state[23])
        ,hdis1
        ,target]

    test1.append(feature)

# make predictions for test data
test1 = pd.DataFrame(test1)
test1.columns = featurename
df_test = xgb.DMatrix(test1[feature_use].fillna(-1))
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
y_pred = loaded_model.predict(df_test)

# output result
result = pd.DataFrame(test1['item'])
result['pre'] = y_pred
result = result.rename(columns={'item':'Id','pre':'Pred'})
result.to_csv('./model/result_.csv',header=True,index=False)

print("Time used:",(datetime.datetime.now()-start_all).seconds)

# '''

