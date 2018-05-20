#coding:utf-8
import lightgbm as lgb
import numpy as np
import pandas as pd
import datetime
import gc
from math import radians,cos,sin,asin,sqrt
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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
    dataresult=[]
    #print (np.sum(data["CALLSTATE"]==3))
    for user in data["TERMINALNO"].unique():
        #print ("user:",user)
        datatemp=data.loc[data["TERMINALNO"]==user,:]     #iloc只能通过行号索引，但是loc可以用标签。用的时候，选择优先级：iloc>loc>ix。
        datatemp=datatemp.reset_index(drop=True)
        #print (datatemp)
        #有多少条记录
        feature_recordNum = datatemp.shape[0]
        #有多少个行程
        feature_tripNum = datatemp["TRIP_ID"].nunique()
        #每种电话状态所占的比例
        feature_callstate_0 = np.sum(datatemp["CALLSTATE"] == 0) / float(feature_recordNum)
        feature_callstate_1 = np.sum(datatemp["CALLSTATE"] == 1) / float(feature_recordNum)
        feature_callstate_2 = np.sum(datatemp["CALLSTATE"] == 2) / float(feature_recordNum)
        feature_callstate_3 = np.sum(datatemp["CALLSTATE"] == 3)/ float(feature_recordNum)
        feature_callstate_4 = np.sum(datatemp["CALLSTATE"] == 4) / float(feature_recordNum)
        # print (feature_callstate_0)
        # print (feature_callstate_1)
        # print (feature_callstate_2)
        # print (feature_callstate_3)
        # print (feature_callstate_4)
        #每个小时占的比例。这个apply+lambda表达式感觉很好用啊
        datatemp["hour"] = datatemp["TIME"].apply(lambda x:datetime.datetime.fromtimestamp(x).hour)
        feature_hour=np.zeros([24,1])
        for i in range(24):
            feature_hour[i] = np.sum(datatemp["hour"]==i)/float(feature_recordNum)
        #print (feature_hour)

        #经纬度，距离
        feature_distance = haversine1(datatemp.loc[0,"LONGITUDE"],datatemp.loc[0,"LATITUDE"],113.9177317,22.54334333)

        #速度的均值均方差(var是方差，方差标准差就差个平方，感觉没啥事......)
        feature_speedMean = datatemp["SPEED"].mean()
        feature_speedVar = datatemp["SPEED"].var()

        #高度的均值均方差
        feature_heightMean = datatemp["HEIGHT"].mean()
        feature_heightVar = datatemp["HEIGHT"].var()


        feature=[user,feature_recordNum,feature_tripNum,feature_distance,feature_speedMean\
            ,feature_speedVar,feature_heightMean,feature_heightVar\
            ,feature_callstate_0,feature_callstate_1,feature_callstate_2,feature_callstate_3,feature_callstate_4]
        for i in range(24):
            feature.append(feature_hour[i][0])
        feature.append(datatemp.loc[0,"Y"])


        dataresult.append(feature)
        #print (feature)
        #break

    #做列名
    cols=["user","recordNum","tripNum","distance","speedMean","speedVar","heightMean","heightVar"]
    for i in range(5):
        cols.append("callstate_"+str(i))
    for i in range(24):
        cols.append("hour_"+str(i))
    cols.append("Y")

    #print (dataresult)
    dataresult=pd.DataFrame(dataresult,columns=cols)
    #print (dataresult)
    return dataresult



def lgb_predict():
    total_data=feature_process(path_train)
    user_num=total_data.shape[0]
    X_train = total_data.iloc[:,1:-1]     #刨出去用户的id和Y
    y_train=total_data["Y"]
    #train去交叉验证选参数，选出来参数之后，迭代的时候算结果用val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=0)
    del total_data
    gc.collect()
    # specify your configurations as a dict
    #下面开始改参数跑着玩，学习学习如何调参数
    print("start grid search!")
    param_test = {
        'num_leaves': range(5, 80, 5),
        'learning_rate': [0.005],
        'n_estimators': range(100, 800, 100),
        'max_bin': [55],
        'bagging_fraction': [0.8],
        'bagging_freq': [5],
        'feature_fraction': [0.2319],
        'feature_fraction_seed': [9],
        'bagging_seed': [9],
        'min_data_in_leaf': [6],
        'min_sum_hessian_in_leaf': [11]
    }
    estimator = LGBMRegressor(
        objective='regression',
        # 'num_leaves':5,
        # 'learning_rate':0.01,
        # 'n_estimators':720,
        silent=True

    )
    gsearch = GridSearchCV(estimator, param_grid=param_test, scoring='neg_mean_squared_error', cv=3)
    gsearch.fit(X_train, y_train)
    params=gsearch.best_params_
    #这个是python的scikit-learn的API，scikit learn的，好像多参数，和params列表不一样的
    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=params['num_leaves'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        max_bin=params['max_bin'],
        bagging_fraction=params['bagging_fraction'],
        bagging_freq=params['bagging_freq'],
        feature_fraction=params['feature_fraction'],
        feature_fraction_seed=params['feature_fraction_seed'],
        bagging_seed=params['bagging_seed'],
        min_data_in_leaf=params['min_data_in_leaf'],
        min_sum_hessian_in_leaf=params['min_sum_hessian_in_leaf']
    )
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': "regression_l2",
    #     'min_sum_hessian_in_leaf':5,
    #     'min_gain_to_split':0.1,
    #     'num_leaves': 3,
    #     'learning_rate': 0.009,
    #     'feature_fraction': 0.35,
    #     'bagging_fraction': 0.65,
    #     'verbose':0
    # }

    #print (params)
    print('Start training...')
    # train
    model_lgb.fit(X_train,y_train,eval_set=[(X_val,y_val)])

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')

    print('Start predicting...')
    X_test = feature_process(path_test)

    out=X_test[["user"]]
    #print (out)
    #out["y_pred"]=gbm.predict(X_test[["SPEED","hour","weekday","LONGITUDE","LATITUDE","DIRECTION","HEIGHT","CALLSTATE"]], num_iteration=gbm.best_iteration)
    out["y_pred"] = model_lgb.predict(
        X_test.iloc[:,1:-1])
    out.to_csv(path_test_out+"result.csv",index=False,header=["Id","Pred"])   #每个用户id做了个平均

    #print(user_num)

    print (params['num_leaves'])
    print (params['n_estimators'])
    print ("finish!!!")





















if __name__=="__main__":
    #feature_process(path_train)
    lgb_predict()



