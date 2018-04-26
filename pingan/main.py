#encoding:utf-8
import os
import csv
import numpy as np
import pandas as pd
import gc
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.model_selection import train_test_split

"""
#20180426新特征：
#按照用户ID group by
#经纬度每个用户取平均数
#方向取平均数
#海拔，速度取平均数
#电话状态选最多的
#时间也是选最多的
"""


#预测的时候，就不做啥操作了，直接测然后求平均，看看效果

# path_train = "/data/dm/train.csv"  # 训练文件
# path_test = "/data/dm/test.csv"  # 测试文件

path_train = "train.csv"  # 训练文件
path_test = "test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

#找这一列每个用户出现最频繁的那个，groupby不熟，用的要死要活，就先用这个有点笨的方法了
#注意，怕是这个打电话的状态得重做，这个给的demo里，打电话状态0和4最多，可是这两个一点用都没有啊
def find_most_frequence(train_data,columnname):
    #这行的作用是按照用户ID分组，然后每组的每个类别都统计下出现了多少次，每个用户ID内降序排列
    train_data_thiscol=train_data[columnname].groupby(train_data["TERMINALNO"]).value_counts()  #这一行明显的看出来，不同的用户打电话状态还是差很多的
    train_data_thiscol=pd.DataFrame(train_data_thiscol)     #只留第一个
    train_data_thiscol.columns=["num"]
    #print (train_data_thiscol)
    train_data_thiscol.reset_index(inplace=True)
    #print (train_data_call[3].idxmax())
    #print (train_data_thiscol)
    train_data_thiscol.drop_duplicates("TERMINALNO",keep='first',inplace=True)   #只留第一个，第一个就是最多的那个
    train_data_thiscol.index=train_data_thiscol["TERMINALNO"]
    train_data_thiscol.drop(["num","TERMINALNO"],axis=1,inplace=True)
    #print (train_data_thiscol)
    return train_data_thiscol


def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b=pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b=pandas_obj.memory_usage(deep=True)
    usage_mb=usage_b/1024**2
    return ("{:03.2f} MB".format(usage_mb))


def data_process_train(path_train):
    #print('Load data......')
    train_data = pd.read_csv(path_train)
    print(train_data.info(memory_usage='deep'))
    print(mem_usage(train_data))

    #trip_id没有用，去掉
    train_data.drop(['TRIP_ID'], axis=1, inplace=True)

    #方向缺失值处理
    train_data["DIRECTION"].replace(-1, np.nan, inplace=True)  # inplace就是直接在train_data上改了，train_data就会变，不用再赋值回去了
    train_data["DIRECTION"].replace(np.nan, train_data["DIRECTION"].mean(0),
                                    inplace=True)

    #时间处理
    train_columns = list(train_data.columns)  # 获取列名
    train_columns.append("weekday")
    train_columns.append("month")
    train_columns.append("hour")
    train_data.reindex(columns=train_columns)
    train_data["TIME"] = pd.to_datetime(train_data["TIME"], unit='s')  # 把unix时间转成人能看明白的，单位到秒（s）
    # 这个dt太方便了，要啥取啥就行
    train_data["weekday"] = train_data["TIME"].dt.weekday  # 星期，The day of the week with Monday=0, Sunday=6
    train_data["month"] = train_data["TIME"].dt.month
    train_data["hour"] = train_data["TIME"].dt.hour


    #数据压缩，把需要加的列，缺失值处理完再弄这个
    train_data_float = train_data.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')
    train_data_int = train_data.select_dtypes(include=['int64']).apply(pd.to_numeric, downcast='unsigned')
    train_data=pd.concat([train_data_float,train_data_int],axis=1)
    print(train_data.info(memory_usage='deep'))
    print(mem_usage(train_data))

    # 以上是每个用户可以共同弄的操作，还不用group by。下面group by

    #float这种，每个用户取平均数(经纬度，速度，高度，方向，预测值)
    train_data_mean = train_data[["LONGITUDE","LATITUDE","DIRECTION","HEIGHT","SPEED","Y"]].groupby(train_data["TERMINALNO"]).mean()
    #print (train_data_mean)

    #类别这种，每个用户取这个用户中出现最频繁的（电话状态，月，小时，周）
    callstate = find_most_frequence(train_data,"CALLSTATE")
    weekday = find_most_frequence(train_data,"weekday")
    hour = find_most_frequence(train_data, "hour")
    month = find_most_frequence(train_data, "month")
    #print (callstate)
    #print (weekday)
    #print (hour)
    #print (month)
    train_data=pd.concat([train_data_mean,callstate,weekday,hour,month],axis=1)

    # 方向，高度，速度，都做归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data[["DIRECTION", "HEIGHT", "SPEED"]] = min_max_scaler.fit_transform(
        train_data[["DIRECTION", "HEIGHT", "SPEED"]])

    #print (train_data)
    print(train_data.info(memory_usage='deep'))
    print(mem_usage(train_data))

    #速度缺失值处理
    # train_data["SPEED"].replace(-1, np.nan, inplace=True)  # inplace就是直接在train_data上改了，train_data就会变，不用再赋值回去了
    # train_data["SPEED"].replace(np.nan, train_data["SPEED"].mean(0),
    #                                 inplace=True)



    # 把Y值挪到最后一列
    cols = list(train_data.columns)
    cols.insert(len(cols) - 1, cols.pop(cols.index('Y')))
    # print(cols)  # 这个是目前的特征和Y值
    # print (train_data.reindex(columns=cols))
    # 这个copy都不管用啊，只能这样了，可读性有点低
    train_data = train_data.reindex(columns=cols)
    del train_columns
    del train_data_float
    del train_data_int
    del train_data_mean
    del callstate
    del weekday
    del hour
    del month

    gc.collect()
    #print ("there1")
    #print (train_data)
    return train_data

def data_process_test(path_test):
    # print('Load data......')
    test_data = pd.read_csv(path_test)
    print(test_data.info(memory_usage='deep'))
    print(mem_usage(test_data))

    # trip_id没有用，去掉
    test_data.drop(['TRIP_ID'], axis=1, inplace=True)

    # 方向缺失值处理
    test_data["DIRECTION"].replace(-1, np.nan, inplace=True)  # inplace就是直接在test_data上改了，test_data就会变，不用再赋值回去了
    test_data["DIRECTION"].replace(np.nan, test_data["DIRECTION"].mean(0),
                                    inplace=True)

    # 时间处理
    test_columns = list(test_data.columns)  # 获取列名
    test_columns.append("weekday")
    test_columns.append("month")
    test_columns.append("hour")
    test_data.reindex(columns=test_columns)
    test_data["TIME"] = pd.to_datetime(test_data["TIME"], unit='s')  # 把unix时间转成人能看明白的，单位到秒（s）
    # 这个dt太方便了，要啥取啥就行
    test_data["weekday"] = test_data["TIME"].dt.weekday  # 星期，The day of the week with Monday=0, Sunday=6
    test_data["month"] = test_data["TIME"].dt.month
    test_data["hour"] = test_data["TIME"].dt.hour

    # 数据压缩，把需要加的列，缺失值处理完再弄这个
    test_data_float = test_data.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')
    test_data_int = test_data.select_dtypes(include=['int64']).apply(pd.to_numeric, downcast='unsigned')
    test_data = pd.concat([test_data_float, test_data_int], axis=1)
    print(test_data.info(memory_usage='deep'))
    print(mem_usage(test_data))

    # 以上是每个用户可以共同弄的操作，还不用group by。下面group by

    # float这种，每个用户取平均数(经纬度，速度，高度，方向)
    test_data_mean = test_data[["LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED"]].groupby(
        test_data["TERMINALNO"]).mean()
    # print (test_data_mean)

    # 类别这种，每个用户取这个用户中出现最频繁的（电话状态，月，小时，周）
    callstate = find_most_frequence(test_data, "CALLSTATE")
    weekday = find_most_frequence(test_data, "weekday")
    hour = find_most_frequence(test_data, "hour")
    month = find_most_frequence(test_data, "month")
    # print (callstate)
    # print (weekday)
    # print (hour)
    # print (month)
    test_data = pd.concat([test_data_mean, callstate, weekday, hour, month], axis=1)

    # 方向，高度，速度，都做归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    test_data[["DIRECTION", "HEIGHT", "SPEED"]] = min_max_scaler.fit_transform(
        test_data[["DIRECTION", "HEIGHT", "SPEED"]])

    # print (test_data)
    print(test_data.info(memory_usage='deep'))
    print(mem_usage(test_data))

    # 速度缺失值处理
    # test_data["SPEED"].replace(-1, np.nan, inplace=True)  # inplace就是直接在test_data上改了，test_data就会变，不用再赋值回去了
    # test_data["SPEED"].replace(np.nan, test_data["SPEED"].mean(0),
    #                                 inplace=True)




    del test_columns
    del test_data_float
    del test_data_int
    del test_data_mean
    del callstate
    del weekday
    del hour
    del month

    gc.collect()
    # print ("there1")
    # print (test_data)
    return test_data

def process():
    total_data=data_process_train(path_train)
    X_train, X_val, y_train, y_val = train_test_split(total_data.iloc[:, 0:-1], total_data.iloc[:, -1], test_size=0.2,
                                                      random_state=0)
    del total_data
    gc.collect()
    # print (X_train)
    # print (X_test)
    # print (y_train)
    # print (y_test)
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train,categorical_feature=["CALLSTATE","weekday","hour","month"])
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,categorical_feature=["CALLSTATE","weekday","hour","month"])
    del X_train
    del X_val
    del y_train
    del y_val
    gc.collect()
    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose':0
    }
    X_test = data_process_test(path_test)
    gc.collect()
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_eval)

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')

    print('Start predicting...')
    # predict


    out=pd.DataFrame(X_test["TERMINALNO"],columns=["TERMINALNO"])   #单独取出来一列是Series，要变成dataframe
    X_test.drop(["TERMINALNO"],axis=1,inplace=True)
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    out.reindex(columns=["TERMINALNO","y_pred"])
    out["y_pred"]=y_pred
    out_result=pd.DataFrame(out.groupby(["TERMINALNO"])["y_pred"].mean(),columns=["TERMINALNO","y_pred"])
    out_result["TERMINALNO"]=out_result.index

    #print (out_result)
    del out
    gc.collect()
    #out.drop_duplicates('TERMINALNO',inplace=True)  #对ID去重，其实这个不应该这么去重，觉得应该平均，或者统计一下才对
    out_result.to_csv(path_test_out+"result.csv",index=False,header=["Id","Pred"])   #每个用户id做了个平均

    print ("finish!!!")






if __name__ == "__main__":
    #print("****************** start **********************")
    # 程序入口
    process()
