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
#20180504新特征：
#按照TERMINALNO,tripID进行group by。就是要以每个行程为单位
#时间取平均数，看时段（只取小时和星期特征）。这个就不能作为分类特征了
#经纬度取平均数吧，反正不是分类特征
#height，direction用标准差（看起伏，高度起伏大的可能是山地，方向起伏大的可能是山路十八弯）
#speed用平均数，衡量车速快不快
#电话状态就取最多的吧（这个就变成了唯一的分类特征）

预测的时候，也把test这么搞。代码重写，test和train放一起弄特征，跟广告一样
"""


#预测的时候，就不做啥操作了，直接测然后求平均，看看效果

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

# path_train = "train.csv"  # 训练文件
# path_test = "test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。




def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b=pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b=pandas_obj.memory_usage(deep=True)
    usage_mb=usage_b/1024**2
    return ("{:03.2f} MB".format(usage_mb))





#先写明白逻辑，再优化内存
def data_process(path):
    data=pd.read_csv(path)
    #test没有Y这样处理
    if "Y" not in list(data.columns):
        data["Y"]=-1

    # 方向缺失值处理
    data["DIRECTION"].replace(-1, np.nan, inplace=True)  # inplace就是直接在data_data上改了，data_data就会变，不用再赋值回去了
    data["DIRECTION"].replace(np.nan, data["DIRECTION"].mean(),
                                    inplace=True)



    #时间处理
    data_columns = list(data.columns)  # 获取列名
    data_columns.append("weekday")
    data_columns.append("hour")
    data.reindex(columns=data_columns)
    data["TIME"] = pd.to_datetime(data["TIME"], unit='s')  # 把unix时间转成人能看明白的，单位到秒（s）
    # 这个dt太方便了，要啥取啥就行
    data["weekday"] = data["TIME"].dt.weekday  # 星期，The day of the week with Monday=0, Sunday=6
    data["hour"] = data["TIME"].dt.hour


    #分组
    data_group=data.groupby(["TERMINALNO","TRIP_ID"])
    #print (data_group.size())
    #print (data_group.get_group((1,1)))    #提取第一个用户的第一个行程，打出来看看

    # #这里的x是指每个元素。这一段注释掉了，发现这样写太麻烦，放在下边for循环里了
    # SPEED_mean=data_group.apply(lambda x:x["SPEED"].mean())  #对每个行程的速度取平均。得到的是series
    # hour_mean=data_group.apply(lambda x:x["hour"].mean())     #小时平均
    # weekday_mean = data_group.apply(lambda x: x["weekday"].mean())  #星期平均
    # LONGITUDE_mean=data_group.apply(lambda x:x["LONGITUDE"].mean())     #经度求平均
    # LATITUDE_mean=data_group.apply(lambda x:x["LATITUDE"].mean())    #纬度求平均
    # DIRECTION_std=data_group.apply(lambda x:x["DIRECTION"].std())   #方向求标准差
    # HEIGHT_std=data_group.apply(lambda x:x["HEIGHT"].std())        #海拔求标准差


    #处理电话状态
    #这个电话状态还是考虑出现最多的那个，不过觉得这样可能还是不对。效果不好的话这个特征不要了再跑一跑
    callstate=data_group.apply(lambda x:x["CALLSTATE"].value_counts())
    #print (callstate)
    callstate=pd.DataFrame(callstate)
    callstate.reset_index(inplace=True)   #把索引变成了列
    #print (callstate)
    callstate.drop_duplicates(["TERMINALNO","TRIP_ID"],keep='first',inplace=True)
    #print (callstate)
    #print (SPEED_mean)
    #print (type(SPEED_mean))
    #print (weekday_mean)


    data_new = data[["TERMINALNO", "TRIP_ID", "Y"]]
    data_new.drop_duplicates(["TERMINALNO","TRIP_ID"],keep='first',inplace=True)
    data_new=pd.merge(data_new,callstate[["TERMINALNO","TRIP_ID","CALLSTATE"]],on=["TERMINALNO","TRIP_ID"],how='left')    #用[[]]就会取成dataframe，用一个[]会取成series

    #处理好的数据放一起，最小单位为行程ID，每行是一个用户的一个行程。
    mean_feature=["SPEED","hour","weekday","LONGITUDE","LATITUDE"]
    std_feature=["DIRECTION","HEIGHT"]
    for feature in mean_feature:
        mean_result=data_group.apply(lambda x:x[feature].mean())   #得到了一个索引
        mean_result=pd.DataFrame(mean_result)      #把索引变成dataframe
        mean_result.columns=[feature]       #取列名
        mean_result.reset_index(inplace=True)     #把"TERMINALNO","TRIP_ID"变回一列
        data_new=pd.merge(data_new,mean_result,on=["TERMINALNO","TRIP_ID"],how='left')

    for feature in std_feature:
        mean_result = data_group.apply(lambda x: x[feature].std())  # 得到了一个索引
        mean_result = pd.DataFrame(mean_result)  # 把索引变成dataframe
        mean_result.columns = [feature]  # 取列名
        mean_result.reset_index(inplace=True)  # 把"TERMINALNO","TRIP_ID"变回一列
        data_new = pd.merge(data_new, mean_result, on=["TERMINALNO", "TRIP_ID"], how='left')



    #有的方向高度只有一个值，算标准差就是缺失值，这些填成0
    data_new.fillna(0,inplace=True)

    print (mem_usage(data_new))
    #数据压缩
    data_float = data_new.select_dtypes(include=['float64']).apply(pd.to_numeric, downcast='float')
    data_int = data_new.select_dtypes(include=['int64']).apply(pd.to_numeric, downcast='unsigned')
    data_new = pd.concat([data_float, data_int], axis=1)


    del data
    del data_group
    del data_columns
    del callstate
    del mean_feature
    del std_feature
    del data_float
    del data_int
    gc.collect()
    #print (data_new.dtypes)
    print (mem_usage(data_new))
    #print (data_new)
    return data_new





def lgb_predict():
    total_data=data_process(path_train)
    #其实如果参数确定了，应该用所有的train训练模型的。交叉验证用来选lgb参数的，选好了就关了交叉验证，直接用所有的train训练模型。如果有空一定要改
    #X_train, X_val, y_train, y_val = train_test_split(total_data[["SPEED","hour","weekday","LONGITUDE","LATITUDE","DIRECTION","HEIGHT","CALLSTATE"]], total_data[["Y"]], test_size=0.2,random_state=0)
    X_train=total_data[["SPEED","hour","weekday","LONGITUDE","LATITUDE","DIRECTION","HEIGHT","CALLSTATE"]]
    y_train=total_data["Y"]
    lgb_train = lgb.Dataset(X_train, y_train,categorical_feature=["CALLSTATE"])
    #lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,categorical_feature=["CALLSTATE"])
    del X_train
    #del X_val
    del y_train
    del total_data
    #del y_val
    gc.collect()
    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': "regression_l2",
        'min_sum_hessian_in_leaf': 5,
        'min_gain_to_split': 0.1,
        'num_leaves': 3,
        'learning_rate': 0.009,
        'feature_fraction': 0.35,
        'bagging_fraction': 0.65,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=4000,
                    valid_sets=lgb_train)

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')

    print('Start predicting...')
    X_test = data_process(path_test)

    out=X_test[["TERMINALNO","TRIP_ID"]]
    #print (out)
    out["y_pred"]=gbm.predict(X_test[["SPEED","hour","weekday","LONGITUDE","LATITUDE","DIRECTION","HEIGHT","CALLSTATE"]], num_iteration=gbm.best_iteration)
    #print (out)
    # out.reindex(columns=["TERMINALNO","y_pred"])
    # out["y_pred"]=y_pred
    out=pd.DataFrame(out.groupby(["TERMINALNO"])["y_pred"].mean(),columns=["TERMINALNO","y_pred"])
    out["TERMINALNO"]=out.index
    #print (out)
    #
    # #print (out_result)
    # del out
    # gc.collect()
    # #out.drop_duplicates('TERMINALNO',inplace=True)  #对ID去重，其实这个不应该这么去重，觉得应该平均，或者统计一下才对
    out.to_csv(path_test_out+"result.csv",index=False,header=["Id","Pred"])   #每个用户id做了个平均

    print ("finish!!!")















if __name__ == "__main__":
    #print("****************** start **********************")
    # 程序入口
    #data_process(path_test)
    lgb_predict()
    #process()
