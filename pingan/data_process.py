#coding:utf-8
import pandas as pd

"""
原始数据的样子：
# TERMINALNO:用户ID
# TIME:unix时间戳
# TRIP_ID:行程id
# LONGITUDE:经度
# LATITUDE:纬度
# DIRECTION:方向(角度)
# HEIGHT:海拔（m）
# SPEED:速度（km/h）
# CALLSTATE:电话状态
# Y（test里是没有的）:客户赔付率
#
# 用户id：用户唯一标志。
# unix时间戳：从1970年1月1日（UTC/GMT的午夜）开始所经过的秒数，不考虑闰秒。
# 行程id：用户行程唯一标志。
# 经度：用户行程目前所在经度。
# 纬度：用户行程目前所在维度。
# 方向(角度)：用户行程目前对应方向，正北为0，顺时针方向计算角度（如正东为90、正南为180），负值代表此时方向不可判断。
# 海拔(m)：用户行程目前所处的海拔高度。
# 速度(km/h)：用户行程目前的速度。
# 电话状态：用户行程目前的通话状态。（0,未知 1,呼出 2,呼入 3,连通 4,断连）
# 客户赔付率：客户赔付情况，为本次建模的目标Y值。（test中不含此字段）

"""


if __name__=='__main__':
    print ('Load data......')
    train_data=pd.read_csv('PINGAN-2018-train_demo.csv')

    """时间特征处理，抽出了星期，月份，小时，但是还没有做哑编码（one-hot）!!!!!!!"""
    #把数据加上星期，月份，小时这三个空列，后面再补内容
    train_columns = list(train_data.columns)  # 获取列名
    train_columns.append("weekday")
    train_columns.append("month")
    train_columns.append("hour")
    train_data.reindex(columns=train_columns)  # 为了添加星期，月份，小时这三个特征
    print(train_columns)

    #时间处理
    # Unix时间戳，1476923580这样子的，转成2016-10-20 08:33:00这样子的
    train_data["TIME"]=pd.to_datetime(train_data["TIME"],unit='s')   #把unix时间转成人能看明白的，单位到秒（s）
    #这个dt太方便了，要啥取啥就行
    train_data["weekday"]=train_data["TIME"].dt.weekday  #星期，The day of the week with Monday=0, Sunday=6
    train_data["month"] = train_data["TIME"].dt.month
    train_data["hour"] = train_data["TIME"].dt.hour

    print (train_data)
    # print (train_data["TIME"].dt.weekday)
    # print (train_data["TIME"].dt.month)
    # print (train_data["TIME"].dt.hour)
#先做数据预处理，比如归一化，抽特征等等，整全了再划分train和validation
#第一列用户ID不要了
#第二列time弄出来年月日星期小时，感觉月，星期，和小时应该是比较关键的特征
#行程id没想好，感觉不一定要用啊，可能跟用户ID一样？
#经纬度不能归一化或者标准化，就这么放着，不能动，归一化啥的这个信息就丢了
#方向大概得标准化/归一化，还要把负值处理一下，负值是无效信息，算是“缺失值”，要做缺失值处理
#海拔觉得要标准化/归一化
#速度觉得要标准化/归一化
#电话状态要进行哑编码（先用one-hot?）





#自己划一个train和validation吧





#train和validation分出来feature和label，然后这里得到的train和validation送到baseline里面去

