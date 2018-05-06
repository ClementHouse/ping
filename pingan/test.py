import pandas as pd
import numpy as np
import datetime


if __name__=="__main__":
    x=1476923580
    print (datetime.datetime.fromtimestamp(x))
    print (datetime.datetime.fromtimestamp(x).hour)
    # data=pd.DataFrame({'A':[1,1,2,2],'B':['a','b','a','b']})
    # print ()
    # print (data)
    #
    # data=data.reindex(columns=['A','B','C'])
    # data['C']=np.array(data['A'])
    # print (data)