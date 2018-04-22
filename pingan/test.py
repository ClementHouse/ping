import pandas as pd
import numpy as np
data=pd.DataFrame({'A':[1,1,2,2],'B':['a','b','a','b']})
print ()
print (data)

data=data.reindex(columns=['A','B','C'])
data['C']=np.array(data['A'])
print (data)