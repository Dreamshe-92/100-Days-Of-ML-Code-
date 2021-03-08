![image](https://user-images.githubusercontent.com/29124038/110324393-dab58400-8050-11eb-8ccc-c82fb0d064fe.png)

第一步：数据预处理
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
dataset = pd.read_csv('studentscores.csv')
dataset.info()
​
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 28 entries, 0 to 27
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   28 non-null     float64
 1   Scores  28 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 576.0 bytes
​
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values
​
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)
```

第二步：训练集使用简单线性回归模型来进行训练
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
```

第三步：预测结果
```python
Y_pred = regressor.predict(X_test)
```

第四步：可视化

训练结果可视化
```python
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()
```

测试结果可视化
```python
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()
```
