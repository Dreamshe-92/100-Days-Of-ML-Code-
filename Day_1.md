# 100-Days-Of-ML-Code-
# 数据预处理

<p align="center">
  <img src="https://github.com/MachineLearning100/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%201.jpg">
</p>


准备工作：

确认我们使用的第三方库的版本信息
numpy                              1.19.2
pandas                              1.1.3
sklearn                               0.23.2


第一步：导入需要的库
import numpy as np
import pandas as pd


第二步：导入数据集
dataset = pd.read_csv('Data.csv')//读取csv文件
​
Country  Age  Salary  Purchased
0  France  44.0  72000.0  No
1  Spain  27.0  48000.0  Yes
2  Germany  30.0  54000.0  No
3  Spain  38.0  61000.0  No
4  Germany  40.0  NaN  Yes
5  France  35.0  58000.0  Yes
6  Spain  NaN  52000.0  No
7  France  48.0  79000.0  Yes
8  Germany  50.0  83000.0  No
9  France  37.0  67000.0  Yes
​
X = dataset.iloc[ : , :-1].values//.iloc[行，列]
Y = dataset.iloc[ : , 3].values  // : 全部行 or 列；[a]第a行 or 列
                                 // [a,b,c]第 a,b,c 行 or 列
X
array([['France', 44.0, 72000.0],
       ['Spain', 27.0, 48000.0],
       ['Germany', 30.0, 54000.0],
       ['Spain', 38.0, 61000.0],
       ['Germany', 40.0, nan],
       ['France', 35.0, 58000.0],
       ['Spain', nan, 52000.0],
       ['France', 48.0, 79000.0],
       ['Germany', 50.0, 83000.0],
       ['France', 37.0, 67000.0]], dtype=object)
 
 Y
[[44.0 72000.0]
 [27.0 48000.0]
 [30.0 54000.0]
 [38.0 61000.0]
 [40.0 nan]
 [35.0 58000.0]
 [nan 52000.0]
 [48.0 79000.0]
 [50.0 83000.0]
 [37.0 67000.0]] 


第三步：处理丢失数据

从上面的dataset数据和Y的数据都是存在nan(即空值情况)，这个空值会影响我们的后续的分析，采用sklearn.impute.Simplelmputer 数据填充
#from sklearn.preprocessing import Imputer（老版本的sklearn调用方式，目前很多教程都是这个调用方式）
from sklearn.impute  import SimpleImputer
​
'''
strategy:也就是你采取什么样的策略去填充空值，总共有4种选择。
分别是mean,median, most_frequent,以及constant，
这是对于每一列来说的，如果是mean，则该列则由该列的均值填充。
而median,则是中位数，most_frequent则是众数。
需要注意的是，如果是constant,则可以将空值填充为自定义的值，
这就要涉及到后面一个参数了，也就是fill_value。
如果strategy=‘constant’,则填充fill_value的值。
'''
​
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
​
X[ : , 1:3]
array([[44.0, 72000.0],
       [27.0, 48000.0],
       [30.0, 54000.0],
       [38.0, 61000.0],
       [40.0, 63777.77777777778],
       [35.0, 58000.0],
       [38.77777777777778, 52000.0],
       [48.0, 79000.0],
       [50.0, 83000.0],
       [37.0, 67000.0]], dtype=object)
​


第四步：解析分类数据

这个的目的是为了将类别或者是文字的资料转换为数字，进而让程序更好的去理解及运算
'''
LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码
OneHotEncoder将每一个分类特征变量的m个可能的取值转变成m个二值特征，
对于每一条数据这m个值中仅有一个特征值为1，其他的都为0。
'''
​
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
​
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
​
X[ : , 0] #国家转换为对应的数字
array([0, 2, 1, 2, 1, 0, 2, 0, 1, 0], dtype=object)
​
onehotencoder = ColumnTransformer([("Country",OneHotEncoder(),[0])],remainder='passthrough')
X = onehotencoder.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
​


第五步：拆分数据集为训练集合和测试集合
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)


第六步：特征量化
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
