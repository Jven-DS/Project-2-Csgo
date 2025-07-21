#Import thư viện
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyRegressor
from sklearn.impute import SimpleImputer
#EDA Dữ Liệu
df=pd.read_excel('C:/Users/HUU DUY/Downloads/csgo.xlsx')
df.head()
df.info()
df.shape
df.describe()
df.isnull().sum()
#Hệ số tương quan của dữ liệu
labelencoder=LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column]=labelencoder.fit_transform(df[column])
corr=df.corr()
plt.figure(figsize=(15,8))
sns.heatmap(corr,cmap='inferno',annot=True)
plt.show()
df.drop(['date','year','day','month','wait_time_s','ping'],axis=1,inplace=True)
numerical_cols=['kills','mvps']
for col in numerical_cols:
    plt.figure(figsize=(8,6))
    sns.regplot(x=col, y='points', data=df)
    plt.title(f'Mối tương quan giữa {col} và Points')
    plt.xlabel(col)
    plt.ylabel('Points')
    plt.show()
#Độ phân bố dữ 
numerical_cols=['points','mvps','assists']
for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(data=df,x=col)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('count')
    plt.show()
#tạo BoxPlot xác định outliers
sns.set_theme(style="darkgrid")
numerical_cols = ['points', 'kills', 'mvps', 'deaths', 'match_time_s']
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=col, palette="magma")
    plt.title(f'Boxplot of {col}')
    plt.show()
#Loại bỏ Outliers
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]
for col in ['points', 'kills']:
    df = remove_outliers(df, col)
#Xây Model
x=df.drop('points',axis=1)
y=df['points']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
clt=LazyRegressor()
pred_model=clt.fit(x_train, x_test, y_train, y_test)
print(pred_model[0].sort_values(by='R-Squared',ascending=False).head(5))
num_transformer=Pipeline(steps=[
    ('impute',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
    ])
nom_transformer=Pipeline(steps=[
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
preprocessor=ColumnTransformer(transformers=[
    ('num_feature',num_transformer,['match_time_s','team_b_rounds','kills','assists','deaths','mvps','hs_percent']),
    ('nom_feature',nom_transformer,['map','result'])
    ])
from sklearn.linear_model import LinearRegression
model=LinearRegression()
reg=Pipeline(steps=[
    ('pre',preprocessor),
    ('model',model)
    ])
reg.fit(x_train,y_train)
#Dự đoán trên tập kiểm tra
y_pred=reg.predict(x_test)
for i,j in zip (y_pred,y_test):
    print('predict values: {} . actual values: {} '.format(y_pred, y_test))
#Đánh giá kết 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # R²
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")
    # Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
    # Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
    # Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
