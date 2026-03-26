import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import joblib 
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib

# Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from anomaly_detector import AnomalyDetector

def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.05)
    Q3 = df[col].quantile(0.95)
    IQR = Q3 - Q1
    # Chỉ giữ lại dữ liệu nằm trong "vùng an toàn"
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# làm sạch tên cột
def clean_col_names(df):
    # Thay thế tất cả các ký tự không phải chữ cái/số bằng dấu "_"
    df.columns = [re.sub(r'[^\w]', '_', col) for col in df.columns]

    # Loại bỏ trường hợp nhiều dấu gạch dưới liên tiếp -> 1 dấu
    df.columns = [re.sub(r'_+', '_', col).strip('_') for col in df.columns]
    return df

# Đọc dữ liệu
df=pd.read_csv('data/well_formed_data.csv')
print(f"Kích thước dữ liệu: {df.shape}")

df = remove_outliers_iqr(df, 'gia_ban')
df_transformed = df.copy()
df_transformed['log_gia_vnd'] = np.log1p(df_transformed['gia_ban'])

scaler = StandardScaler()
# StandardScaler để thu nhỏ các số lớn về quanh mốc 0
num_cols = ['dien_tich_dat','so_phong_ngu','so_phong_ve_sinh','tong_so_tang', 'gia_m2_tham_khao', 'chieu_ngang']
df_transformed[num_cols] = scaler.fit_transform(df_transformed[num_cols])

# Mã hóa biến phân loại
cat_cols = ['phuong', 'loai_hinh', 'giay_to_phap_ly', 'dac_diem', 'tinh_trang_noi_that']
df_transformed = pd.get_dummies(df_transformed, columns=cat_cols, drop_first=True)

# Áp dụng
df_transformed = clean_col_names(df_transformed)

X = df_transformed.drop(columns=['gia_ban', 'log_gia_vnd'])
y = df_transformed['log_gia_vnd']

#TRAINING Anomaly 
best_model = joblib.load('C:/DL07_Do_An/Streamlit_project/models/xgboost.pkl')
y_pred_train = best_model.predict(X) 

detector = AnomalyDetector()

detector.fit(
    df_X=X,
    y_true=np.expm1(y),
    y_pred=np.expm1(y_pred_train),
    min_price=500_000_000,
    max_price=50_000_000_000,
)

joblib.dump(detector, "C:/DL07_Do_An/Streamlit_project/models/anomaly_detector.pkl")
print('done')