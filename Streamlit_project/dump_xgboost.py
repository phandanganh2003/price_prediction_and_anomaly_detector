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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#XGBoost
param_distributions = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_depth": [3,4,5,6,7,8,10],
    "min_child_weight": [1,2,3,5,7],
    "gamma": [0, 0.1, 0.3, 0.5, 1],
    "subsample": [0.6,0.7,0.8,0.9,1],
    "colsample_bytree": [0.6,0.7,0.8,0.9,1],
    "reg_lambda": [0.1,1,5,10],
    "reg_alpha": [0,0.01,0.1,1],
    "n_estimators": [200,300,500,700,1000]
}

print("Performing GridSearchCV...")
print()
final_xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

xgb_grid = RandomizedSearchCV(
    final_xgb_model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=10,
    scoring="r2",
    verbose=1,
    n_jobs=-1
)

#dump 
xgb_grid.fit(X, y)
best_model = xgb_grid.best_estimator_
joblib.dump(best_model, "C:/DL07_Do_An/Streamlit_project/models/xgboost.pkl") 