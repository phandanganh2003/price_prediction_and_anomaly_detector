import pandas as pd 
import joblib 

def validate_input(data):
    missing = []

    for key, value in data.items():
        if value in [None, "", 0]:
            missing.append(key)

    return missing

def categorical_encoding(df):
    cat_cols = ['phuong', 'loai_hinh', 'giay_to_phap_ly', 'dac_diem', 'tinh_trang_noi_that']
    trained_columns = joblib.load("Streamlit_project/models/trained_columns.pkl")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.reindex(columns=trained_columns, fill_value=0)
    return df 












# if st.button("Predict"):
#     missing_fields = validate_input(input_data)

#     if missing_fields:
#         st.error(f"⚠️ Thiếu dữ liệu: {', '.join(missing_fields)}")
#     else:
#         st.success("✅ Dữ liệu hợp lệ → chạy model")
