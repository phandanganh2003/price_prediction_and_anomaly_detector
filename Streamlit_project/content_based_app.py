import streamlit as st
from pathlib import Path
import pandas as pd 
import numpy as np
import base64
from utils import validate_input, categorical_encoding 
import joblib 

st.set_page_config(page_title="NhaTot App", layout="wide")

IMG_PATH = Path(__file__).resolve().parent / "images" / "nha_tot.png"

def to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_b64 = to_base64(IMG_PATH)

st.markdown("""
<style>
/* ===== Toàn trang ===== */
html, body, [class*="css"]  {
    font-family: Arial, sans-serif;
}

/* Giảm khoảng trắng trên/dưới của main page */
.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 1rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1200px;
}

/* Main area fit đẹp hơn */
.main-wrap {
    width: 100%;
    max-width: 1100px;
    margin: 0 auto;
}

/* Hero section */
.hero-wrap {
    width: 100%;
    text-align: center;
    margin-top: 0.2rem;
    margin-bottom: 1.2rem;
}

.hero-logo {
    width: min(220px, 28vw);
    height: auto;
    display: inline-block;
    margin-bottom: 0.5rem;
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    color: #2f3142;
    line-height: 1.15;
    margin: 0.2rem 0 1rem 0;
}

/* Banner */
.info-banner {
    background: #e8f1fb;
    color: #0f4f96;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    font-size: 1rem;
    margin: 0.8rem 0 1.2rem 0;
}

/* Button */
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 0.75rem 0.8rem;
    font-size: 1rem;
}

/* Section title */
.section-title {
    font-size: clamp(1.4rem, 2.2vw, 2rem); 
    font-weight: 700;                  
    color: #2f3142;
    margin: 0.8rem 0 0.5rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #eef0f4;
}

.sb-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1f77c8;
    margin-bottom: 1rem;
}

.sb-divider {
    border-top: 1px solid #cfd6df;
    margin: 1rem 0;
}

.sb-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #243447;
    text-align: center;
    margin-top: 0.2rem;
}

.sb-email {
    font-size: 0.95rem;
    color: #5f6f82;
    text-align: center;
    margin-bottom: 0.8rem;
    overflow-wrap: anywhere;
}

.sb-advisor-title {
    font-size: 0.95rem;
    font-style: italic;
    color: #8a6d3b;
    text-align: center;
}

.sb-advisor-name {
    font-size: 1rem;
    font-style: italic;
    color: #6c6c6c;
    text-align: center;
    margin-top: 0.35rem;
}
</style>
""", unsafe_allow_html=True)

# ===== Sidebar =====
with st.sidebar:
    st.markdown('<div class="sb-title">👥 Team Members</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-name">Phan Đặng Anh</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-email">phandanganh2003@gmail.com</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-name">Tang Đông Hy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-email">hyyhtang696969@gmail.com </div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-name">Phó Quốc Dũng</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-email">phoqdung89@gmail.com </div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-advisor-title">🎓 Giảng Viên Hướng Dẫn:</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-advisor-name">Khuất Thùy Phương</div>', unsafe_allow_html=True)

# ===== Main wrapper =====
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

st.markdown(f"""
<div class="hero-wrap">
    <img class="hero-logo" src="data:image/png;base64,{img_b64}">
    <div class="hero-title">Price prediction System and anomaly detector</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Content</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["About", "Tools", "Visualization"])

with tab1:
    st.subheader("Business Problem")
    st.markdown("""
    **Bối cảnh:** Dữ liệu nhà ở được đăng bán trên Nhà Tốt.  

    **Nhu cầu:**
    - **Người bán:** cần gợi ý giá hợp lý, không bị “hớ” → Đồng thời cung cấp giá phù hợp cho người mua.  
    - **Nền tảng/kiểm duyệt:** cần cảnh báo tin đăng bài bán nhà có giá bất thường để xử lý.  

    **Xác định vấn đề:**
    - **Mục tiêu / vấn đề:**
        - Xây dựng mô hình dự đoán & gợi ý giá đăng bán hợp lý  
        - Phát hiện bất thường: bài đăng đưa giá quá thấp / quá cao  
    - **Xây dựng mô hình:**
        - Dự đoán giá  
        - Phát hiện bất thường  
    """)

    st.subheader("Data description") 
    st.markdown("""
    **Tập dữ liệu:** Sử dụng dữ liệu Nhà Tốt trên địa bàn 3 quận Bình Thạnh, Gò Vấp, và Phú Nhuận.  

    **Số mẫu:** ~ 7900 mẫu dữ liệu (sau khi xử lý)
    """)

    st.subheader("Core Files/Folders Used") 
    st.markdown("""
    **📁 data/**
    - Chứa dữ liệu đã xử lý sạch phục vụ training & inference  
    - `well_formed_data.csv`: dataset chính dùng cho mô hình  

    **📁 images/**
    - Lưu trữ hình ảnh dùng cho giao diện Streamlit  

    **📁 models/**
    - Chứa các model đã train (XGBoost, scaler, artifacts...)  

    ---

    **📄 anomaly_detector.py**  
    - Class chính xử lý **phát hiện bất thường (anomaly detection)**  
    - Tính toán anomaly score & xác định threshold  

    **📄 content_based_app.py**  
    - File chính chạy **Streamlit app**  
    - Nhận input từ user → gọi model → hiển thị kết quả  

    **📄 dump_xgboost.py**  
    - Script lưu model XGBoost sau khi training (`.pkl`)  

    ---

    **📄 train_anomaly.py**  
    - Script train hệ thống anomaly detection  

    **📄 utils.py**  
    - Các hàm hỗ trợ chung: xử lý data, logging, helper functions  
    
    ---
    """)

with tab2:
    st.markdown('<div class="section-title">Tools</div>', unsafe_allow_html=True)

    st.markdown("#### 🏠 Your house information")   # nhỏ hơn subheader

    st.write("Nhập mô tả của bạn")
    mo_ta = st.text_area(
    "",
    placeholder="e.g., gần đường lớn, căn góc, mặt tiền...",
    height=100
)
    col1, col2 = st.columns(2)
    with col1:
        dien_tich_dat = st.number_input("Diện tích (m2)")
        so_phong_ngu = st.number_input("Số phòng ngủ", min_value=0)
        so_phong_ve_sinh = st.number_input("Số phòng vệ sinh", min_value=0)
    with col2:
        tong_so_tang = st.number_input("Tổng số tầng", min_value=0)
        chieu_ngang = st.number_input("Chiều ngang (m)", min_value=0.0, step=0.1, format = "%.2f", value=0.0)
        gia_m2_tham_khao = st.number_input("Giá m2 khu vực xung quanh", min_value=0.0, step=0.1, format = "%.2f", value=0.0)

    phuong = st.selectbox("Phường", ['Phường Gia Định', 'Phường Bình Lợi Trung', 'Phường Thạnh Mỹ Tây',
       'Phường Bình Thạnh', 'Phường Bình Quới', 'Phường An Hội Tây',
       'Phường An Hội Đông', 'Phường Thông Tây Hội', 'Phường Gò Vấp',
       'Phường An Nhơn', 'Phường Hạnh Thông', 'Phường Cầu Kiệu',
       'Phường Đức Nhuận', 'Phường Phú Nhuận'])

    loai_hinh = st.selectbox("Loại hình", ['Nhà ngõ, hẻm', 'Nhà phố liền kề', 'Nhà mặt phố, mặt tiền',
       'Nhà biệt thự'])

    giay_to_phap_ly = st.selectbox("Giấy tờ pháp lý", ['Đã có sổ', 'Chưa xác định', 'Đang chờ sổ',
       'Sổ chung / công chứng vi bằng', 'Giấy tờ viết tay', 'Không có sổ'])

    tinh_trang_noi_that = st.selectbox("Tình trạng nội thất", ['Nội thất đầy đủ', 'Không rõ', 'Hoàn thiện cơ bản',
       'Nội thất cao cấp', 'Bàn giao thô'])

    dac_diem = st.selectbox("Đặc điểm", ['Nhà nở hậu', 'Thông thường', 'Hẻm xe hơi', 'Hiện trạng khác',
       'Nhà nát', 'Nhà chưa hoàn công', 'Đất chưa chuyển thổ',
       'Nhà tóp hậu', 'Nhà dính quy hoạch / lộ giới'])
    #-------------------------------------------------------------------------
    st.markdown("---")

    input_data = {
        "mo_ta": mo_ta,
        "dien_tich_dat": dien_tich_dat,
        "so_phong_ngu": so_phong_ngu,
        "so_phong_ve_sinh": so_phong_ve_sinh,
        "tong_so_tang": tong_so_tang,
        "chieu_ngang": chieu_ngang,
        "gia_m2_tham_khao": gia_m2_tham_khao,
        "phuong": phuong,
        "loai_hinh": loai_hinh,
        "giay_to_phap_ly": giay_to_phap_ly,
        "tinh_trang_noi_that": tinh_trang_noi_that,
        "dac_diem": dac_diem
    }

    df = pd.DataFrame([input_data])
    scaler = joblib.load('Streamlit_project/models/scaler.pkl') # Chỗ này bạn sửa lại đường dẫn để chạy 
    #Scaling 
    num_cols = ['dien_tich_dat','so_phong_ngu','so_phong_ve_sinh','tong_so_tang', 'gia_m2_tham_khao', 'chieu_ngang']
    df[num_cols] = scaler.transform(df[num_cols])
    #Cat_encoding
    df = categorical_encoding(df)
    #-------------------------------------------------------------------------
    st.markdown("#### Anomoly Detector") 

    best_model = joblib.load('Streamlit_project/models/xgboost.pkl') # Chỗ này bạn sửa lại đường dẫn để chạy 
    y_pred = np.expm1(best_model.predict(df)[0])
    gia_ban_du_kien = st.number_input("Giá bán dự kiến của bạn (đơn vị: đồng)")

    missing_fields = [k for k, v in input_data.items() if v in [None, "", 0, 0.0]]

    if st.button("Check Anomaly"):
        if missing_fields:
            st.error(f"Bạn chưa nhập đầy đủ các trường: {', '.join(missing_fields)}")
            st.stop()  # ❗ dừng luôn, không chạy model

        anomaly_detector = joblib.load('Streamlit_project/models/anomaly_detector.pkl')
        df_input = df.iloc[[0]]

        result = anomaly_detector.predict_one(
            x_input=df_input,
            actual_price=gia_ban_du_kien,
            predicted_price=y_pred
        )

        is_anomaly = bool(result["Composite_Score"] >= anomaly_detector.threshold_)
        st.write("Predicted price:", result["Gia_du_doan_VND"])
        st.write("Actual price:", result["Gia_rao_ban_VND"])
        st.write("Residual score:", result["S_Resid"])
        st.write("Min-Max score:", result["S_MinMax"])
        st.write("Percentile score:", result["S_Percentile"])
        st.write("Isolation score:", result["S_ML"])
        st.write(f"Threshold (percentile = {anomaly_detector.default_percentile_}):", anomaly_detector.threshold_)
        st.write("Anomaly score:", result["Composite_Score"])
        st.write("Bất thường:", bool(is_anomaly))

    st.markdown("---")
    st.markdown("#### Price prediction") 

    if st.button("Predict Price"):
        if missing_fields:
            st.error(f"Bạn chưa nhập đầy đủ các trường: {', '.join(missing_fields)}")
            st.stop()  # ❗ dừng luôn, không chạy model

        st.write("*Predicted price:*", y_pred)

with tab3:
    st.subheader("Visualization")
    st.image("images/bang_thong_ke_processing.png", caption="Bảng thống kê tiền xử lý", use_container_width=True)
    st.image("images/phan_phoi_gia.png", caption="Phân phối giá", use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/top_10_features.png", caption="Top 10 Features", use_container_width=True)
    with col2:
        st.image("images/predicted_and_actual.png", caption="Predicted vs Actual Values", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
