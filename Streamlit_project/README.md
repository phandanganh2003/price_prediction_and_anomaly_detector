🏠 Dự đoán giá bất động sản & phát hiện bất thường

📌 Tổng quan

Đây là một ứng dụng Machine Learning được xây dựng nhằm:

🔹 Dự đoán giá bất động sản dựa trên các đặc trưng của căn nhà
🔹 Phát hiện các bất động sản có giá bất thường (anomaly)
🔹 Cung cấp giao diện tương tác cho người dùng thông qua Streamlit

Dữ liệu sử dụng trong project là dataset đã được cung cấp sẵn. 

📌 Bài toán kinh doanh

Trong thị trường bất động sản, giá đăng bán có thể không phản ánh đúng giá trị thực. Do đó, xuất hiện các trường hợp:
🔹 Giá quá cao → khó bán
🔹 Giá quá thấp → có thể là lỗi hoặc gian lận

📌 Mục tiêu của project:

🔹  Ước lượng giá hợp lý (reference price)
🔹  Xác định bất động sản có bất thường hay không
🔹  Hỗ trợ: Người mua, người bán, nền tảng đăng tin 

📌 Phương pháp tiếp cận

1. Dự đoán giá

Mô hình sử dụng:
- XGBoost Regressor

Target:
- log1p(price) để xử lý skew dữ liệu

Sau khi dự đoán:
- Dùng expm1() để đưa về giá trị thực

2. Phát hiện bất thường (Anomaly Detection)

Sử dụng Composite Score, kết hợp:

- Residual-z 
- Vi phạm Min/Max
- Ngoài khoảng tin cậy P10-P90
- Isolation Forest

Quy trình:
    Tính prediction error
    Chuẩn hóa thành anomaly score
    So sánh với percentile threshold từ model được train 

3. Luồng xử lý

        User input
        ↓
        Preprocessing (encode + align features)
        ↓
        Model prediction
        ↓
        Compute anomaly score
        ↓
        Compare with threshold
        ↓
        Output result

🗂️ Cấu trúc project
Streamlit_project/
│
├── content_based_app.py       # File chính chạy Streamlit
│
├── data/                      # Chứa dữ liệu 
│
├── models/
│   ├── xgboost.pkl            # Model đã train
│   ├── scaler.pkl             # Scaler (nếu có)
│   └── trained_columns.pkl    # Danh sách cột sau encoding
│
├── src/
│   ├── anomaly_detector.py    # Logic phát hiện bất thường
│   ├── dump_xgboost.py/       # Train Xgboost
│   └── train_anomaly.py       # Train anomaly model 
│
├── utils.py/                  # File hàm hỗ trợ 
│
├── images/                    # Folder ảnh 
├── requirements.txt
└── README.md

⚙️ Công nghệ sử dụng

Python
Pandas / NumPy
Scikit-learn
XGBoost
Streamlit
Joblib
...

💻 Cách chạy project
1. Cài thư viện
    pip install -r requirements.txt
2. Chạy ứng dụng
    streamlit run app.py

🖥️ Giao diện người dùng

Người dùng có thể:

Nhập thông tin bất động sản:
    Diện tích
    Số phòng
    Vị trí
    ...

Xem kết quả:
- Giá dự đoán
- Mức độ bất thường

👤 Tác giả
Phan Đặng Anh, Tang Đông Hy, Phó Quốc Dũng