import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import json
from datetime import datetime

# Thiết lập cấu hình trang
st.set_page_config(page_title="Text Sentiment Analysis", layout="wide")

# Tạo thư mục để lưu model
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Hàm lưu model
def save_model(vectorizer, model, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"{model_name}_{timestamp}.joblib")
    joblib.dump({'vectorizer': vectorizer, 'model': model}, model_path)
    return model_path

# Hàm tải danh sách model
def get_model_list():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]

# Hàm dự đoán
def predict_sentiment(text, model_path):
    try:
        # Tải model và vectorizer
        loaded = joblib.load(os.path.join(MODEL_DIR, model_path))
        vectorizer = loaded['vectorizer']
        model = loaded['model']
        
        # Chuyển đổi text thành vector
        text_vector = vectorizer.transform([text])
        
        # Dự đoán
        prediction = model.predict(text_vector)[0]
        return prediction
    except Exception as e:
        return f"Lỗi: {str(e)}"

# Hàm huấn luyện model
def train_model(data, text_column, label_column, progress_bar):
    try:
        # Cập nhật progress
        progress_bar.progress(10)
        
        # Chuẩn bị dữ liệu
        X = data[text_column]
        y = data[label_column]
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        progress_bar.progress(30)
        
        # Vector hóa văn bản
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        progress_bar.progress(50)
        
        # Huấn luyện model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vec, y_train)
        progress_bar.progress(80)
        
        # Đánh giá
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Lưu model
        model_name = "sentiment_model"
        model_path = save_model(vectorizer, model, model_name)
        progress_bar.progress(100)
        
        return True, f"Huấn luyện thành công! Accuracy: {accuracy:.2f}\nModel được lưu tại: {model_path}"
    except Exception as e:
        return False, f"Lỗi khi huấn luyện: {str(e)}"

# Giao diện chính
st.title("Phân Tích Cảm Xúc Văn Bản")

# Tabs
tab1, tab2 = st.tabs(["Dự Đoán", "Huấn Luyện"])

# Tab Dự Đoán
with tab1:
    st.header("Dự Đoán Cảm Xúc")
    
    # Select box chọn model
    model_list = get_model_list()
    if not model_list:
        st.warning("Chưa có model nào được huấn luyện!")
    else:
        selected_model = st.selectbox("Chọn model", model_list)
        
        # Input text
        input_text = st.text_area("Nhập văn bản cần dự đoán:", height=100)
        
        # Nút dự đoán
        if st.button("Dự Đoán"):
            if input_text.strip():
                with st.spinner("Đang dự đoán..."):
                    result = predict_sentiment(input_text, selected_model)
                    if result in ['tiêu cực', 'trung tính', 'tích cực']:
                        st.success(f"Kết quả: **{result}**")
                    else:
                        st.error(result)
            else:
                st.error("Vui lòng nhập văn bản!")

# Tab Huấn Luyện
with tab2:
    st.header("Huấn Luyện Model")
    
    # Upload file
    uploaded_file = st.file_uploader("Tải lên file dữ liệu (CSV hoặc JSON)", type=['csv', 'json'])
    
    if uploaded_file:
        # Đọc file
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_json(uploaded_file)
            
            # Hiển thị preview
            st.write("Preview dữ liệu:")
            st.dataframe(data.head())
            
            # Chọn cột
            columns = data.columns.tolist()
            text_column = st.selectbox("Chọn cột chứa văn bản", columns)
            label_column = st.selectbox("Chọn cột chứa nhãn (tiêu cực/trung tính/tích cực)", columns)
            
            # Nút huấn luyện
            if st.button("Bắt Đầu Huấn Luyện"):
                if text_column and label_column:
                    progress_bar = st.progress(0)
                    success, message = train_model(data, text_column, label_column, progress_bar)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.error("Vui lòng chọn cột văn bản và nhãn!")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {str(e)}")

# Hướng dẫn sử dụng
st.sidebar.header("Hướng Dẫn")
st.sidebar.markdown("""
1. **Dự Đoán**:
   - Chọn model từ danh sách
   - Nhập văn bản cần dự đoán
   - Nhấn "Dự Đoán" để xem kết quả
   
2. **Huấn Luyện**:
   - Tải lên file CSV/JSON (cần có cột văn bản và nhãn)
   - Chọn cột chứa văn bản và nhãn
   - Nhấn "Bắt Đầu Huấn Luyện"
   - Theo dõi thanh tiến độ

**Lưu ý**: Nhãn phải là "tiêu cực", "trung tính", hoặc# Basic Salad Recipe hoặc "tích cực"
""")