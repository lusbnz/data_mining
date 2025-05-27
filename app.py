import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from datasets import Dataset
import torch
import os
import json
import pandas as pd

st.set_page_config(page_title="Phân Tích Cảm Xúc", layout="wide")

MODEL_DIR = "models"
HISTORY_PATH = "history.json"

@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def preprocess_data(df, tokenizer):
    df['text'] = df['text'].astype(str)
    dataset = Dataset.from_pandas(df)
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=258)
    return dataset.map(preprocess, batched=True)

def get_next_model_dir(base_dir=MODEL_DIR):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return os.path.join(base_dir, "model_1")
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("model_")]
    indices = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_index = max(indices) + 1 if indices else 1
    return os.path.join(base_dir, f"model_{next_index}")

# Callback để log quá trình training
class StreamlitCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.logs.append(logs)
            if 'loss' in logs:
                st.session_state['training_logs'].append(f"Epoch {int(state.epoch)} - Loss: {logs['loss']}")

def train_and_save_model(uploaded_file, num_train_epochs=2):
    if uploaded_file is None:
        st.warning("Vui lòng upload file dữ liệu")
        return False

    try:
        # Đọc file CSV hoặc JSON
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Chỉ hỗ trợ định dạng CSV hoặc JSON")
            return False
        
        if 'text' not in df.columns or 'label' not in df.columns:
            st.error("File phải có cột 'text' và 'label'")
            return False
        
        # Hiển thị bảng dữ liệu vừa upload
        st.subheader("📊 Dữ liệu huấn luyện:")
        st.dataframe(df)

        output_dir = get_next_model_dir()
        os.makedirs(output_dir, exist_ok=True)

        pretrained_model_name = "roberta-base"

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=3)
        tokenized = preprocess_data(df, tokenizer)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=8,
            save_steps=500,
            save_total_limit=1,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            remove_unused_columns=False,
        )

        # Khởi tạo session_state để lưu log training
        if 'training_logs' not in st.session_state:
            st.session_state['training_logs'] = []

        trainer = Trainer(
            model=model, 
            args=args, 
            train_dataset=tokenized, 
            callbacks=[StreamlitCallback()]
        )

        with st.spinner("Đang huấn luyện model..."):
            trainer.train()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return True
    except Exception as e:
        st.error(f"Lỗi khi xử lý file: {e}")
        return False

def save_history(text, label):
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    history.append({"text": text, "label": label})
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Sidebar
st.sidebar.header("Chức năng")

if os.path.exists(MODEL_DIR):
    model_list = [os.path.join(MODEL_DIR, d) for d in sorted(os.listdir(MODEL_DIR)) if os.path.isdir(os.path.join(MODEL_DIR, d))]
else:
    model_list = []

if not model_list:
    st.sidebar.warning("Chưa có model nào trong thư mục models. Vui lòng train model mới.")
else:
    selected_model = st.sidebar.selectbox("Chọn model để dự đoán", model_list)

uploaded_file = st.sidebar.file_uploader("Upload CSV hoặc JSON chứa cột 'text' và 'label' để huấn luyện", type=['csv', 'json'])

if st.sidebar.button("Huấn luyện model"):
    success = train_and_save_model(uploaded_file)
    if success:
        st.success("✅ Huấn luyện thành công! Model đã lưu và sẵn sàng dự đoán.")
        # Reload model list sau khi train xong
        model_list = [os.path.join(MODEL_DIR, d) for d in sorted(os.listdir(MODEL_DIR)) if os.path.isdir(os.path.join(MODEL_DIR, d))]
        selected_model = model_list[-1]
        # Hiển thị log training
        st.subheader("📈 Log quá trình huấn luyện:")
        for log in st.session_state['training_logs']:
            st.write(log)
    else:
        st.error("❌ Huấn luyện thất bại!")

if model_list:
    tokenizer, model = load_model(selected_model)
else:
    st.warning("Không có model để load. Vui lòng train model trước.")
    st.stop()

# Phần dự đoán
st.title("🔍 Phân Tích Cảm Xúc Văn Bản")

input_text = st.text_area("Nhập văn bản để dự đoán cảm xúc:", height=150)

if st.button("Dự đoán"):
    if input_text.strip():
        with st.spinner("Đang dự đoán..."):
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
            label_map = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
            predicted_label = label_map.get(predicted_class, "Không xác định")
            save_history(input_text, predicted_label)
            st.success(f"Kết quả dự đoán: **{predicted_label}**")
        history = load_history()
        st.subheader("🕘 Lịch sử dự đoán (tối đa 10):")
        for i, record in enumerate(reversed(history[-10:]), 1):
            st.write(f"{i}. [{record['label']}] {record['text']}")
    else:
        st.warning("Vui lòng nhập văn bản để dự đoán.")
