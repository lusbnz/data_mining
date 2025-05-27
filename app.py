import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os
import json

st.set_page_config(page_title="Phân Tích Cảm Xúc", layout="wide")

MODEL_DIR = "models"
PRETRAINED_MODEL = "vinai/phobert-base"
HISTORY_PATH = "history.json"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model

def train_and_save_model(output_dir=MODEL_DIR, model_dir=MODEL_DIR, num_train_epochs=2):
    os.makedirs(output_dir, exist_ok=True)

    # Nếu đã có model cũ, load để tiếp tục huấn luyện
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print("Load model PhoBERT đã train trước đó để tiếp tục fine-tune...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        print("Chưa có model trước đó, load PhoBERT gốc từ Huggingface...")
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=3)

    # Load dataset train (bạn có thể thay bằng dữ liệu riêng)
    dataset = load_dataset("yelp_polarity", split="train[:5000]")  # Demo: chỉ lấy 5000 mẫu

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=258)

    tokenized = dataset.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Training arguments
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

    trainer = Trainer(model=model, args=args, train_dataset=tokenized)
    trainer.train()

    # Save lại model đã fine-tune tiếp
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model đã fine-tune thêm và lưu vào {output_dir}")

def save_history(text, label):
    # Đọc file history cũ
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    # Thêm record mới
    history.append({"text": text, "label": label})

    # Ghi lại file
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Sidebar tùy chọn
st.sidebar.header("Tùy chọn")
if st.sidebar.button("Huấn luyện thêm PhoBERT"):
    with st.spinner("Đang huấn luyện tiếp từ PhoBERT..."):
        train_and_save_model()
        st.success("Huấn luyện thêm hoàn tất! Model đã lưu và sẵn sàng dự đoán.")
        st.experimental_rerun()

# Load model
tokenizer, model = load_model()

# Giao diện chính
st.title("Phân Tích Cảm Xúc Văn Bản (PhoBERT)")

input_text = st.text_area("Nhập văn bản để dự đoán cảm xúc:", height=150)

if st.button("Dự đoán"):
    if input_text.strip():
        with st.spinner("Đang dự đoán..."):
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
            label_map = {0: "tiêu cực", 1: "trung tính", 2: "tích cực"}
            predicted_label = label_map.get(predicted_class, "Không xác định")
            
            # Lưu lịch sử dự đoán
            save_history(input_text, predicted_label)

            st.success(f"Kết quả dự đoán: **{predicted_label}**")

        # Hiển thị lịch sử 10 dự đoán gần nhất
        history = load_history()
        st.subheader("Lịch sử dự đoán:")
        for i, record in enumerate(reversed(history[-10:]), 1):
            st.write(f"{i}. [{record['label']}] {record['text']}")
    else:
        st.warning("Vui lòng nhập văn bản trước khi dự đoán.")
