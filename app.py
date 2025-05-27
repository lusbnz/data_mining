import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score
import torch
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Phân Tích Cảm Xúc", layout="wide")

MODEL_DIR = "models"
HISTORY_PATH = "history.json"
CHECKPOINT_PATH = "checkpoint.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model.to(DEVICE)

def preprocess_data(df, tokenizer):
    df['text'] = df['text'].astype(str)
    dataset = Dataset.from_pandas(df)
    def preprocess(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)
    tokenized = dataset.map(preprocess, batched=True)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized

def get_dataloader(tokenized_dataset, batch_size=8):
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

def get_next_model_dir(base_dir=MODEL_DIR):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("model_")]
    indices = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_index = max(indices) + 1 if indices else 1
    return os.path.join(base_dir, f"model_{next_index}")

def flat_accuracy(preds, labels):
    preds = np.argmax(preds, axis=1)
    acc = np.mean(preds == labels)
    f1 = f1_score(labels, preds, average='macro')
    return acc, f1

def save_history(text, label):
    history = load_history()
    history.append({"text": text, "label": label})
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def train_custom_model(uploaded_file, num_train_epochs=3):
    if uploaded_file is None:
        st.warning("Vui lòng upload file dữ liệu")
        return False

    try:
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

        pretrained_model_name = "vinai/phobert-base"
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=3)
        model.to(DEVICE)

        tokenized = preprocess_data(df, tokenizer)
        dataloader = get_dataloader(tokenized)

        optimizer = AdamW(model.parameters(), lr=3e-5)
        total_steps = len(dataloader) * num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

        start_epoch = 0
        if os.path.exists(CHECKPOINT_PATH):
            try:
                checkpoint = torch.load(CHECKPOINT_PATH)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                st.info(f"Đã load checkpoint từ epoch {start_epoch}")
            except RuntimeError as e:
                st.warning(f"Không thể load checkpoint do lỗi: {e}. Bỏ qua checkpoint và huấn luyện lại từ đầu.")

        log_placeholder = st.empty()
        progress_bar = st.progress(0)

        for epoch in range(start_epoch, num_train_epochs):
            model.train()
            total_loss, all_preds, all_labels = 0, [], []
            for batch in dataloader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                all_preds.extend(logits.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            acc, f1 = flat_accuracy(np.array(all_preds), np.array(all_labels))
            log = f"Epoch {epoch+1}/{num_train_epochs} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}"
            log_placeholder.text(log)
            progress_bar.progress((epoch+1)/num_train_epochs)
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, CHECKPOINT_PATH)

        # Save final model
        output_dir = get_next_model_dir()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        return True

    except Exception as e:
        st.error(f"Lỗi khi xử lý file: {e}")
        return False

# --- Streamlit UI ---
st.sidebar.header("Chức năng")
if os.path.exists(MODEL_DIR):
    model_list = [os.path.join(MODEL_DIR, d) for d in sorted(os.listdir(MODEL_DIR)) if os.path.isdir(os.path.join(MODEL_DIR, d))]
else:
    model_list = []

if not model_list:
    st.sidebar.warning("Chưa có model nào. Vui lòng huấn luyện trước.")
else:
    selected_model = st.sidebar.selectbox("Chọn model để dự đoán", model_list)

uploaded_file = st.sidebar.file_uploader("Upload dữ liệu huấn luyện (CSV/JSON)", type=['csv', 'json'])

if st.sidebar.button("Huấn luyện model"):
    with st.spinner("Đang huấn luyện..."):
        success = train_custom_model(uploaded_file)
        if success:
            st.success("Huấn luyện thành công!")
            model_list = [os.path.join(MODEL_DIR, d) for d in sorted(os.listdir(MODEL_DIR)) if os.path.isdir(os.path.join(MODEL_DIR, d))]
            selected_model = model_list[-1]
        else:
            st.error("Huấn luyện thất bại!")

if model_list:
    tokenizer, model = load_model(selected_model)
else:
    st.warning("Không có model nào. Vui lòng huấn luyện.")
    st.stop()

st.title("Phân Tích Cảm Xúc Văn Bản")
input_text = st.text_area("Nhập văn bản để dự đoán:", height=150)

if st.button("Dự đoán"):
    if input_text.strip():
        with st.spinner("Đang dự đoán..."):
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
            label_map = {0: "tiêu cực", 1: "trung tính", 2: "tích cực"}
            st.success(f"Kết quả dự đoán: **{label_map.get(pred, 'Không xác định')}**")
            save_history(input_text, label_map.get(pred, 'Không xác định'))

        st.subheader("Lịch sử dự đoán (10 gần nhất):")
        history = load_history()
        for i, record in enumerate(reversed(history[-10:]), 1):
            st.write(f"{i}. [{record['label']}] {record['text']}")
    else:
        st.warning("Vui lòng nhập văn bản.")
