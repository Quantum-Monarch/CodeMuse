import os
import sqlite3
import pickle
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
import joblib

def init_db():
    path = "user_edits.db"
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_edits (
        id INTEGER PRIMARY KEY,
        original_comment TEXT NOT NULL,
        user_edit TEXT NOT NULL,
        code_embedding BLOB
    )
    """)
    conn.commit()
    return path

def save_edit(db, original_comment, user_edit, code_embedding=None):
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO user_edits (original_comment, user_edit, code_embedding)
        VALUES (?, ?, ?)
        """, (
            original_comment,
            user_edit,
            pickle.dumps(code_embedding) if code_embedding else None
            ))

def load_appropriate_model():
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB

        if gpu_memory >= 16:  # High-end GPU
            modelname="Salesforce/codet5-large"

        elif gpu_memory >= 8:  # Mid-range GPU
            modelname = "Salesforce/codet5-base"

        else:  # Low-end GPU
            modelname = "Salesforce/codet5-small"

    else:  # CPU only
        modelname ="Salesforce/codet5-small"


    return modelname






def assing_context_tags():
    # ===== CONFIGURATION =====
    DB_PATH = "user_edits.db"      # path to your SQLite DB
    N_CLUSTERS = 10                # number of clusters / colors
    COLOR_PALETTE = [
        "Red", "Blue", "Green", "Yellow", "Purple",
        "Orange", "Cyan", "Magenta", "Brown", "Teal"
    ]  # Must have at least N_CLUSTERS entries

    # ===== 1. Connect to the DB =====
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ===== 2. Ensure the color_tag column exists =====
    cursor.execute("PRAGMA table_info(user_edits)")
    columns = [col[1] for col in cursor.fetchall()]
    if "color_tag" not in columns:
        cursor.execute("ALTER TABLE user_edits ADD COLUMN color_tag TEXT")
        conn.commit()

    # ===== 3. Load embeddings from the DB =====
    cursor.execute("SELECT id, code_embedding FROM user_edits WHERE code_embedding IS NOT NULL")
    rows = cursor.fetchall()

    ids = []
    embeddings = []

    for row in rows:
        row_id, emb_blob = row
        emb_vector = pickle.loads(emb_blob)
        ids.append(row_id)
        embeddings.append(emb_vector)

    if not embeddings:
        print("No embeddings found in DB. Exiting.")
        conn.close()
        exit()

    embeddings = np.array(embeddings)

    # ===== 4. Run KMeans clustering =====
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    joblib.dump(cluster_labels,"kmeans_model.pkl")

    # ===== 5. Assign color tags based on cluster =====
    cluster_to_color = {i: COLOR_PALETTE[i] for i in range(N_CLUSTERS)}
    color_tags = [cluster_to_color[label] for label in cluster_labels]

    # ===== 6. Update DB with color tags =====
    for row_id, color in zip(ids, color_tags):
        cursor.execute(
            "UPDATE user_edits SET color_tag = ? WHERE id = ?",
            (color, row_id)
        )

    conn.commit()
    conn.close()


def fetch_training_data(db="user_edits.db"):
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT original_comment,user_edit,color_tag FROM user_edits""")
        data=cursor.fetchall()
    return data

separator_token="<|sep|>"
stylemodelname=load_appropriate_model()
stylemodel = AutoModelForSeq2SeqLM.from_pretrained(stylemodelname)
tokenizer = AutoTokenizer.from_pretrained(stylemodelname)
tokenizer.add_special_tokens({'additional_special_tokens': [separator_token]})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stylemodel.to(device)

class styletransfermodel(Dataset):
    def __init__(self, pairs, tokenizer=tokenizer, max_len=128):
        self.examples = []

        for color_tag, orig_comment, user_edit in pairs:
            text = f" {color_tag} {separator_token} {orig_comment} {tokenizer.eos_token} {user_edit} {tokenizer.eos_token}"
            encodings_input = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            text2=f"{color_tag} {separator_token} {orig_comment} {tokenizer.eos_token}"
            enc_prompt = tokenizer(
                text2,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            input__ids=encodings_input["input_ids"].squeeze()
            labelz = input__ids.clone()
            labelz[: enc_prompt["input_ids"].ne(tokenizer.pad_token_id).sum()] = -100
            labelz[input__ids == tokenizer.pad_token_id] = -100

            self.examples.append({
                "input_ids": encodings_input["input_ids"].squeeze(),
                "attention_mask": encodings_input["attention_mask"].squeeze(),
                "labels": labelz
            })


def trainstylemodel(stylemodel2, save_dir="style_model_checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    data=fetch_training_data("user_edits.db")
    dataset = styletransfermodel(data, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = AdamW(stylemodel2.parameters(), lr=5e-4)
    stylemodel2.to(device)
    stylemodel2.train()
    epochs = 5

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        for batch in tqdm(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = stylemodel(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stylemodel2.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss+=loss.item()

            print(f"Loss: {loss.item():.4f}")

        checkpoint_path = os.path.join(save_dir, f"style_model_epoch{epoch + 1}.pt")
        torch.save(stylemodel2.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    final_model_path = os.path.join(save_dir, "style_model_final.pt")
    torch.save(stylemodel2.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    stylemodel2.eval()

def check_num_entries(db="user_edits.db"):
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT COUNT(*) FROM user_edits""")
        count=cursor.fetchone()[0]
    flag=False
    if count%500==0:
        flag=True
    return flag

def check_num(db="user_edits.db"):
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT COUNT(*) FROM user_edits""")
        count=cursor.fetchone()[0]
    style=False
    if count>500:
        style=True
    return style

def load_latest_checkpoint(model=stylemodel, save_dir="style_model_checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "style_model_final.pt")

    if os.path.exists(final_path):
        print(f"Loading checkpoint from {final_path}")
        model.load_state_dict(torch.load(final_path, map_location=device))

    else:
        tokenizer = AutoTokenizer.from_pretrained(stylemodelname)
        tokenizer.add_special_tokens({'additional_special_tokens': [separator_token]})
        print("No checkpoint found. Starting fresh from base model.")

    return model

def assign_color_tags(embeddings):
    kmeans_model= joblib.load("kmeans_model.pkl")
    COLOR_PALETTE = {
        0:"Red", 1:"Blue", 2:"Green", 3:"Yellow", 4:"Purple",
        5:"Orange", 6:"Cyan", 7:"Magenta", 8:"Brown", 9:"Teal"
    }
    embedding_2d = np.array(embeddings).reshape(1, -1)
    cluster_idx = kmeans_model.predict(embedding_2d)[0]

    return COLOR_PALETTE[cluster_idx]

def generate_user_style_comment(orig_comment,color_tag, max_len=128):
    """
    Generates a user-style comment given an original comment and color tag.
    """
    # Construct input sequence (matching training formatting)
    input_text = f"{color_tag} {separator_token} {orig_comment} {tokenizer.eos_token}"
    model=load_latest_checkpoint()
    # Tokenize
    inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt"
    ).to(device)
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            num_beams=5,  # can tweak for creativity vs precision
            early_stopping=True
        )
    # Decode
    generated_comment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_comment

def main():
    flag=check_num_entries()
    if flag:
        assing_context_tags()
        model=load_latest_checkpoint(stylemodel)
        trainstylemodel(model)
        return
    else:
        return