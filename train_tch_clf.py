#train_tch_clf.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import json
import csv
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# ================= CONFIG =================
QUERIES_TSV = "data/dataset/synthetic_gold_queries.tsv"
REASONINGS_JSONL = "data/reasoning/train_1196_deepseek_clean.jsonl"
TRIALS_JSONL = "data/clinicaltrials/json_corpus/parsed/concatenated_trials.jsonl"
MODEL_NAME = "yikuan8/Clinical-Longformer"

MAX_LENGTH = 4096
BATCH_SIZE = 4
EPOCHS = 10
BASE_LR = 3e-5
CLS_LR = 6e-5
PATIENCE = 5
ALPHA_VALUES = [0.1, 0.2, 0.3]
#ALPHA_VALUES = [0.2]  # ✅ multiple alpha runs


# ================= Logging Utility =================
def log_message(msg, log_file):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ================= Data Loaders =================
def load_queries(tsv_file):
    queries = {}
    with open(tsv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) == 2:
                topic_id, query_text = row
                queries[topic_id] = query_text.strip()
    return queries


def load_reasonings(jsonl_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            label = 1 if str(obj.get("relevance", "")).lower() == "relevant" else 0
            data.append((obj["topic_id"], obj["trial_id"], obj.get("reasoning", ""), label))
    return data


def load_trials(jsonl_file):
    trials = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            trials[obj["id"]] = obj.get("concatenated_text", "")
    return trials


# ================= Dataset =================
class TrialDataset(Dataset):
    def __init__(self, data, queries, trials, tokenizer):
        self.data = data
        self.queries = queries
        self.trials = trials
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        topic_id, trial_id, reasoning, label = self.data[idx]
        query = self.queries[topic_id]
        trial_text = self.trials[trial_id]
        second_text = f"{trial_text} {self.tokenizer.sep_token} Reasoning: {reasoning}" if reasoning else trial_text

        tokenized = self.tokenizer(
            query,
            second_text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
            "topic_id": topic_id,
            "trial_id": trial_id
        }


# ================= Model =================
class TeacherReranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.longformer = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.longformer.config.hidden_size
        if hasattr(self.longformer, "pooler"):
            self.longformer.pooler = None
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits.squeeze(1)


# ================= Pairwise Ranking =================
def pairwise_ranking_loss_from_batch(logits, labels, topic_ids, device):
    logits_cpu = logits.detach().cpu()
    labels_cpu = labels.detach().cpu()
    groups = defaultdict(lambda: {"pos": [], "neg": []})
    for i, tid in enumerate(topic_ids):
        if labels_cpu[i].item() == 1:
            groups[tid]["pos"].append(logits_cpu[i].to(device))
        else:
            groups[tid]["neg"].append(logits_cpu[i].to(device))

    losses = []
    for tid, lists in groups.items():
        pos_list = lists["pos"]
        neg_list = lists["neg"]
        if len(pos_list) == 0 or len(neg_list) == 0:
            continue
        for p in pos_list:
            for n in neg_list:
                losses.append(-torch.log(torch.sigmoid(p - n) + 1e-12))
    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


# ================= Training Function =================
def train_teacher(alpha):
    save_dir = f"models_new/Teacher_ClinicalLongformer_1196/alpha{alpha}"
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"training_log_alpha{alpha}.txt")

    log_message(f"\n==================== TRAINING TEACHER (α={alpha}) ====================", log_file)
    log_message(f"Using GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}", log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    queries = load_queries(QUERIES_TSV)
    reasonings_data = load_reasonings(REASONINGS_JSONL)
    trials = load_trials(TRIALS_JSONL)
    train_data, val_data = train_test_split(reasonings_data, test_size=0.2, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = MAX_LENGTH

    train_loader = DataLoader(TrialDataset(train_data, queries, trials, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(TrialDataset(val_data, queries, trials, tokenizer),
                            batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    # ---- Model ----
    model = TeacherReranker().to(device)
    model.longformer.gradient_checkpointing_enable()
    log_message("Gradient checkpointing enabled.", log_file)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log_message(f"Using DataParallel with {torch.cuda.device_count()} GPUs.", log_file)

    # ---- Optimizer & Scheduler ----
    param_longformer = model.module.longformer.parameters() if isinstance(model, nn.DataParallel) else model.longformer.parameters()
    param_classifier = model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters()
    optimizer = torch.optim.AdamW([
        {"params": param_longformer, "lr": BASE_LR},
        {"params": param_classifier, "lr": CLS_LR}
    ])
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler(
        "cosine", optimizer=optimizer,
        num_warmup_steps=max(1, int(0.05 * num_training_steps)),
        num_training_steps=num_training_steps
    )

    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    best_val_loss = float("inf")
    patience_counter = 0

    # ---- Epoch loop ----
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss, total_bce, total_rank, batches_with_rank = 0, 0, 0, 0

        for batch in tqdm(train_loader, desc=f"[α={alpha}] Epoch {epoch+1} [Train]"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            topic_ids = batch["topic_id"]

            try:
                with autocast("cuda"):
                    logits = model(input_ids, attention_mask)
                    bce_loss = criterion(logits, labels)
                    rank_loss = pairwise_ranking_loss_from_batch(logits, labels, topic_ids, device)
                    loss = bce_loss + alpha * rank_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                total_train_loss += loss.item()
                total_bce += bce_loss.item()
                total_rank += rank_loss.item()
                batches_with_rank += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise

        avg_train_loss = total_train_loss / len(train_loader)
        avg_bce = total_bce / len(train_loader)
        avg_rank = total_rank / max(1, batches_with_rank)
        log_message(f"[α={alpha}] Epoch {epoch+1} Train: Loss={avg_train_loss:.4f} | BCE={avg_bce:.4f} | Rank={avg_rank:.6f}", log_file)

        # ---- Validation ----
        model.eval()
        total_val_loss, total_val_rank, val_batches_with_rank = 0, 0, 0
        all_logits, all_labels = [], []

        with torch.no_grad(), autocast("cuda"):
            for batch in tqdm(val_loader, desc=f"[α={alpha}] Epoch {epoch+1} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                topic_ids = batch["topic_id"]

                logits = model(input_ids, attention_mask)
                bce_loss = criterion(logits, labels)
                rank_loss = pairwise_ranking_loss_from_batch(logits, labels, topic_ids, device)
                loss = bce_loss + alpha * rank_loss

                total_val_loss += loss.item()
                total_val_rank += rank_loss.item()
                val_batches_with_rank += 1
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_rank = total_val_rank / max(1, val_batches_with_rank)
        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()

        try:
            val_auc = roc_auc_score(all_labels, all_logits) if len(set(all_labels.tolist())) == 2 else float("nan")
        except Exception:
            val_auc = float("nan")

        log_message(f"[α={alpha}] Epoch {epoch+1} Val: Loss={avg_val_loss:.4f} | Rank={avg_val_rank:.6f} | AUC={val_auc:.4f}", log_file)

        # ---- Checkpoint ----
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(to_save, os.path.join(save_dir, f"best_teacher_alpha{alpha}.pt"))
            log_message(f"✅ Saved new best teacher model (α={alpha})", log_file)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log_message(f"⏹️ Early stopping triggered for α={alpha}", log_file)
                break

    log_message(f"Training complete for α={alpha}. Best Val Loss: {best_val_loss:.4f}", log_file)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    for alpha in ALPHA_VALUES:
        train_teacher(alpha)
        torch.cuda.empty_cache()
