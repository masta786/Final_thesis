"""Train the fixed BiLSTM retriever and measure it against the original.

The original retriever failed for five compounding reasons; each is fixed here:
  1. mean pooling included PAD positions   -> masked mean pooling
  2. OOV words were mapped onto PAD id 0   -> separate UNK id (0=PAD, 1=UNK)
  3. 3k vocab dropped most medical terms   -> 20k vocab
  4. answers truncated to 50 tokens        -> 96 tokens
  5. BCE against 1 random negative (trivial) -> InfoNCE with in-batch negatives

Sized for a 2-core CPU: 1 BiLSTM layer, hidden 192.
Reports Recall@k / MRR, which is the right metric for retrieval - the notebook's
ROUGE-L-against-gold proxy cannot tell "relevant" from "fluent".
"""
import json, random, re, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

ROOT = Path(__file__).parent
MODELS = ROOT / "saved_models"; MODELS.mkdir(exist_ok=True)
torch.set_num_threads(2)
random.seed(42); np.random.seed(42); torch.manual_seed(42)

VOCAB_SIZE, MAX_Q, MAX_A = 20000, 24, 96
EMB, HID, LAYERS, DROP = 128, 192, 1, 0.2
EPOCHS, BS, TEMP, LR = 8, 64, 0.05, 1e-3


def clean_text(t):
    t = re.sub(r"[^a-z0-9 ]", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()


print("loading data...", flush=True)
df = pd.read_csv(ROOT / "data/medquad.csv")
df.columns = [c.lower().strip() for c in df.columns]
df = df[["question", "answer"]].dropna()
df = df[df["answer"].str.len() >= 60].reset_index(drop=True)
df = df.sample(n=3000, random_state=42).reset_index(drop=True)
df["q_clean"] = df["question"].apply(clean_text)
df["a_clean"] = df["answer"].apply(clean_text)

n = len(df); idx = list(range(n)); random.shuffle(idx)
train_idx, val_idx, test_idx = idx[:int(.70*n)], idx[int(.70*n):int(.80*n)], idx[int(.80*n):]
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)
print(f"train {len(train_df)}  val {len(val_df)}  test {len(test_df)}", flush=True)

counts = Counter(" ".join(df["q_clean"].tolist() + df["a_clean"].tolist()).split())
vocab_words = [w for w, _ in counts.most_common(VOCAB_SIZE)]
word2idx = {w: i + 2 for i, w in enumerate(vocab_words)}
old_vocab = set(w for w, _ in counts.most_common(3000))
probes = ["metformin", "lisinopril", "hypertension", "hypoglycemia", "insulin"]
print("medical terms present in old 3k vocab: "
      + ", ".join(f"{p}={'yes' if p in old_vocab else 'NO'}" for p in probes), flush=True)


def encode(txt, max_len):
    toks = clean_text(txt).split()[:max_len]
    ids = [word2idx.get(w, 1) for w in toks]
    return ids + [0] * (max_len - len(ids))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE + 2, EMB, padding_idx=0)
        self.lstm = nn.LSTM(EMB, HID, num_layers=LAYERS, batch_first=True,
                            bidirectional=True,
                            dropout=DROP if LAYERS > 1 else 0.0)
        self.drop = nn.Dropout(DROP)

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        out, _ = self.lstm(self.drop(self.embed(x)))
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return F.normalize(pooled, dim=-1)


def batches(pairs, bs, shuffle=True):
    order = list(range(len(pairs)))
    if shuffle:
        random.shuffle(order)
    for s in range(0, len(order), bs):
        chunk = [pairs[i] for i in order[s:s + bs]]
        yield (torch.tensor([encode(q, MAX_Q) for q, _ in chunk], dtype=torch.long),
               torch.tensor([encode(a, MAX_A) for _, a in chunk], dtype=torch.long))


def train():
    enc = Encoder()
    opt = torch.optim.Adam(enc.parameters(), lr=LR)
    tr = list(zip(train_df["q_clean"], train_df["a_clean"]))
    va = list(zip(val_df["q_clean"], val_df["a_clean"]))
    best, best_state = 1e9, None
    for ep in range(EPOCHS):
        t0 = time.time(); enc.train(); tot = nb = 0
        for q, a in batches(tr, BS):
            opt.zero_grad()
            qe, ae = enc(q), enc(a)
            logits = qe @ ae.T / TEMP
            lab = torch.arange(len(qe))
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step(); tot += loss.item(); nb += 1
        enc.eval(); vt = vn = 0
        with torch.no_grad():
            for q, a in batches(va, BS, shuffle=False):
                qe, ae = enc(q), enc(a)
                lab = torch.arange(len(qe))
                vt += F.cross_entropy(qe @ ae.T / TEMP, lab).item(); vn += 1
        v = vt / max(vn, 1)
        print(f"  epoch {ep+1}/{EPOCHS}  train {tot/nb:.4f}  val {v:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in enc.state_dict().items()}
    enc.load_state_dict(best_state)
    return enc


def embed(enc, texts, max_len, bs=64):
    enc.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(texts), bs):
            x = torch.tensor([encode(t, max_len) for t in texts[s:s+bs]], dtype=torch.long)
            out.append(enc(x).numpy())
    return np.vstack(out)


def ir_metrics(sim, gold, ks=(1, 5, 10)):
    order = np.argsort(-sim, axis=1)
    ranks = np.array([np.where(order[i] == g)[0][0] + 1 for i, g in enumerate(gold)])
    r = {f"R@{k}": float((ranks <= k).mean()) for k in ks}
    r["MRR"] = float((1.0 / ranks).mean())
    return r


# ---- retrieval is measured over the WHOLE corpus so each test question's own
# ---- answer is in the index and Recall@k is well defined
kb_all_clean = df["a_clean"].tolist()
test_qs = test_df["q_clean"].tolist()
gold_rows = np.array(test_idx)
results = {}

print("\nTF-IDF baseline...", flush=True)
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
kb_tfidf = tfidf.fit_transform(kb_all_clean)
sim_tfidf = sk_cosine(tfidf.transform(test_qs), kb_tfidf)
results["TF-IDF"] = ir_metrics(sim_tfidf, gold_rows)
print("   ", results["TF-IDF"], flush=True)

print("original BiLSTM (as shipped, with the bugs)...", flush=True)
try:
    old_w2i = {w: i + 1 for i, w in enumerate([w for w, _ in counts.most_common(3000)])}

    def old_encode(t, max_len=50):
        toks = clean_text(t).split()[:max_len]
        return [old_w2i.get(w, 0) for w in toks] + [0] * (max_len - len(toks[:max_len]))

    class OldEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(3001, 128, padding_idx=0)
            self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True,
                                bidirectional=True, dropout=0.3)
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            out, _ = self.lstm(self.drop(self.embed(x)))
            return F.normalize(out.mean(dim=1), dim=-1)   # unmasked: the bug

    sd = torch.load(MODELS / "bilstm.pt", map_location="cpu")
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()}
    old = OldEnc(); old.load_state_dict(sd); old.eval()

    def old_embed(texts, max_len=50, bs=64):
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), bs):
                x = torch.tensor([old_encode(t, max_len) for t in texts[s:s+bs]],
                                 dtype=torch.long)
                out.append(old(x).numpy())
        return np.vstack(out)

    sim_old = sk_cosine(old_embed(test_qs), old_embed(kb_all_clean))
    results["BiLSTM original"] = ir_metrics(sim_old, gold_rows)
    print("   ", results["BiLSTM original"], flush=True)
except Exception as e:
    print("    could not evaluate original:", str(e)[:120], flush=True)

print("\ntraining fixed BiLSTM...", flush=True)
enc = train()

print("embedding corpus...", flush=True)
kb_emb_all = embed(enc, kb_all_clean, MAX_A)
q_emb = embed(enc, test_qs, MAX_Q)
sim_new = sk_cosine(q_emb, kb_emb_all)
results["BiLSTM fixed"] = ir_metrics(sim_new, gold_rows)


def rerank(dense, sparse, top=50, w=0.5):
    out = np.full_like(dense, -1e9)
    for i in range(dense.shape[0]):
        c = np.argsort(-dense[i])[:top]
        out[i, c] = w * dense[i, c] + (1 - w) * sparse[i, c]
    return out


results["BiLSTM fixed + TF-IDF rerank"] = ir_metrics(rerank(sim_new, sim_tfidf), gold_rows)

print("\n" + "=" * 72)
print(f"{'retriever':<32}{'R@1':>9}{'R@5':>9}{'R@10':>9}{'MRR':>9}")
print("-" * 72)
for k, v in results.items():
    print(f"{k:<32}{v['R@1']:>9.3f}{v['R@5']:>9.3f}{v['R@10']:>9.3f}{v['MRR']:>9.3f}")
print("=" * 72, flush=True)

probe_qs = ["what is metformin used for", "how to control blood pressure",
            "what are symptoms of diabetes", "what foods should diabetic patients avoid"]
pd_ = sk_cosine(embed(enc, probe_qs, MAX_Q), kb_emb_all)
ps = sk_cosine(tfidf.transform([clean_text(p) for p in probe_qs]), kb_tfidf)
blend = rerank(pd_, ps)
print("\nsanity check - the four questions the original got wrong:")
for i, p in enumerate(probe_qs):
    j = int(np.argmax(blend[i]))
    print(f"  Q: {p}")
    print(f"     matched: {df['question'].iloc[j][:78]}")
    print(f"     answer : {df['answer'].iloc[j][:110]}...", flush=True)

torch.save(enc.state_dict(), MODELS / "bilstm_fixed.pt")
json.dump({"vocab": vocab_words, "config": dict(
    VOCAB_SIZE=VOCAB_SIZE, MAX_Q=MAX_Q, MAX_A=MAX_A, EMB=EMB, HID=HID,
    LAYERS=LAYERS, DROP=DROP)}, open(MODELS / "vocab_fixed.json", "w"))
json.dump(results, open(ROOT / "outputs/retrieval_metrics.json", "w"), indent=2)
print("\nsaved bilstm_fixed.pt, vocab_fixed.json, outputs/retrieval_metrics.json")
