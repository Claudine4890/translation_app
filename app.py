import json

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Model (same classes as in the notebook) ----------
class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class Block(nn.Module):
    def __init__(self, n_embd, block_size, use_ln=True):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd)
        )
        self.ln1 = MyLayerNorm(n_embd) if use_ln else nn.Identity()
        self.ln2 = MyLayerNorm(n_embd) if use_ln else nn.Identity()

    def forward(self, x):
        h = self.ln1(x)
        B, T, C = h.shape
        k, q, v = self.key(h), self.query(h), self.value(h)
        scores = q @ k.transpose(-2, -1) / C ** 0.5
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        x = x + w @ v
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=32, n_layer=2, use_ln=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, block_size, use_ln) for _ in range(n_layer)])
        self.ln_f = MyLayerNorm(n_embd) if use_ln else nn.Identity()
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------- Loading ----------
@st.cache_resource
def load_models():
    with open("config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    models = {}
    for name, file, use_ln in [
        ("With layer norm", "model_ln.pt", True),
        ("Without layer norm", "model_no.pt", False),
    ]:
        m = TinyGPT(cfg["vocab_size"], cfg["block_size"], n_embd=cfg["n_embd"],
                    n_layer=cfg["n_layer"], use_ln=use_ln)
        m.load_state_dict(torch.load(file, map_location="cpu"))
        m.eval()
        models[name] = m
    return cfg, models


def translate(model, cfg, sentence, max_new=60):
    stoi = {c: i for i, c in enumerate(cfg["chars"])}
    itos = {i: c for c, i in stoi.items()}
    prompt = sentence.lower().strip().replace("\u2019", "'") + " = "
    ids = [stoi[c] for c in prompt if c in stoi]
    x = torch.tensor([ids])
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(x[:, -cfg["block_size"]:])
            nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
            if itos[nxt.item()] == "\n":
                break
    return "".join(itos[i] for i in x[0].tolist()).split(" = ", 1)[-1].strip()


# ---------- Page ----------
st.set_page_config(page_title="Kinyarwanda to English: tiny transformer", page_icon="🔤")
st.title("Kinyarwanda to English: tiny transformer")
st.caption(
    "A character-level transformer trained from scratch on 3,000 short sentence pairs, "
    "built to study layer normalization. Expect wrong translations. Group 3."
)

cfg, models = load_models()

choice = st.selectbox("Example sentences from the training data", ["(type my own)"] + cfg["examples"])
sentence = st.text_input("Kinyarwanda sentence", value="" if choice == "(type my own)" else choice)

if st.button("Translate", type="primary"):
    if not sentence.strip():
        st.warning("Please type or choose a sentence first.")
    else:
        col1, col2 = st.columns(2)
        for col, (name, model) in zip([col1, col2], models.items()):
            with col:
                st.subheader(name)
                st.write(translate(model, cfg, sentence))

st.divider()
st.write(
    "Both models have all five parts of the transformer block (embedding, attention, "
    "feed-forward, residual connections and normalization). The only difference is that "
    "the second one has layer normalization switched off."
)
