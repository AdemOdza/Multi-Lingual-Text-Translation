import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from pathlib import Path
from tqdm import tqdm
import sacrebleu
from sacrebleu.metrics import BLEU
import pandas as pd

DEVICE = "cuda"
DATA_DIR = Path("/content/data/hicm")
BATCH_SIZE = 32
MAX_LEN = 128
TRAIN_LIMIT = 200000
train_source_path = DATA_DIR / "hicm_corpus.train.txt"
train_target_path = DATA_DIR / "train.txt"
valid_source_path = DATA_DIR / "hicm_corpus.valid.txt"
valid_target_path = DATA_DIR / "valid.txt"
test_source_path  = DATA_DIR / "hicm_corpus.test.txt"
test_target_path  = DATA_DIR / "test.txt"


# Tokenizer + mBERT embeddings

#Multilingual BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

#BERT embeddings 
mbert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

# we're only using the mbert word embeddings
mbert_embeddings = mbert_model.embeddings.word_embeddings

# freeze embeddings (don't update while training)
mbert_embeddings.requires_grad_(False)


# Dataset and batching

def batch(source_texts, target_texts, maxlength=MAX_LEN):
    sourcenc = tokenizer(
        source_texts,
        padding=True,
        truncation=True,
        max_length=maxlength,
        return_tensors="pt"
    )
    targenc = tokenizer(
        target_texts,
        padding=True,
        truncation=True,
        max_length=maxlength,
        return_tensors="pt"
    )
    s = sourcenc["input_ids"].to(DEVICE) 
    t = targenc["input_ids"].to(DEVICE)


    sourcepaddingmask = (sourcenc["attention_mask"] == 0).to(DEVICE)
    targetpaddingmask = (targenc["attention_mask"] == 0).to(DEVICE)
    # shift target
    input = t[:, :-1]

    out = t[:, 1:] # Remove first token
    targetpaddingmask = targetpaddingmask[:, :-1]
    return {
        "source_tokens": s,
        "input": input,
        "out": out,
        "sourcepaddingmask": sourcepaddingmask,
        "targetpaddingmask": targetpaddingmask,
    }


class TranslationDataset(Dataset):
    def __init__(self, source_file, target_file, limit=None):
        with open(source_file, encoding="utf-8") as f:
            self.source = [l.strip() for l in f if l.strip()]

        with open(target_file, encoding="utf-8") as f:
            self.target = [l.strip() for l in f if l.strip()]

        if limit is not None:
            self.source = self.source[:limit]
            self.target = self.target[:limit]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return {
            "source_text": self.source[index],
            "target_text": self.target[index]
        }
    
def collate_fn(batch):
    source_texts = [b["source_text"] for b in batch]
    target_texts = [b["target_text"] for b in batch]
    return batch(source_texts, target_texts)


#data loading
train_dataset = TranslationDataset(train_source_path, train_target_path, limit=TRAIN_LIMIT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_dataset = TranslationDataset(test_source_path, test_target_path)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)




# Transformer


#transformer modules
class PositionalEncoding(nn.Module):
    def __init__(self, moddim, maxlength=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(maxlength, moddim)
        pos = torch.arange(0, maxlength).unsqueeze(1)
        div = torch.exp(torch.arange(0, moddim, 2) * (-math.log(10000.0) / moddim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class MultiHeadAttention(nn.Module):
    def __init__(self, moddim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = moddim // heads
        self.q = nn.Linear(moddim, moddim)
        self.k = nn.Linear(moddim, moddim)
        self.v = nn.Linear(moddim, moddim)
        self.output = nn.Linear(moddim, moddim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, msk=None):
        B = q.size(0)

        def split(x):
            return x.view(B, -1, self.heads, self.head_dim).transpose(1, 2)

        Q = split(self.q(q))
        K = split(self.k(k))
        V = split(self.v(v))
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim) #calculate scores

        if msk is not None:
            scores = scores.masked_fill(msk, -1e9)

        attention = self.dropout(F.softmax(scores, dim=-1))
        output = attention @ V
        output = output.transpose(1, 2).contiguous().view(B, -1, self.heads * self.head_dim)
        return self.output(output)


class FeedForward(nn.Module):
    def __init__(self, moddim, feedforwarddim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(moddim, feedforwarddim)
        self.fc2 = nn.Linear(feedforwarddim, moddim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, moddim, heads, feedforwarddim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(moddim, heads, dropout)
        self.ff = FeedForward(moddim, feedforwarddim, dropout)
        self.ln1 = nn.LayerNorm(moddim)
        self.ln2 = nn.LayerNorm(moddim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, msk=None):
        x = x + self.dropout(self.self_attention(self.ln1(x), self.ln1(x), self.ln1(x), msk))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, moddim, heads, feedforwarddim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(moddim, heads, dropout)
        self.cross_attention = MultiHeadAttention(moddim, heads, dropout)
        self.ff = FeedForward(moddim, feedforwarddim, dropout)
        self.ln1 = nn.LayerNorm(moddim)
        self.ln2 = nn.LayerNorm(moddim)
        self.ln3 = nn.LayerNorm(moddim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, memory, target_mask=None, mem_mask=None):
        x = x + self.dropout(self.self_attention(self.ln1(x), self.ln1(x), self.ln1(x), target_mask))
        x = x + self.dropout(self.cross_attention(self.ln2(x), memory, memory, mem_mask))
        x = x + self.dropout(self.ff(self.ln3(x)))
        return x


# transformer with frozen mBERT embeddings

class TransformerMBERT(nn.Module):
    def __init__(self, embeddings, moddim=256, heads=8, layers=2, feedforwarddim=1024, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(768, moddim) # we are projecting 768 to model size = 256 for performance

        self.pos = PositionalEncoding(moddim)# positional embedding
        self.embedd = embeddings #frozen mbert embeddings

        #two layer encouder decoder
        self.encoder = nn.ModuleList([
            EncoderLayer(moddim, heads, feedforwarddim, dropout) for _ in range(layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(moddim, heads, feedforwarddim, dropout) for _ in range(layers)
        ])
        # project output to mBERT vocab
        self.lm_head = nn.Linear(moddim, embeddings.num_embeddings)

    def forward(self, s, t, source_mask=None, target_mask=None):
        target = self.pos(self.proj(self.embedd(t))) #[B,T,768]

        source = self.pos(self.proj(self.embedd(s))) #[B,T,256]
        for layer in self.encoder:
            source = layer(source, source_mask)
        for layer in self.decoder:
            target = layer(target, source, target_mask, source_mask)

        return self.lm_head(target)



# seq2seq

# attention mechanism where model only attends to input sequence relevant to current prediction
class BahdanauAttention(nn.Module): 
    def __init__(self, hidden_dim):
        super().__init__()
        self.Wd = nn.Linear(hidden_dim, hidden_dim)
        self.We = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, dec_hidden, enc_out, source_mask):
        T = enc_out.size(1)
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, T, 1)
        energy = torch.tanh(self.Wd(dec_hidden)+ self.We(enc_out))
        scores = self.v(energy).squeeze(-1).masked_fill(source_mask, -1e10)
        attention = F.softmax(scores, dim=1)
        context = torch.bmm(attention.unsqueeze(1), enc_out).squeeze(1)
        return context

class EncoderLSTM(nn.Module):
    def __init__(self, moddim, hidden_dim, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            moddim, hidden_dim // 2,
            layers, bidirectional=True,
            dropout=dropout if layers > 1 else 0,
            batch_first=True
        )

    def forward(self, x):
        output, (h, c) = self.lstm(x)
        h = self._merge(h)
        c = self._merge(c)
        return output, h, c

    def _merge(self, states):
        layers = states.size(0) // 2
        states = states.view(layers, 2, states.size(1), states.size(2))
        return torch.cat([states[:, 0], states[:, 1]], dim=-1)


class DecoderLSTM(nn.Module):
    def __init__(self, moddim, hidden_dim, vocab_size, layers, dropout):
        super().__init__()
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(
            moddim + hidden_dim,
            hidden_dim,
            layers,
            dropout=dropout if layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h, c, enc_out, source_mask):
        context = self.attention(h[-1], enc_out, source_mask)
        lstm_in = torch.cat([x.unsqueeze(1), context.unsqueeze(1)], dim=2)
        output, (h, c) = self.lstm(lstm_in, (h, c))
        return self.fc(output.squeeze(1)), h, c


class Seq2SeqMBERT(nn.Module):
    def __init__(self, embeddings, moddim=256, layers=2, dropout=0.1):
        super().__init__()
       # we get mBERT embeddings and then project
        self.embedd = embeddings
        self.proj = nn.Linear(768, moddim)
        self.encoder = EncoderLSTM(moddim, moddim, layers, dropout)
        self.decoder = DecoderLSTM(moddim, moddim, embeddings.num_embeddings, layers, dropout)

    def forward(self, s, t, source_mask, tf=0.5):
        source = self.proj(self.embedd(s))
        target = self.proj(self.embedd(t))
        enc_out, h, c = self.encoder(source)
        outputs = []
        x = target[:, 0]
        for t in range(1, target.size(1)):
            output, h, c = self.decoder(x, h, c, enc_out, source_mask)
            outputs.append(output)
            use_tf = torch.rand(1).item() < tf
            x = target[:, t] if use_tf else self.proj(self.embedd(output.argmax(1)))
        return torch.stack(outputs, dim=1)

#traiining function
def create_attn_mask_from_padding(pad_mask):
    return pad_mask.unsqueeze(1).unsqueeze(2)

def create_causal_mask(T):
    # casaul mask
    return torch.triu(torch.ones(T, T), diagonal=1).bool()

def train_model(model, dataloader, epochs=10, lr=5e-4, model_name="model"):
    # training loop
      model = model.to(DEVICE)
      optimizer = optim.AdamW(model.parameters(), lr=lr)
      criterion = nn.CrossEntropyLoss(ignore_index=0)

      for epoch in range(1, epochs + 1):
          model.train()
          total_loss = 0

          for batch in tqdm(dataloader, desc=f"{model_name} Epoch {epoch}"):
              src = batch["src_tokens"].to(DEVICE)
              tgt_in = batch["tgt_input"].to(DEVICE)
              tgt_out = batch["tgt_output"].to(DEVICE)
              src_pad = batch["src_padding_mask"].to(DEVICE)
              tgt_pad = batch["tgt_padding_mask"].to(DEVICE)

              if model_name=="Seq2Seq-mBERT":
                  # 2D mask for seq2seq
                  src_mask = src_pad
                  tgt_mask = None
              else:
                  # 4D mask for transformer
                  src_mask = create_attn_mask_from_padding(src_pad)
                  tgt_mask_padding = create_attn_mask_from_padding(tgt_pad)
                  causal = create_causal_mask(tgt_in.size(1)).to(DEVICE)
                  tgt_mask = tgt_mask_padding | causal

              # Forward pass
              logits = model(src, tgt_in, src_mask, tgt_mask)

              # Compute loss
              if model_name=="Seq2Seq-mBERT":
        
                  loss = criterion(
                      logits.reshape(-1, logits.size(-1)),
                      tgt_out[:, 1:].reshape(-1)  # Skip first token
                  )
              else:
                  
                  loss = criterion(
                      logits.reshape(-1, logits.size(-1)),
                      tgt_out.reshape(-1)
                  )

              # Backward pass
              optimizer.zero_grad()
              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              optimizer.step()

              total_loss += loss.item()

          avg_loss = total_loss / len(dataloader)
          print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")


transformer_mbert = TransformerMBERT(
    mbert_embeddings=mbert_embeddings,
    d_model=256,
    num_heads=8,
    num_layers=2,
    d_ff=1024,
    dropout=0.1
)
seq2seq_mbert = Seq2SeqMBERT(
    mbert_embeddings=mbert_embeddings,
    d_model=256,
    num_layers=2,
    dropout=0.1
)

total_params = sum(p.numel() for p in transformer_mbert.parameters())
trainable_params = sum(p.numel() for p in transformer_mbert.parameters() if p.requires_grad)
train_model(transformer_mbert, train_loader, epochs=10, lr=5e-4, model_name="Transformer-mBERT")
total_params = sum(p.numel() for p in seq2seq_mbert.parameters())
trainable_params = sum(p.numel() for p in seq2seq_mbert.parameters() if p.requires_grad)
train_model(seq2seq_mbert, train_loader, epochs=15, lr=5e-4, model_name="Seq2Seq-mBERT")



# inference
def translate_transformer(sentence, max_len=128):
  #Translate using Transformer-mBERT model
  transformer_mbert.eval()
  with torch.no_grad():
      src_encoded = tokenizer(
          [sentence],
          padding=True,
          truncation=True,
          max_length=max_len,
          return_tensors='pt'
      )
      src_tokens = src_encoded['input_ids'].to(DEVICE)
      src_padding_mask = (src_encoded['attention_mask'] == 0).to(DEVICE)


      tgt_tokens = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).to(DEVICE)

      for _ in range(max_len): #decode autoregressivelt
          tgt_padding_mask = torch.zeros_like(tgt_tokens, dtype=torch.bool).to(DEVICE)
          src_mask = create_attn_mask_from_padding(src_padding_mask)
          tgt_mask_padding = create_attn_mask_from_padding(tgt_padding_mask)
          causal = create_causal_mask(tgt_tokens.size(1)).to(DEVICE)
          tgt_mask = tgt_mask_padding | causal
          # Forward pass
          logits = transformer_mbert(src_tokens, tgt_tokens, src_mask, tgt_mask)
          next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
          if next_token.item() == tokenizer.sep_token_id:
              break

          tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

      return tokenizer.decode(tgt_tokens[0], skip_special_tokens=True)


def translate_seq2seq(sentence, max_len=128):
  #Translate using Seq2Seq-mBERT model
  seq2seq_mbert.eval()
  with torch.no_grad():

      src_encoded = tokenizer(
          [sentence],
          padding=True,
          truncation=True,
          max_length=max_len,
          return_tensors='pt'
      )
      src_tokens = src_encoded['input_ids'].to(DEVICE)
      src_padding_mask = (src_encoded['attention_mask'] == 0).to(DEVICE)
      src_embed = seq2seq_mbert.embed_proj(seq2seq_mbert.mbert_embed(src_tokens))
      encoder_outputs, hidden, cell = seq2seq_mbert.encoder(src_embed, src_padding_mask)
      input_token = torch.tensor([tokenizer.cls_token_id], dtype=torch.long).to(DEVICE)
      translated_tokens = []

      for _ in range(max_len):# decode autoregressively
          input_embed = seq2seq_mbert.embed_proj(seq2seq_mbert.mbert_embed(input_token))
          output, hidden, cell = seq2seq_mbert.decoder(input_embed, hidden, cell, encoder_outputs, src_padding_mask)
          next_token = output.argmax(dim=1)
          if next_token.item() in [tokenizer.sep_token_id, tokenizer.pad_token_id]:
              break

          translated_tokens.append(next_token.item())
          input_token = next_token

      return tokenizer.decode(translated_tokens, skip_special_tokens=True)


#bleu score calculation
def calculate_bleu(model_name, translate_fn, num_samples=500):
    #compute BLEU score on test data
    test_src_lines = open(test_source_path, "r", encoding="utf-8").read().splitlines()[:num_samples]
    test_tgt_lines = open(test_target_path, "r", encoding="utf-8").read().splitlines()[:num_samples]

    predictions = []
    references = []

    for src_text, ref_text in tqdm(zip(test_src_lines, test_tgt_lines), total=len(test_src_lines), desc=model_name):
        pred_text = translate_fn(src_text)
        predictions.append(pred_text)
        references.append(ref_text)

    bleu = BLEU()
    score = bleu.corpus_score(predictions, [references])
    return score.score, predictions, references


# Calculate BLEU for both models

transformer_bleu, transformer_preds, transformer_refs = calculate_bleu("Transformer-mBERT", translate_transformer, 500)
seq2seq_bleu, seq2seq_preds, seq2seq_refs = calculate_bleu("Seq2Seq-mBERT", translate_seq2seq, 500)

print(f"Transformer-mBERT BLEU: {transformer_bleu:.2f}")
print(f"Seq2Seq-mBERT BLEU:     {seq2seq_bleu:.2f}")

# print translations
for i in range(5):
    print(f"\n{i+1}. Source (Hindi):  {open(test_source_path).readlines()[i].strip()}")
    print(f"   Transformer:     {transformer_preds[i]}")
    print(f"   Seq2Seq:         {seq2seq_preds[i]}")
    print(f"   Reference:       {transformer_refs[i]}")