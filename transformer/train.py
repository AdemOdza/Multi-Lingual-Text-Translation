import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from transformer_model import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_attn_mask_from_padding(pad_mask):
    # pad_mask: [B, T] True=pad
    return pad_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]


def create_causal_mask(T):
    return torch.triu(torch.ones(T, T), diagonal=1).bool()  # [T, T]


def train(model, dataloader, epochs=10, lr=5e-4):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            src = batch["src_tokens"].to(DEVICE)
            tgt_in = batch["tgt_input"].to(DEVICE)
            tgt_out = batch["tgt_output"].to(DEVICE)

            src_pad = batch["src_padding_mask"].to(DEVICE)
            tgt_pad = batch["tgt_padding_mask"].to(DEVICE)

            src_mask = create_attn_mask_from_padding(src_pad)
            tgt_mask_padding = create_attn_mask_from_padding(tgt_pad)

            causal = create_causal_mask(tgt_in.size(1)).to(DEVICE)

            # Combine (padding OR causal)
            tgt_mask = tgt_mask_padding | causal

            logits = model(src, tgt_in, src_mask, tgt_mask)

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss / len(dataloader):.4f}")

