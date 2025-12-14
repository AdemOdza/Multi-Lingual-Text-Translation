import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_inp = "/content/data/hicm/hicm_corpus.test.txt"
test_out = "/content/data/hicm/train.txt"

# ------------------------------------------------------------
# 1. List of models to compare
# ------------------------------------------------------------
MODELS = {
    "opus_hi_en": "Helsinki-NLP/opus-mt-hi-en",
    "opus_mul_en": "Helsinki-NLP/opus-mt-mul-en",
    "m2m100": "facebook/m2m100_418M",
    "nllb": "facebook/nllb-200-distilled-600M",
}

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------
with open(test_inp, "r", encoding="utf-8") as f:
    test_src = [x.strip() for x in f]

with open(test_out, "r", encoding="utf-8") as f:
    test_tgt = [x.strip() for x in f]

print(f"Loaded {len(test_src)} total samples.")

# ------------------------------------------------------------
# 3. Sample ONLY 500 random pairs
# ------------------------------------------------------------
SAMPLE_SIZE = 500
indices = random.sample(range(len(test_src)), SAMPLE_SIZE)

sample_src = [test_src[i] for i in indices]
sample_tgt = [test_tgt[i] for i in indices]

print(f"Using {SAMPLE_SIZE} random samples for evaluation.")

# ------------------------------------------------------------
# 4. Translation helper
# ------------------------------------------------------------
def translate_sentence(model, tokenizer, sentence):
    if "m2m100" in tokenizer.name_or_path:
        tokenizer.src_lang = "hi"

    if "nllb" in tokenizer.name_or_path:
        tokenizer.src_lang = "hin_Deva"

    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)

    out = model.generate(
        **enc,
        max_length=256,
        num_beams=5,
    )

    return tokenizer.decode(out[0], skip_special_tokens=True)

# ------------------------------------------------------------
# 5. Evaluate all models
# ------------------------------------------------------------
results = {}

for nickname, model_name in MODELS.items():
    print("\n===============================================")
    print(f" Evaluating model: {model_name}")
    print("===============================================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    preds = []
    for src in tqdm(sample_src, total=len(sample_src)):
        preds.append(translate_sentence(model, tokenizer, src))

    bleu = sacrebleu.corpus_bleu(preds, [sample_tgt])
    results[nickname] = bleu.score

    print(f"\nModel: {nickname}")
    print(f"BLEU on 500 samples: {bleu.score:.2f}")

    # print a demo
    print("\nSample Translation:")
    print("SRC :", sample_src[0])
    print("PRED:", preds[0])
    print("TGT :", sample_tgt[0])

# ------------------------------------------------------------
# 6. Final ranking summary
# ------------------------------------------------------------
print("\n===============================================")
print(" FINAL RANKING (Higher BLEU = Better)")
print("===============================================")

for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:15s} : BLEU {score:.2f}")
