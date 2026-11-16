# Llama 3.2 1B Instruct - Türkçe LoRA SFT (Kaggle, tek GPU)
# Veri: turkish-lora-layers deneyindeki 6.5 bin Türkçe talimat (Aya + GPT-4 Self-Instruct Turkish).
# LoRA: tüm lineer katmanlar, r=16 (Qwen deneyinde en iyi sonuç veren yerleşim).
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Tüm çıktı /kaggle/working/run.log'a da yazılır; hata olursa traceback orada kalır.
LOG = open("/kaggle/working/run.log", "a", buffering=1, encoding="utf-8")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = Tee(sys.__stdout__, LOG)
sys.stderr = Tee(sys.__stderr__, LOG)


def on_error(t, v, tb):
    traceback.print_exception(t, v, tb, file=LOG)
    LOG.flush()
    sys.__excepthook__(t, v, tb)


sys.excepthook = on_error


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout[-4000:], flush=True)
    if p.returncode:
        raise RuntimeError(f"komut başarısız ({p.returncode}): {cmd}")


t_start = time.time()
# transformers v5 bazı TrainingArguments ayarlarını kaldırdı; bilinen çalışan sürümlere sabitle
sh('pip install -q "transformers==4.57.1" "peft==0.17.1" "accelerate>=1.0" "datasets>=3.0" 2>&1 | grep -v bigframes || true')
sh('python -c "import transformers, peft; print(transformers.__version__, peft.__version__)"')
sh("pip uninstall -y -q torchao || true")  # Kaggle'ın eski torchao'su peft ile çakışıyor
sh("nvidia-smi -L")

sh("git clone -q https://github.com/ayse-solmaz/turkish-lora-layers.git /kaggle/working/tll")
os.chdir("/kaggle/working/tll")
sh("python src/prepare_data.py")
sh("wc -l data/*.jsonl")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed  # noqa: E402

BASE = "unsloth/Llama-3.2-1B-Instruct"  # meta-llama/Llama-3.2-1B-Instruct ile aynı ağırlıklar, token gerektirmez
META_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUT = "/kaggle/working/Llama-3.2-1B-Instruct-tr-lora"
MAXLEN = 1024
IGNORE = -100
EOT = "<|eot_id|>"
set_seed(42)

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = "<|finetune_right_pad_id|>"
EOT_ID = tok.convert_tokens_to_ids(EOT)


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode(row):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
    )
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(row["response"] + EOT, add_special_tokens=False)["input_ids"]
    if len(p) >= MAXLEN - 8:
        return None
    return {"input_ids": (p + r)[:MAXLEN], "labels": ([IGNORE] * len(p) + r)[:MAXLEN]}


def load(path):
    return [x for x in (encode(r) for r in read(path)) if x]


train, val, test = load("data/train.jsonl"), load("data/val.jsonl"), load("data/test.jsonl")
test_rows = read("data/test.jsonl")
print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)


def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [IGNORE] * (m - len(b["labels"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]),
    }


@torch.no_grad()
def eval_loss(model, data, bs=8):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(data), bs):
        b = {k: v.cuda() for k, v in collate(data[i : i + bs]).items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**b)
        ntok = (b["labels"][:, 1:] != IGNORE).sum().item()
        total += out.loss.float().item() * ntok
        n += ntok
    return round(total / n, 4)


PROMPTS = [r["prompt"] for r in test_rows[:3]] + [
    "Türkiye'nin en uzun nehri hangisidir ve nereden doğar?",
    "Yeni başlayan biri için Python öğrenmeye dair üç kısa tavsiye verir misin?",
]


@torch.no_grad()
def generate(model, q):
    model.eval()
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(
            **ids, max_new_tokens=200, do_sample=False, repetition_penalty=1.1, use_cache=True,
            pad_token_id=tok.pad_token_id, eos_token_id=[EOT_ID, tok.eos_token_id],
        )
    return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()


# Ayarlar model yüklenmeden önce kurulur: yanlış bir ayar varsa koşu ilk dakikada düşer
args = TrainingArguments(
    output_dir="/kaggle/working/ckpt", per_device_train_batch_size=4, gradient_accumulation_steps=4,
    num_train_epochs=1, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    logging_steps=20, eval_strategy="steps", eval_steps=100, per_device_eval_batch_size=8,
    save_strategy="no", fp16=True, report_to="none", group_by_length=True, disable_tqdm=True,
    remove_unused_columns=False, seed=42, dataloader_num_workers=2,
)

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).cuda()
gpu = torch.cuda.get_device_name(0)
print("GPU:", gpu, flush=True)

base_test, base_val = eval_loss(model, test), eval_loss(model, val)
print(f"BASE  test_loss={base_test} val_loss={base_val}", flush=True)
base_samples = {q: generate(model, q) for q in PROMPTS}

cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, cfg)
for p in model.parameters():  # fp16 base + fp32 LoRA (GradScaler fp16 gradyan kabul etmez)
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=collate)
t0 = time.time()
trainer.train()
train_min = round((time.time() - t0) / 60, 1)

model.gradient_checkpointing_disable()
model.config.use_cache = True
tuned_test, tuned_val = eval_loss(model, test), eval_loss(model, val)
print(f"TUNED test_loss={tuned_test} val_loss={tuned_val}", flush=True)
tuned_samples = {q: generate(model, q) for q in PROMPTS}

model.save_pretrained(OUT)
cfg_path = os.path.join(OUT, "adapter_config.json")
with open(cfg_path) as f:
    c = json.load(f)
c["base_model_name_or_path"] = META_ID
with open(cfg_path, "w") as f:
    json.dump(c, f, indent=2)

results = {
    "base_model": META_ID,
    "gpu": gpu,
    "data": {"train": len(train), "val": len(val), "test": len(test)},
    "lora": {"r": 16, "alpha": 32, "targets": "all linear", "trainable_params": trainable},
    "train_minutes": train_min,
    "total_minutes": round((time.time() - t_start) / 60, 1),
    "test_loss": {"base": base_test, "lora": tuned_test},
    "val_loss": {"base": base_val, "lora": tuned_val},
    "samples": [{"prompt": q, "base": base_samples[q], "lora": tuned_samples[q]} for q in PROMPTS],
    "log_history": trainer.state.log_history,
}
with open("/kaggle/working/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in results.items() if k != "log_history"}, ensure_ascii=False, indent=2), flush=True)

os.chdir("/kaggle/working")
shutil.rmtree("/kaggle/working/tll", ignore_errors=True)
shutil.rmtree("/kaggle/working/ckpt", ignore_errors=True)
print("BITTI", flush=True)
