"""Kaggle çıktısını (LoRA adapter + results.json) model kartıyla birlikte Hugging Face'e yükler.

Kullanım:  python upload_to_hf.py <kaggle_output_klasoru>
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "ayse-solmaz/Llama-3.2-1B-Instruct-tr-lora"
KAGGLE_URL = "https://www.kaggle.com/code/aysenursolmaz/llama-3-2-1b-tr-lora"
STUDY_URL = "https://github.com/ayse-solmaz/turkish-lora-layers"

out = Path(sys.argv[1])
adapter = out / "Llama-3.2-1B-Instruct-tr-lora"
r = json.loads((out / "results.json").read_text(encoding="utf-8"))

samples = "\n\n".join(
    f"**Soru:** {s['prompt']}\n\n- *Base:* {s['base'][:400]}\n- *LoRA:* {s['lora'][:400]}" for s in r["samples"]
)

card = f"""---
base_model: meta-llama/Llama-3.2-1B-Instruct
library_name: peft
license: llama3.2
language:
- tr
pipeline_tag: text-generation
tags:
- llama
- llama-3.2
- lora
- peft
- turkish
- instruction-tuning
datasets:
- CohereLabs/aya_dataset
- CausalLM/GPT-4-Self-Instruct-Turkish
---

# Llama-3.2-1B-Instruct-tr-lora

**Built with Llama.** Meta'nın [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
modeli için Türkçe talimat verisiyle eğitilmiş bir LoRA adapter'ı.

*A LoRA adapter for Meta's Llama 3.2 1B Instruct, fine-tuned on Turkish instruction data.*

## Sonuçlar

| | Base | LoRA |
|---|---|---|
| Türkçe test kaybı (Aya test, {r['data']['test']} örnek) | {r['test_loss']['base']} | **{r['test_loss']['lora']}** |
| Doğrulama kaybı ({r['data']['val']} örnek) | {r['val_loss']['base']} | **{r['val_loss']['lora']}** |

Tek seed (42), tek koşu.

## Eğitim

- **Veri:** {r['data']['train']} Türkçe talimat-cevap çifti (CohereLabs/aya_dataset Türkçe kısmı + CausalLM/GPT-4-Self-Instruct-Turkish). Test seti Aya'nın kendi Türkçe test bölümü, eğitimde hiç görülmedi.
- **LoRA:** r={r['lora']['r']}, alpha={r['lora']['alpha']}, dropout 0.05, tüm lineer katmanlar (q, k, v, o, gate, up, down), {r['lora']['trainable_params']:,} eğitilebilir parametre.
- **Ayarlar:** 1 epoch, efektif batch 16, lr 2e-4 (cosine), fp16, maksimum 1024 token, kayıp sadece cevap token'larında.
- **Donanım:** {r['gpu']} (Kaggle), eğitim {r['train_minutes']} dakika.
- **Kod ve koşu:** [Kaggle]({KAGGLE_URL})

LoRA yerleşimi (tüm lineer katmanlar, r=16), Qwen2.5-0.5B üzerinde yaptığım
[LoRA yerleşimi deneyinin]({STUDY_URL}) sonuçlarına göre seçildi; aynı veri ve aynı test seti kullanıldı.

## Örnek cevaplar (greedy, 200 token)

{samples}

## Kullanım

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "meta-llama/Llama-3.2-1B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "{REPO}")

msgs = [{{"role": "user", "content": "Türkiye'nin en uzun nehri hangisidir?"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=200)[0][ids.shape[1]:], skip_special_tokens=True))
```

## Lisans

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
Bu adapter [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) ve
[Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy/) koşullarına tabidir.
"""

(adapter / "README.md").write_text(card, encoding="utf-8")
(adapter / "NOTICE").write_text(
    "Llama 3.2 is licensed under the Llama 3.2 Community License, "
    "Copyright © Meta Platforms, Inc. All Rights Reserved.\n"
    "License: https://www.llama.com/llama3_2/license/\n",
    encoding="utf-8",
)
(adapter / "results.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path=str(adapter), repo_id=REPO, repo_type="model",
                  commit_message="Llama 3.2 1B Instruct Turkish LoRA adapter")
print("https://huggingface.co/" + REPO)
