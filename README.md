# Llama 3.2 1B Instruct - Türkçe LoRA

Meta'nın [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) modelini Türkçe talimat verisiyle LoRA kullanarak eğitme projesi.

🤗 Model: [ayse-solmaz/Llama-3.2-1B-Instruct-tr-lora](https://huggingface.co/ayse-solmaz/Llama-3.2-1B-Instruct-tr-lora)

## Sonuçlar

| | Base | LoRA | Değişim |
|---|---|---|---|
| Türkçe test kaybı (Aya test, 250 örnek) | 3.2647 | **2.9307** | -%10.2 |
| Doğrulama kaybı (300 örnek) | 2.4318 | **1.9718** | -%18.9 |

Tek seed (42), tek koşu. Kaggle'da tek Tesla T4 GPU'da eğitim 25 dakika sürdü (11.3M eğitilebilir parametre, 6573 eğitim örneği). Örnek cevaplar ve tüm eğitim kayıtları için [model kartına](https://huggingface.co/ayse-solmaz/Llama-3.2-1B-Instruct-tr-lora) bakın.

## Kurulum

- **Veri:** ~6.5 bin Türkçe talimat-cevap çifti (CohereLabs/aya_dataset Türkçe kısmı + CausalLM/GPT-4-Self-Instruct-Turkish). Veri, [turkish-lora-layers](https://github.com/ayse-solmaz/turkish-lora-layers) projesindeki `prepare_data.py` ile hazırlanıyor. Test seti Aya'nın Türkçe test bölümü (250 örnek), eğitimde kullanılmıyor.
- **LoRA:** r=16, alpha=32, dropout 0.05, tüm lineer katmanlar (q, k, v, o, gate, up, down). Bu yerleşim, Qwen2.5-0.5B üzerinde yaptığım LoRA yerleşimi deneyinde en iyi sonucu verdiği için seçildi.
- **Eğitim:** 1 epoch, efektif batch 16, lr 2e-4 (cosine), fp16, maksimum 1024 token, kayıp sadece cevap token'larında, gradient checkpointing.
- **Değerlendirme:** Eğitim öncesi ve sonrası Türkçe test/doğrulama kaybı ve aynı sorulara verilen örnek cevaplar.

## Dosyalar

| Dosya | Ne yapar |
|---|---|
| `llama32_tr_lora.py` | Kaggle'da tek GPU'da çalışan eğitim scripti: kurulum, veri, base model değerlendirmesi, LoRA eğitimi, sonuçlar |
| `kernel-metadata.json` | Kaggle kernel ayarları (GPU, internet) |
| `upload_to_hf.py` | Kaggle çıktısını model kartıyla birlikte Hugging Face'e yükler |

## Çalıştırma

```bash
kaggle kernels push -p . --accelerator NvidiaTeslaT4
kaggle kernels output aysenursolmaz/llama-3-2-1b-tr-lora -p out
python upload_to_hf.py out
```

## Lisans

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
Model, [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) ve [Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy/) koşullarına tabidir.
