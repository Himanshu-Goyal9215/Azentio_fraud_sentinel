# Fraud Sentinel

End-to-end banking fraud detection pipeline using a fine-tuned Llama 3.2 1B Instruct model with QLoRA.

Built for the Azentio AI Engineering Hackathon.

## Results

| Metric | Base Model | Fine-Tuned |
|---|---|---|
| Valid JSON Output | 100% | 100% |
| Fraud Detection Accuracy | 30% | **50%** |

## How It Works

```
transactions.csv ──┐
accounts.csv ──────┤──→ MERGE ──→ CLEAN ──→ SANITIZE ──→ LABEL ──→ TRAIN ──→ EVALUATE ──→ PREDICT
customers.csv ─────┘
```

1. **Merge** 3 relational banking tables using LEFT JOIN
2. **Clean** with domain-specific logic (not blind imputation — banking rules matter)
3. **Sanitize** text fields for adversarial prompt injection attempts
4. **Label** using weighted weak supervision heuristics (no ground truth provided)
5. **Benchmark** the base Llama 3.2 1B model on a holdout set
6. **Fine-tune** using QLoRA with NEFTune, cosine scheduler, `all-linear` target modules
7. **Evaluate** fine-tuned model on same holdout set (side-by-side comparison)
8. **Predict** on all transactions and export strict JSON

## Project Structure

```
├── fraud_sentinel.ipynb          # Main notebook (run this in Google Colab)
├── pipeline_architecture.md      # Detailed architecture documentation
├── data/
│   ├── transactions.csv          # 1000 transactional records (messy)
│   ├── accounts.csv              # 178 account records
│   └── customers.csv             # 124 customer records
├── fraud_lora_adapter/           # Saved LoRA adapter weights (~20MB)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── predictions.json              # Final output (strict JSON per transaction)
```

## Quick Start

### 1. Open in Google Colab
Upload `fraud_sentinel.ipynb` to [Google Colab](https://colab.research.google.com/).

### 2. Set GPU Runtime
`Runtime` → `Change runtime type` → Select **T4 GPU**

### 3. Upload Data
Upload `transactions.csv`, `accounts.csv`, and `customers.csv` to the Colab file browser (they go to `/content/`).

### 4. Run All Cells
The notebook will prompt for your Hugging Face token (get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)). Paste it in the secure input box and run all cells top to bottom.

## Tech Stack

| Component | Tool |
|---|---|
| Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Fine-tuning | QLoRA via `peft` + `trl` (SFTTrainer) |
| Quantization | 4-bit NF4 via `bitsandbytes` |
| Data | `pandas`, `numpy` |
| Labeling | Weak supervision (rule-based fraud scoring) |
| Defense | Regex-based prompt injection sanitization |

## Key Design Decisions

**Why fine-tune instead of just prompting?**
Long prompts are expensive, unreliable, and slow. Fine-tuning encodes domain knowledge directly into the weights.

**Why weak supervision?**
No ground truth `is_fraud` label was provided. We used banking domain heuristics (overdraft detection, velocity checks, AML flags) to create synthetic labels.

**Why `target_modules='all-linear'`?**
Hardcoding layer names like `q_proj` only works for Llama. `all-linear` auto-detects the right layers for ANY model architecture.

**Why balanced sampling?**
Imbalanced data teaches the model to always predict the majority class. We sample equal fraud/safe examples.

**Why bf16 instead of fp16?**
Llama 3.2 stores weights in BFloat16. The fp16 gradient scaler crashes on BFloat16 tensors.

## Output Format

Every prediction follows this strict JSON schema:

```json
{
  "transaction_id": "TXN_00001",
  "is_fraud": true,
  "confidence": 0.85,
  "justification": "Spend 5x above customer average; Transaction caused account overdraft"
}
```

## License

MIT
