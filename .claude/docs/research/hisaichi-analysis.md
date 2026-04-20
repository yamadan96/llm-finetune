# Hisaichi Research Project - WebApp Deployment Analysis

Project root: `/home/yamada_24/projects/hisaichi/`

## 1. Model Architecture

- **Backbone**: DINOv2 ViT-L/14 (`vit_large_patch14_dinov2` via timm)
- **Fine-tuning**: LoRA (r=16, alpha=16, dropout=0.1) targeting `qkv` attention modules
- **Wrapper class**: `DINOv2MultiHeadModel` in `src/train.py`
  - Backbone -> `feature_transform` (Linear + ReLU + Dropout) -> classification heads
  - Main head: `head_full` (Linear 1024 -> 6 classes)
  - Auxiliary heads (optional): `head_damage` (2), `head_disaster_type` (2), `head_severity` (3)
- **LoRA modules_to_save**: `feature_transform`, `head_full`, `head_damage`, `head_disaster_type`, `head_severity`
- **Hidden dim**: 1024
- **peft version**: 0.18.1

## 2. 6-Class Definitions

Dataset location: `/rda5/users/yamada_24/storage/datasets/class6/`

| Index | Folder | Label (JP)  | Category    | Count |
|-------|--------|-------------|-------------|-------|
| 0     | `0`    | 被害なし     | No Damage   | 303   |
| 1     | `E1`   | 地震大       | Earthquake Large  | 313   |
| 2     | `E2`   | 地震中       | Earthquake Medium | 112   |
| 3     | `E3`   | 地震小       | Earthquake Small  | 98    |
| 4     | `T1`   | 津波大       | Tsunami Large     | 88    |
| 5     | `T3`   | 津波小       | Tsunami Small     | 126   |

Total: 1040 images. ImageFolder ordering: 0 < E1 < E2 < E3 < T1 < T3 (alphabetical).

## 3. Checkpoint Format & Best Model Location

Two checkpoint formats are saved per experiment:

1. **`best_model.pth`** (~1.2 GB) - Full state dict (backbone + LoRA + heads). Loaded via `model.load_state_dict()`.
2. **`lora_adapter/`** (~11 MB) - PEFT adapter only. Contains `adapter_config.json` + `adapter_model.safetensors`. Can be loaded via `PeftModel.from_pretrained()`.

155 checkpoints exist across experiments. Best candidates for WebApp:

- Ablation baseline (seed=42): `experiments/ablation/baseline_s42_aux_s42/`
- Any `covt_e0*` checkpoint (11 seeds, used in selective classification analysis)

Config per experiment: `config.json` in the same directory.

## 4. Input Preprocessing

From `get_transforms()` in `src/train.py`:

**Validation / Inference transforms**:
```python
transforms.Compose([
    transforms.Resize(int(image_size * 1.1)),  # e.g., 518 * 1.1 = 569
    transforms.CenterCrop(image_size),          # e.g., 518
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

- **Image size**: 518 (ablation experiments) or 392 (default config). Check `config.json` per checkpoint.
- **Center crop**: enabled by default (`use_center_crop: true`)
- Input format: RGB PIL Image -> Tensor [B, 3, image_size, image_size]

## 5. Inference Flow

### Entry point (existing)

`paper_reproduction/src/eval_inference_enhancements.py` has the cleanest inference code:

```python
# From eval_inference_enhancements.py
def build_model(config, device):
    model = DINOv2MultiHeadModel(config).to(device)
    lora_config = LoraConfig(r=16, lora_alpha=16, target_modules=["qkv"], ...)
    model = get_peft_model(model, lora_config)
    return model

def load_checkpoint(model, exp_dir, device):
    state_dict = torch.load(exp_dir / "best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
```

### Output format

`model(image_tensor)` returns a dict:
```python
{
    "full": torch.Tensor,        # (B, 6) - main classification logits
    "damage": torch.Tensor,      # (B, 2) - optional, may be None
    "disaster_type": torch.Tensor, # (B, 2) - optional, may be None
    "severity": torch.Tensor,    # (B, 3) - optional, may be None
}
```

For inference: `logits = outputs["full"]`, then `probs = softmax(logits)`, `pred = argmax(logits)`.

## 6. Selective Classification Implementation

**Full implementation**: `paper_reproduction/analysis/confidence_analysis.py` (~830 LOC)

Key mechanisms:

### Standard threshold-based rejection
```python
def compute_risk_coverage(probs, preds, labels, thresholds):
    max_probs = np.max(probs, axis=1)
    for theta in thresholds:
        mask = max_probs >= theta  # Accept if confidence >= threshold
        # Compute metrics only on accepted samples
```

### Cost-sensitive selective classification
```python
def compute_cost_sensitive_selective(probs, preds, labels, normal_threshold, cher_threshold, ambiguity_threshold=0.2):
    for each sample:
        if cross_hazard_ambiguous(probs, pred):
            accept = max_prob >= cher_threshold     # Stricter for ambiguous
        else:
            accept = max_prob >= normal_threshold   # Normal threshold
```

Cross-hazard ambiguity: if predicted E* but any T* class has prob > 0.2 (or vice versa).

### Best reported results (from MEMORY.md)
- ECE = 0.1147
- theta=0.70: Coverage 69.1%, Acc 76.9%, CHER 8.87%
- Cost-sensitive (0.50/0.80): Coverage 84.7%, CHER 8.80%

## 7. Existing App Infrastructure

- **FastAPI + Uvicorn already in pyproject.toml** (fastapi>=0.115, uvicorn>=0.32)
- **annotation_app** (`src/annotation_app/__main__.py`): Bounding box annotation web app using FastAPI. Can serve as template for the inference WebApp.

## 8. Missing Pieces for WebApp Deployment

1. **No inference API endpoint** - Need to create a FastAPI route that accepts an image upload, runs the model, and returns classification results + confidence.
2. **No model loading singleton** - Need to load model once at startup (not per-request). The `eval_inference_enhancements.py::build_model()` + `load_checkpoint()` pattern can be reused.
3. **No selective classification integration in API** - The `confidence_analysis.py` functions are analysis-only (batch mode). Need to wrap `compute_cost_sensitive_selective` logic for single-image real-time use.
4. **No frontend** - Need a web UI for image upload, result display, confidence visualization. The annotation_app HTML/JS template can be adapted.
5. **GPU/CPU device management** - Need to detect available device and handle gracefully.
6. **Config resolution** - Need to decide which checkpoint to ship (recommend: best single seed from ablation baseline or a seed with highest accuracy).
7. **Image size mismatch** - ablation configs use image_size=518, default ExperimentConfig uses 392. Must match checkpoint's training config.
8. **Cost matrix** - `create_cost_matrix()` in train.py is needed for cost-aware decoding and cost-sensitive thresholds.

## 9. Key Files for WebApp Development

| Purpose | File |
|---------|------|
| Model class | `/home/yamada_24/projects/hisaichi/src/train.py` (`DINOv2MultiHeadModel`) |
| Config dataclass | `/home/yamada_24/projects/hisaichi/src/utils/config.py` (`ExperimentConfig`) |
| Metrics | `/home/yamada_24/projects/hisaichi/src/utils/metrics.py` (`ExtendedMetrics`) |
| Model loading pattern | `/home/yamada_24/projects/hisaichi/paper_reproduction/src/eval_inference_enhancements.py` |
| Selective classification | `/home/yamada_24/projects/hisaichi/paper_reproduction/analysis/confidence_analysis.py` |
| Cost-aware decoding | `/home/yamada_24/projects/hisaichi/src/patches/cost_aware_decoding.py` |
| Existing FastAPI app | `/home/yamada_24/projects/hisaichi/src/annotation_app/__main__.py` |
| Cost matrix | `/home/yamada_24/projects/hisaichi/src/train.py` (`create_cost_matrix()`) |
| Transforms | `/home/yamada_24/projects/hisaichi/src/train.py` (`get_transforms()`) |
| Example checkpoint | `/home/yamada_24/projects/hisaichi/experiments/ablation/baseline_s42_aux_s42/` |
| Dataset | `/rda5/users/yamada_24/storage/datasets/class6/` |
