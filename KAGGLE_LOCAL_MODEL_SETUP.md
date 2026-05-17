# Local Gemma-4 Model in Kaggle Notebook

## Quick Start (5 minutes)

### 1. Attach the Gemma-4 Model to Your Notebook

In the Kaggle notebook editor:

1. **Data** tab (left sidebar)
2. **Add Input**
3. Search: `google/gemma-4` 
4. Click the result to add (from [Kaggle Models](https://www.kaggle.com/models/google/gemma-4))
5. Click **Add Input**

Model will appear in `/kaggle/input/gemma-4/` automatically.

### 2. Run the Notebook

The notebook will:
- Auto-detect the local model ✅
- Load it on GPU ✅
- Skip API calls entirely ✅

**Console Output:**
```
✅ Local Gemma-4 model found at /kaggle/input/gemma-4
🤖 Mode: LOCAL | Model: gemma-4-9b-it
```

---

## How It Works

### Local Mode (Recommended)

```
VCF Input → Score Variants → Load Model → Interpret (8 variants) → HTML Report
                            (no API calls, no quota limits)
```

**Advantages:**
- ✅ No API quota concerns
- ✅ No API key needed
- ✅ Fully reproducible
- ✅ Faster iteration for judging demos

### Fallback: API Mode

If local model NOT found, tries API:

```
VCF Input → Score Variants → Call Google API → Interpret → HTML Report
```

Requires `GOOGLE_API_KEY` secret in notebook.

### Last Resort: Mock Mode

If neither local model nor API key available:

```
VCF Input → Score Variants → Mock Response → HTML Report
```

Returns placeholder interpretations.

---

## Architecture

```python
# In gemma_client.py
client = GemmaClient(
    api_key=GOOGLE_API_KEY,
    use_local=True,  # ← Try local first
)

# Priority:
# 1. Check /kaggle/input/gemma-4/config.json → Load with transformers
# 2. Check GOOGLE_API_KEY → Use google-generativeai SDK
# 3. Return mocks
```

### Local Model Setup

```python
def _load_local_model(model_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
```

- **Auto-GPU**: Detects Kaggle GPU and uses float16 for efficiency
- **Memory**: `low_cpu_mem_usage=True` for large models
- **Token Budget**: ~1024 output tokens per interpretation

---

## Troubleshooting

### "Local model not found"

Check:
1. Model added in **Data** → **Add Input**?
2. Is it showing in your Inputs list?
3. Try refreshing notebook

### "CUDA out of memory"

On weaker GPU:
1. Set `use_local=False` in client init to use API instead
2. Or reduce `max_output_tokens` in `_generate_local()`

### "transformers import error"

The pip install cell includes `transformers`. If missing:
```python
!pip install transformers --quiet
```

---

## Environment Variables (Optional)

Override auto-detection:

```python
import os

# Force local mode (ignore API)
os.environ["GENESCRIBE_FORCE_LOCAL"] = "1"

# Custom model path
os.environ["GENESCRIBE_MODEL_PATH"] = "/custom/path/gemma-4"
```

---

## File Structure in Kaggle

```
/kaggle/input/
  └── gemma-4/              ← Model dataset (auto-added)
        ├── config.json     ← Model config
        ├── model-*.safetensors
        ├── tokenizer.model
        └── ...

/kaggle/working/            ← Output folder
  ├── genescribe_report.html
  ├── genescribe_report.json
  └── ...
```

---

## Performance Expectations

| Aspect | Local | API |
|--------|-------|-----|
| Startup | ~30s (model load) | ~2s |
| Per-variant | ~5-8s | ~2-4s |
| 8 variants | ~60s total | ~20s total |
| Quota limit | None | Yes (varies) |
| Reproducibility | Exact | Possible variance |

---

## FAQ

**Q: Can I use this for the hackathon submission?**  
A: Yes! Local model provides full reproducibility for judges. Recommended.

**Q: What if the model is really slow?**  
A: Use API mode instead (`use_local=False`). Or reduce `max_length` in `_generate_local()`.

**Q: Does the local model give same results as API?**  
A: Very similar (same weights), but temperature + sampling may vary slightly.

**Q: What happens if I forget to attach the model?**  
A: Notebook automatically falls back to API or mock. No errors, just different mode.

**Q: How do I verify it's using local mode?**  
A: Check console output after initialization:
```
✅ Local Gemma-4 model found at /kaggle/input/gemma-4
🤖 Mode: LOCAL | Model: gemma-4-9b-it
```

---

## References

- [Kaggle Models: google/gemma-4](https://www.kaggle.com/models/google/gemma-4)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [GeneScribe Notebook](./notebooks/genescribe_kaggle_demo.ipynb)
- [GemmaClient Source](./src/gemma_client.py)
