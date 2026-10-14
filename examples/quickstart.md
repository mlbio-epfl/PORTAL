# Quickstart: PORTAL to LLaVA Stage-II and evaluation

This walkthrough builds a PORTAL projector and plugs it into the standard LLaVA
pipeline. PORTAL replaces only the projector-alignment (Stage-I) step; the
instruction-tuning (Stage-II) and evaluation stacks are the usual LLaVA /
lmms-eval ones.

## 1. Prepare an image pool

PORTAL estimates vision statistics from an *unlabeled* pool of natural images —
no captions, no pairing. The paper uses COCO `val2017` with 1024 images:

```bash
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

Any directory of `.jpg`/`.png` images works. A small single-domain pool is
sufficient (paper Sec. 5.3).

## 2. Build the projector

```bash
python scripts/build_projector.py \
    --image-dir data/coco/val2017 \
    --llm-model llava-hf/llava-1.5-7b-hf \
    --clip-model openai/clip-vit-large-patch14-336 \
    --k 512 \
    --n-images 1024 \
    --out checkpoints/portal_vicuna_clip
```

Output: `checkpoints/portal_vicuna_clip/mm_projector_f.bin`. This uses the same
format as a trained LLaVA projector (a 2-layer MLP state dict), so it drops
straight into the LLaVA loader.

To also copy a reference `config.json` alongside it, pass
`--config-source /path/to/a/trained/llava/projector`.

## 3. Use it as the Stage-II initialization

In your LLaVA training setup, point the projector weights at the PORTAL
checkpoint and run the **unchanged** LLaVA-665k instruction tuning. Because
`W2 = I`, PORTAL is also a valid initialization for a single-linear-layer
projector (see the linear-vs-MLP ablation).

All configurations in the paper share identical Stage-II data and
hyperparameters; only the projector initialization differs (Trained / Random /
PORTAL). This isolates the contribution of the initialization.

## 4. Evaluate

Evaluate the instruction-tuned model with
[`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) on the 16-benchmark
suite grouped into the four Cambrian-1 categories (Knowledge / General / OCR /
Vision-Centric). See the paper's Appendix for the exact benchmark list and
protocol.

## Variants

**Other backbones.** Swap `--llm-model` and `--clip-model`. Dimensions are
inferred automatically; the construction is agnostic to which models produce and
consume the features.

```bash
python scripts/build_projector.py --image-dir data/coco/val2017 \
    --llm-model meta-llama/Llama-3.1-8B \
    --out checkpoints/portal_llama_clip
```

**Random baseline** (Table 1). No image pool needed — dimensions come from the
model configs:

```bash
python scripts/build_projector.py --method random \
    --llm-model llava-hf/llava-1.5-7b-hf \
    --out checkpoints/random_vicuna_clip
```

**PORTAL-ctx-L** (target hidden states after block L):

```bash
python scripts/build_projector.py --image-dir data/coco/val2017 \
    --llm-model meta-llama/Llama-3.1-8B \
    --layer 4 --prompts examples/words.txt \
    --out checkpoints/portal_ctx4_llama
```

Expand `examples/words.txt` to a few thousand words/phrases for a more stable
hidden-state estimate.
