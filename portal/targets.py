"""Language-side (target) features for PORTAL.

PORTAL matches vision features to the LLM's input embedding table by default
(layer=0): the vectors the LLM looks up for each text token before its first
transformer block. The PORTAL-ctx-L variant instead targets the hidden states
after block L, collected by passing a list of English words and short phrases
through the LLM (paper Sec. 4.1, 5.4). Either way pca_stats turns the result
into target statistics.
"""

from pathlib import Path

import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

__all__ = ["llm_embedding_table", "llm_hidden_states", "load_prompts"]


def _is_llava(model_name: str) -> bool:
    return "llava" in model_name.lower()


def _load_llm(model_name: str, dtype: torch.dtype):
    """Load an LLM as a LLaVA model or a plain CausalLM, depending on its name."""
    if _is_llava(model_name):
        from transformers import LlavaForConditionalGeneration

        cls = LlavaForConditionalGeneration
    else:
        from transformers import AutoModelForCausalLM

        cls = AutoModelForCausalLM
    return cls.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)


def _text_decoder(model):
    """The text transformer stack, for LLaVA (nested under language_model) and
    plain CausalLM alike; its forward exposes output_hidden_states."""
    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


def llm_embedding_table(llm_model: str) -> Tensor:
    """Load the LLM input-embedding matrix (the layer=0 target for PORTAL).

    Returns (vocab_size, d_model) on CPU in float32. get_input_embeddings keeps
    this working for plain CausalLM and LLaVA models across transformers layouts.
    """
    logger.info("loading LLM embedding table: {}", llm_model)
    model = _load_llm(llm_model, torch.float32)
    emb = model.get_input_embeddings().weight.detach().clone().float()
    del model
    logger.info("embedding table: {}", tuple(emb.shape))
    return emb


def llm_hidden_states(
    llm_model: str,
    prompts: list[str],
    layer: int,
    batch_size: int = 16,
    max_length: int = 64,
    device: str | None = None,
) -> Tensor:
    """Collect token hidden states after transformer block `layer`.

    Used by PORTAL-ctx-L (layer > 0); layer=0 is the input embeddings, for which
    llm_embedding_table is preferred. `layer` indexes output_hidden_states
    (0 = embeddings, L = after block L). `prompts` are words/short phrases run
    through the LLM; fp16 on CUDA, fp32 on CPU. Returns (num_tokens, d_model) on
    CPU in float32, with padding removed via the attention mask.
    """
    from transformers import AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    on_cuda = str(device).startswith("cuda")
    dtype = torch.float16 if on_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_llm(llm_model, dtype)
    decoder = _text_decoder(model).to(device).eval()

    logger.info(
        "collecting hidden states at layer {} over {} prompts", layer, len(prompts)
    )
    collected: list[Tensor] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"L={layer} states"):
        enc = tokenizer(
            prompts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            out = decoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
        mask = enc["attention_mask"].bool()
        collected.append(out.hidden_states[layer][mask].float().cpu())

    del model
    if on_cuda:
        torch.cuda.empty_cache()

    hidden = torch.cat(collected, dim=0)
    logger.info("hidden states: {}", tuple(hidden.shape))
    return hidden


def load_prompts(path: str | Path) -> list[str]:
    """Load newline-delimited prompts from a file, skipping blank and '#' lines."""
    lines = (ln.strip() for ln in Path(path).read_text().splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]
