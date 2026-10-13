"""NF4 model, precisely scoped adapters, and separate projector persistence."""
import hashlib
import json
from pathlib import Path
import re

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LLM_TARGET = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGET = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"


def is_projector(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def set_cache(model, enabled):
    text_config = getattr(model.config, "text_config", model.config)
    previous = getattr(text_config, "use_cache", True)
    text_config.use_cache = enabled
    model.config.use_cache = enabled
    return previous


def persist_base_snapshot(model):
    """Copy the original base checkpoint inside Kaggle, never to the PC."""
    from huggingface_hub import snapshot_download
    import shutil
    destination = Path("/kaggle/working/base_model")
    if (destination / "revision.txt").exists():
        return
    mounted = sorted(Path("/kaggle/input").rglob("base_model/revision.txt"))
    if mounted:
        source = mounted[0].parent
    else:
        source = snapshot_download(MODEL_ID, revision=model.config.lab_model_revision, local_files_only=True)
    shutil.copytree(source, destination, symlinks=False, dirs_exist_ok=True)
    (destination / "revision.txt").write_text(model.config.lab_model_revision, encoding="utf-8")


def load_base():
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("Training/inference requires the Kaggle GPU. Never run on the local machine.")
    cached = sorted(Path("/kaggle/input").rglob("base_model/revision.txt"))
    cached += [p for p in (Path("/kaggle/working/base_model/revision.txt"),) if p.exists()]
    if cached:
        source = str(cached[0].parent)
        revision = cached[0].read_text(encoding="utf-8").strip()
    else:
        source = MODEL_ID
        revision = HfApi().model_info(MODEL_ID).sha
    # Transformers 5.19 matches prefixes against fully qualified names; "visual"
    # alone no longer excludes model.visual.blocks.*. Include the full prefix.
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["model.visual", "visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(source, revision=revision, quantization_config=quantization, dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
    model.config.lab_model_revision = revision
    processor = AutoProcessor.from_pretrained(source, revision=revision)
    processor.tokenizer.padding_side = "left"
    for name, module in model.named_modules():
        if name.startswith("model.visual.") and type(module).__name__ == "Linear4bit":
            raise RuntimeError(f"Vision must not be quantized: {name}")
    return model, processor


def apply_setup(model, setup, rank=16):
    import torch
    from peft import LoraConfig, get_peft_model
    model.requires_grad_(False)
    pattern = LLM_TARGET if setup != "e3" else f"(?:{LLM_TARGET}|{VIT_TARGET})"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
    expected = 348 if setup == "e3" else 252
    if len(matched) != expected:
        raise RuntimeError(f"Expected {expected} adapter targets, found {len(matched)}: {matched}")
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2*rank, lora_dropout=.05, target_modules=pattern, task_type="CAUSAL_LM"))
    if setup in ("e2", "e3"):
        for name, parameter in model.named_parameters():
            if is_projector(name):
                parameter.data = parameter.data.to(torch.float32)
                parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_cache(model, False)
    info = {"setup": setup, "rank": rank, "matched_module_count": len(matched), "matched_modules": matched, "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), "projector_trainable_params": sum(p.numel() for n,p in model.named_parameters() if is_projector(n) and p.requires_grad), "trainable_groups": {}}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = "projector" if is_projector(name) else ("vit_lora" if "visual.blocks." in name else "llm_lora")
            info["trainable_groups"][group] = info["trainable_groups"].get(group, 0) + parameter.numel()
    if setup in ("e2", "e3") and not 108_000_000 < info["projector_trainable_params"] < 110_000_000:
        raise RuntimeError(f"Projector parameter count differs from lab: {info}")
    print(json.dumps(info), flush=True)
    return model, info


def projector_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if is_projector(name):
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(model, processor, path, info):
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
    extra = {n:p.detach().float().cpu().contiguous() for n,p in model.named_parameters() if is_projector(n) and p.requires_grad}
    if extra:
        save_file(extra, str(path / "projector.safetensors"))
    metadata = dict(info)
    if extra:
        metadata["projector_sha256"] = projector_fingerprint(model)
    (path / "checkpoint.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def restore_checkpoint(base, path, trainable=False):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    path = Path(path)
    model = PeftModel.from_pretrained(base, path, is_trainable=trainable)
    info = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if info.get("setup") in ("e2", "e3"):
        extra = load_file(str(path / "projector.safetensors"))
        parameters = dict(model.named_parameters())
        expected = {n for n in parameters if is_projector(n)}
        if expected != set(extra):
            raise RuntimeError("Saved projector keys do not match model parameters.")
        with torch.no_grad():
            for name, tensor in extra.items():
                parameters[name].data = tensor.to(device=parameters[name].device, dtype=torch.float32)
                parameters[name].requires_grad_(trainable)
        if projector_fingerprint(model) != info["projector_sha256"]:
            raise RuntimeError("Projector round-trip hash differs.")
    return model, info
