"""Check the exact lab environment on Kaggle, without loading model weights."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    from scripts.prepare_data import _find_root, prepare

    assert Path("/kaggle").exists(), "This preflight must run on Kaggle."
    assert torch.cuda.is_available(), "Kaggle did not allocate a GPU."
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
    meta = prepare(_find_root("/kaggle/input"), "data")
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    config = AutoConfig.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration(config)
    model.tie_weights()
    names = [name for name, _ in model.named_modules()]
    architecture = {"total": sum(p.numel() for p in model.parameters())}
    for name in ("visual", "visual.merger", "visual.deepstack_merger_list", "language_model"):
        architecture[name] = sum(p.numel() for n, p in model.named_parameters() if n.startswith("model." + name + "."))
    result = {"gpu": torch.cuda.get_device_name(0), "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 2**30, "packages": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "transformers", "peft", "trl", "bitsandbytes", "accelerate")}, "dataset": meta, "architecture": architecture, "module_names": names, "processor_class": type(processor).__name__}
    Path("report/evidence").mkdir(parents=True, exist_ok=True)
    Path("report/evidence/preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("module_names", "dataset")}, indent=2))
    print("PREFLIGHT PASS", flush=True)


if __name__ == "__main__":
    main()
