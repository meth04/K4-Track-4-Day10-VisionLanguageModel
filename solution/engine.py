"""Training, deterministic generation, and official scoring on Kaggle."""
import gc
import json
import math
from pathlib import Path
import random
import time
from PIL import Image

from scripts.labcommon import read_jsonl
from scripts.score import evaluate
from solution.data import SupervisedDataset, answer, collate, user_message
from solution.modeling import save_checkpoint


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed=42):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def release_memory():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def diagnose(dataset, processor, config, output):
    first = dataset[0]
    ids, labels = first["input_ids"][0], first["labels"][0]
    image_tokens = int((ids == config.image_token_id).sum())
    if image_tokens != 220:
        raise ValueError(f"Expected 220 vision tokens for 640x360; found {image_tokens}.")
    decoded = processor.tokenizer.decode(ids[labels != -100], skip_special_tokens=False)
    expected = answer(dataset.entries[0], dataset.coord)
    if decoded != expected + "<|im_end|>" and decoded.strip() != expected + "<|im_end|>":
        raise ValueError(f"Supervised token decode differs from answer: {decoded!r}")
    write_json(output, {"image_id": dataset.entries[0]["image_id"], "vision_tokens": image_tokens, "masked_tokens": int((labels == -100).sum()), "supervised_tokens": int((labels != -100).sum()), "decoded_supervision": decoded, "answer": expected})


def generate(model, processor, entries, output, coord="norm1000", batch_size=4, max_new_tokens=512, resume=False):
    import torch
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(output) if resume and output.exists() else []
    done = {r["image_id"] for r in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate predictions in existing file.")
    pending = [e for e in entries if e["image_id"] not in done]
    model.eval()
    was_cache = model.config.use_cache
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    start = time.perf_counter()
    with output.open("a" if resume else "w", encoding="utf-8") as stream:
        index = 0
        while index < len(pending):
            batch_entries = pending[index:index+batch_size]
            images = []
            for e in batch_entries:
                with Image.open(Path("data") / e["image_path"]) as image:
                    images.append(image.convert("RGB"))
            conversations = [[user_message(im, coord)] for im in images]
            try:
                inputs = processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True).to("cuda")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    sequences = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=processor.tokenizer.pad_token_id)
                texts = processor.batch_decode(sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for e, raw in zip(batch_entries, texts, strict=True):
                    stream.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
                stream.flush()
                index += len(batch_entries)
                del inputs, sequences
                if index % 40 == 0 or index == len(pending):
                    print(f"generation {output.name}: {index}/{len(pending)} {time.perf_counter()-start:.1f}s", flush=True)
            except torch.OutOfMemoryError:
                release_memory()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size//2)
                print(f"Reducing inference batch size to {batch_size}", flush=True)
    model.config.use_cache = was_cache
    preds = {r["image_id"]:r["raw"] for r in read_jsonl(output)}
    if set(preds) != {e["image_id"] for e in entries}:
        raise ValueError("Prediction coverage differs from split.")
    return evaluate(entries, preds, coord=coord), time.perf_counter()-start


def train_sft(model, processor, entries, val_entries, directory, info, steps=300, coord="norm1000", batch_size=2, grad_accum=4, lr=2e-4, seed=42):
    import torch
    from torch.utils.data import DataLoader
    seed_everything(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = SupervisedDataset(entries, processor, coord=coord)
    diagnose(dataset, processor, model.config, directory / "diagnostics.json")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed), collate_fn=lambda b:collate(b, processor.tokenizer.pad_token_id), num_workers=0)
    iterator = iter(loader)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0)
    warmup = max(1, math.ceil(.05*steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step+1)/warmup if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup))))
    scaler = torch.amp.GradScaler("cuda", init_scale=256)
    torch.cuda.reset_peak_memory_stats()
    elapsed = 0.0
    best_f1 = -1.0
    history = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, steps+1):
        model.train()
        model.config.use_cache = False
        torch.cuda.synchronize()
        started = time.perf_counter()
        step_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k:v.to("cuda") for k,v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}; stop and debug on validation.")
            step_loss += float(loss.detach())
            scaler.scale(loss).backward()
            del batch, loss
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradients at step {step}; no invalid update was saved.")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed += time.perf_counter()-started
        record = {"step": step, "loss": step_loss, "lr": scheduler.get_last_lr()[0], "grad_norm": float(norm), "elapsed_train_seconds": elapsed}
        history.append(record)
        if step % 10 == 0 or step == 1:
            print(json.dumps(record), flush=True)
            write_json(directory / "train.log.json", history)
        if step % 100 == 0 or step == steps:
            meta = {**info, "step": step, "train_minutes": elapsed/60, "peak_vram_gb": torch.cuda.max_memory_allocated()/2**30, "batch_size": batch_size, "grad_accum": grad_accum, "max_steps": steps, "lr": lr, "seed": seed, "coord": coord, "n_train_images": len(entries)}
            save_checkpoint(model, processor, directory / "latest", meta)
            score, duration = generate(model, processor, val_entries, directory / f"val_step{step}.jsonl", coord=coord)
            record["validation_f1"] = score["micro"]["f1"]
            record["validation_seconds"] = duration
            if score["micro"]["f1"] > best_f1:
                best_f1 = score["micro"]["f1"]
                save_checkpoint(model, processor, directory / "best", {**meta, "validation_f1": best_f1})
            print(f"step {step} val F1={score['micro']['f1']:.4f}, best={best_f1:.4f}", flush=True)
            write_json(directory / "train.log.json", history)
    info = {**meta, "best_validation_f1": best_f1, "training_steps_completed": steps}
    write_json(directory / "run.json", info)
    return info
