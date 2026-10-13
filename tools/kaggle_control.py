"""Package source for Kaggle; never download data/model weights to this computer."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONUTF8"] = "1"
DATASET = "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"


def build(owner, phase, sources=(), options=()):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("scripts", "tests", "configs", "solution", "tools"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(ROOT).as_posix())
        for name in ("requirements-colab.txt", "pytest.ini", "README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md", "lab_day10.ipynb"):
            archive.write(ROOT / name, name)
    payload = base64.b64encode(buffer.getvalue()).decode()
    bootstrap = f'''import base64, io, os, pathlib, shutil, subprocess, sys, zipfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/tmp/lab-huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
root = pathlib.Path("/kaggle/working/lab")
root.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))).extractall(root)
os.chdir(root)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-colab.txt", "pytest", "datasets"], check=True)
try:
    subprocess.run([sys.executable, "-u", "tools/kaggle_preflight.py"] if {phase!r} == "preflight" else [sys.executable, "-u", "-m", "solution.run", "--phase", {phase!r}, *{list(options)!r}], check=True)
finally:
    # These are disposable copies created by prepare_data, not the input dataset.
    copies = root / "data" / "images"
    if copies.is_dir() and copies.resolve().is_relative_to(root.resolve()):
        shutil.rmtree(copies)
'''
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": bootstrap.splitlines(keepends=True)}
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": [cell]}
    slug = f"day10-qwen3vl-{phase}"
    directory = ROOT / "work" / "kernels" / phase
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lab.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    metadata = {"id": f"{owner}/{slug}", "title": f"Day10 Qwen3VL {phase}", "code_file": "lab.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": phase != "report", "enable_internet": True, "machine_shape": "NvidiaTeslaT4" if phase != "report" else "", "dataset_sources": [DATASET], "kernel_sources": list(sources), "competition_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory, metadata["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "push", "status", "collect", "log"))
    parser.add_argument("--owner", default="soobinms")
    parser.add_argument("--phase", default="preflight", choices=("preflight", "smoke", "e0", "e1", "e2", "e3", "e4", "e5", "report"))
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--option", action="append", default=[])
    args = parser.parse_args()
    ref = f"{args.owner}/day10-qwen3vl-{args.phase}"
    if args.action in ("build", "push"):
        directory, ref = build(args.owner, args.phase, args.source, args.option)
        print(directory)
        if args.action == "push":
            command = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(directory)]
            if args.phase != "report":
                command += ["--accelerator", "NvidiaTeslaT4"]
            subprocess.run(command, check=True)
    elif args.action == "status":
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", ref], check=True)
    elif args.action == "log":
        from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
        api = KaggleApi()
        api.authenticate()
        request = ApiListKernelSessionOutputRequest()
        request.user_name = args.owner
        request.kernel_slug = f"day10-qwen3vl-{args.phase}"
        request.page_size = 1
        with api.build_kaggle_client() as client:
            result = client.kernels.kernels_api_client.list_kernel_session_output(request)
        directory = ROOT / "work" / "downloads" / args.phase
        directory.mkdir(parents=True, exist_ok=True)
        content = result.log or ""
        (directory / "remote.log").write_text(content, encoding="utf-8")
        try:
            rows = json.loads(content)
            for row in rows:
                message = str(row.get("data", ""))
                if any(word in message for word in ("Error", "Traceback", "PASS", "generation", "step", "vision tokens")):
                    print(message[-1500:])
        except ValueError:
            print(content[-6000:])
    else:
        # Whitelist only small submission artifacts; exclude images of the dataset,
        # adapters, projector weights, predictions, and model cache.
        destination = ROOT / "work" / "downloads" / args.phase
        destination.mkdir(parents=True, exist_ok=True)
        pattern = r"(^|/)(report/.*\.(json|png|jpg|md)|REPORT\.md|.*\.log|preflight\.json|.*\.log\.json|run\.json|validation\.json|diagnostics\.json|architecture\.json|meta\.json|submission-check\.txt)$"
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", ref, "-p", str(destination), "--file-pattern", pattern, "--force", "--page-size", "200"], check=True)
        print(f"Collected small artifacts to {destination}")


if __name__ == "__main__":
    main()
