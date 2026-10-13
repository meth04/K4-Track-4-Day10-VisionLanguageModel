"""Fill the provided notebook with readable calls to the student implementation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/"lab_day10.ipynb"
    notebook=json.loads(path.read_text(encoding="utf-8"))
    replacements={
        0:"# Lab Day 10 — Nguyễn Văn Thân\nTriển khai QLoRA Qwen3-VL-4B trên Kaggle. Mã chi tiết ở `solution/`; cách điều khiển từ máy cá nhân ở `KAGGLE.md`. Notebook này để đọc và chạy từng phần; chiến dịch tự động dùng các notebook riêng cho mỗi setup. Các cell huấn luyện chỉ chạy trên Kaggle.\n",
        1:"## 0. Môi trường (Kaggle T4)\nChỉ dùng một T4; GPU còn lại không được dùng. Chạy từ thư mục repo đã được đưa lên Kaggle.\n",
        2:"import os\nos.environ['CUDA_VISIBLE_DEVICES'] = '0'\nos.environ['HF_HOME'] = '/tmp/lab-huggingface'\nos.environ['TOKENIZERS_PARALLELISM'] = 'false'\nfrom pathlib import Path\nassert Path('/kaggle/input').is_dir(), 'Chạy notebook này trên Kaggle.'\n!nvidia-smi -L\n!pip -q install -r requirements-colab.txt pytest datasets\n",
        3:"# Dataset công khai được gắn qua dataset_sources. Không cần đặt API key trong notebook.\nimport torch\nassert torch.cuda.is_available()\nprint(torch.cuda.get_device_name(0))\n",
        5:"from scripts.prepare_data import _find_root, prepare\nmeta = prepare(_find_root('/kaggle/input'), 'data')\nprint(meta['dataset_version'], meta['counts'])\n",
        6:"from scripts.labcommon import read_jsonl\nfrom solution.data import target_objects\nfrom PIL import Image, ImageDraw\ntrain = read_jsonl('data/splits/train.jsonl')\nval = read_jsonl('data/splits/val.jsonl')\ntest = read_jsonl('data/splits/test.jsonl')\nsessions = [{e['session'] for e in rows} for rows in (train, val, test)]\nassert all(not sessions[i].intersection(sessions[j]) for i in range(3) for j in range(i))\nfor entry in train[:6]:\n    image = Image.open(Path('data') / entry['image_path']).convert('RGB')\n    drawing = ImageDraw.Draw(image)\n    for box in target_objects(entry, 'pixel'):\n        drawing.rectangle(box['bbox_2d'], outline='lime', width=2)\n    display(image)\n",
        8:"from solution.data import answer, prompt, target_objects\nentry = train[0]\nprint(prompt())\nprint(answer(entry))\nprint('Pixel:', target_objects(entry, 'pixel'))\n",
        10:"# E0: chỉ chạy cell này khi cần thực hiện baseline lần đầu.\n# Chiến dịch CLI tự chạy E0 trong kernel riêng và không chấm test lặp lại.\n!python -u -m solution.run --phase e0\n",
        11:"import json\nprint(json.loads(Path('report/evidence/e0.json').read_text())['micro'])\n",
        13:"from solution.data import SupervisedDataset, collate\nfrom solution.engine import diagnose\n# diagnose(dataset, processor, model.config, output_path) kiểm tra 220 token ảnh\n# và decode các labels != -100 thành JSON đáp án + token kết thúc.\n# SupervisedDataset kiểm tra tiền tố và mask loss; collate pad labels=-100.\nimport inspect\nprint(inspect.getsource(SupervisedDataset))\nprint(inspect.getsource(collate))\n",
        15:"from solution.modeling import apply_setup, restore_checkpoint, save_checkpoint\nprint(inspect.getsource(apply_setup))\n# E1/E2: 252 module; E3: 348 module. Projector trainable giữ float32.\n",
        17:"from solution.engine import train_sft\nprint(inspect.getsource(train_sft))\n# AdamW, warmup 5%, cosine, fp16 AMP, gradient checkpointing, clipping.\n# Lưu riêng projector và chọn checkpoint theo validation; test không chọn model.\n",
        19:"# Chọn một setup mỗi phiên GPU để giải phóng tài nguyên giữa các lượt chạy.\nSETUP = 'e1'  # lần lượt e1, e2, e3\n!python -u -m solution.run --phase {SETUP}\n",
        21:"# E4/E5 cần gắn output kernels E1, E2, E3 vào /kaggle/input.\n# CLI tự gắn chúng, chọn baseline theo validation và giữ nguyên các biến còn lại.\n!python -m pytest -q solution/test_solution.py\n# Chạy từng phase trong kernel riêng:\n# !python -u -m solution.run --phase e4\n# !python -u -m solution.run --phase e5\n# !python -u -m solution.run --phase report\n",
    }
    for index,source in replacements.items():
        notebook['cells'][index]['source']=source.splitlines(keepends=True)
        if notebook['cells'][index]['cell_type']=='code':
            notebook['cells'][index]['outputs']=[]
            notebook['cells'][index]['execution_count']=None
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")


if __name__=='__main__':
    main()
