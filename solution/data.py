"""Grounding targets and response-only supervised examples."""
import json
from functools import lru_cache
from pathlib import Path
from PIL import Image

CLASSES = ("pothole", "crack", "manhole")


def prompt(coord="norm1000"):
    scale = "integer coordinates normalized to 0-1000 relative to image width and height" if coord == "norm1000" else "integer pixel coordinates in the original 640x360 image"
    return ("Locate every pothole, crack, and manhole visible on the road in this image. "
            "Use only these labels: pothole, crack, manhole. "
            f"Return bounding boxes as [x1,y1,x2,y2] using {scale}. "
            'Return ONLY a JSON list of objects with keys "label" and "bbox_2d". '
            "Order objects from top to bottom, then left to right. Return [] if none are present.")


def target_objects(entry, coord="norm1000"):
    sx, sy = (1000, 1000) if coord == "norm1000" else (entry["width"], entry["height"])
    objects = []
    for obj in entry["objects"]:
        xc, yc, w, h = obj["yolo"]
        bb = [round(max(0, min(1, value)) * scale) for value, scale in zip((xc-w/2, yc-h/2, xc+w/2, yc+h/2), (sx, sy, sx, sy))]
        if bb[2] <= bb[0] or bb[3] <= bb[1]:
            raise ValueError(f"Box collapsed during rounding: {entry['image_id']}: {bb}")
        objects.append({"label": obj["label"], "bbox_2d": bb})
    return sorted(objects, key=lambda o: (o["bbox_2d"][1], o["bbox_2d"][0], o["label"]))


def answer(entry, coord="norm1000"):
    return json.dumps(target_objects(entry, coord), separators=(",", ":"))


def user_message(image, coord="norm1000"):
    return {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt(coord)}]}


class SupervisedDataset:
    def __init__(self, entries, processor, data_dir="data", coord="norm1000"):
        self.entries, self.processor = entries, processor
        self.data_dir, self.coord = Path(data_dir), coord

    def __len__(self):
        return len(self.entries)

    @lru_cache(maxsize=64)
    def __getitem__(self, index):
        import torch
        entry = self.entries[index]
        with Image.open(self.data_dir / entry["image_path"]) as image:
            user = user_message(image.convert("RGB"), self.coord)
            options = {"tokenize": True, "return_dict": True, "return_tensors": "pt"}
            prefix = self.processor.apply_chat_template([user], add_generation_prompt=True, **options)
            full = self.processor.apply_chat_template([user, {"role": "assistant", "content": answer(entry, self.coord)}], add_generation_prompt=False, **options)
        length = prefix["input_ids"].shape[1]
        if not torch.equal(full["input_ids"][:, :length], prefix["input_ids"]):
            raise ValueError("The generation prompt is not a prefix of the training conversation.")
        labels = full["input_ids"].clone()
        labels[:, :length] = -100
        full["labels"] = labels
        return full


def collate(examples, pad_token_id):
    import torch
    result = {}
    length = max(e["input_ids"].shape[1] for e in examples)
    sequence_keys = ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids", "labels")
    for key in examples[0]:
        if key in sequence_keys:
            value = pad_token_id if key == "input_ids" else (-100 if key == "labels" else 0)
            result[key] = torch.cat([torch.nn.functional.pad(e[key], (0, length-e[key].shape[1]), value=value) for e in examples], dim=0)
        elif key in ("pixel_values", "image_grid_thw"):
            result[key] = torch.cat([e[key] for e in examples], dim=0)
        else:
            raise ValueError(f"Review unsupported processor key before training: {key}")
    return result
