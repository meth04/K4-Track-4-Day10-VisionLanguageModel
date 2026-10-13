"""Build the report from actual predictions and official evidence only."""
import json
from pathlib import Path
import random
import shutil
import statistics
from PIL import Image, ImageDraw
import yaml

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, _match, _prf, evaluate, iou_xyxy, parse_output
from solution.engine import write_json
from solution.reward import iou_reward


def locate(relative):
    direct = Path(relative)
    if direct.exists():
        return direct
    candidates = sorted(Path("/kaggle/input").rglob(relative))
    if not candidates:
        raise FileNotFoundError(f"Attach prior kernel output containing {relative}")
    return candidates[0]


def paired_interval(entries, first, second, n_boot=2000, by_session=False):
    rows = []
    for e in entries:
        pair = []
        for pred in (first, second):
            counts, _ = _match(_gt_boxes(e), parse_output(pred[e["image_id"]]) or [])
            pair.append(tuple(sum(v[k] for v in counts.values()) for k in range(3)))
        rows.append(pair)
    groups = {}
    for i,e in enumerate(entries):
        groups.setdefault(e["session"] if by_session else str(i), []).append(i)
    blocks = list(groups.values())
    rng = random.Random(42)
    values = []
    for _ in range(n_boot):
        sample = [i for _ in blocks for i in rng.choice(blocks)]
        f = [_prf(*(sum(rows[i][j][k] for i in sample) for k in range(3)))["f1"] for j in (0,1)]
        values.append(f[1]-f[0])
    values.sort()
    return [values[int(.025*n_boot)], values[int(.975*n_boot)-1]]


def illustrate(entries, predictions, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    for e in entries:
        pred = parse_output(predictions[e["image_id"]])
        counts, _ = _match(_gt_boxes(e), pred or [])
        errors = sum(v[1]+v[2] for v in counts.values())
        cases.append((errors,e,pred or []))
    chosen = []
    used = set()
    for label in CLASSES:
        relevant = [c for c in cases if any(o["label"] == label for o in c[1]["objects"]) and c[1]["image_id"] not in used]
        if relevant:
            c = min(relevant, key=lambda c:(c[0],c[1]["image_id"]))
            chosen.append(c); used.add(c[1]["image_id"])
        relevant = [c for c in cases if any(o["label"] == label for o in c[1]["objects"]) and c[1]["image_id"] not in used]
        if relevant:
            c = max(relevant, key=lambda c:(c[0],c[1]["image_id"]))
            chosen.append(c); used.add(c[1]["image_id"])
    descriptions = []
    for index,(errors,e,pred) in enumerate(chosen,1):
        with Image.open(Path("data")/e["image_path"]) as original:
            canvas = Image.new("RGB", (e["width"], e["height"]+50), "white")
            canvas.paste(original.convert("RGB"),(0,50))
        drawing = ImageDraw.Draw(canvas)
        drawing.text((8,5), f"{e['image_id']} | errors: {errors}", fill="black")
        drawing.text((8,23), "GREEN: ground truth     RED: prediction", fill="black")
        for boxes,color in ((_gt_boxes(e),"lime"),(pred,"red")):
            for box in boxes:
                x1,y1,x2,y2=box["bbox"]
                pixel=(x1*e["width"]/1000,y1*e["height"]/1000+50,x2*e["width"]/1000,y2*e["height"]/1000+50)
                drawing.rectangle(pixel, outline=color,width=2)
                drawing.text((max(0,pixel[0]),max(50,pixel[1]-12)),box["label"],fill=color,stroke_width=1,stroke_fill="black")
        name=f"example_{index}.png"
        canvas.save(directory/name)
        descriptions.append({"file": f"report/figures/{name}", "image_id":e["image_id"], "errors":errors, "ground_truth":_gt_boxes(e), "predictions":pred})
    write_json("report/evidence/examples.json",descriptions)
    return descriptions


def fmt(value):
    return f"{value:.4f}"


def error_breakdown(entries, predictions):
    counts={"JSON hỏng":0,"bỏ sót GT":0,"box trùng":0,"sai lớp với IoU ≥ 0.5":0,"sai định vị, IoU cùng lớp 0.1–0.5":0,"FP có IoU cùng lớp < 0.1":0}
    areas={c:{"matched":[],"missed":[]} for c in CLASSES}
    for entry in entries:
        parsed=parse_output(predictions[entry["image_id"]])
        if parsed is None: counts["JSON hỏng"]+=1
        pred=parsed or []
        gt=_gt_boxes(entry)
        pairs=sorted(((iou_xyxy(g['bbox'],p['bbox']),i,j) for i,g in enumerate(gt) for j,p in enumerate(pred) if g['label']==p['label']),reverse=True)
        used_g,used_p=set(),set()
        for overlap,i,j in pairs:
            if overlap<.5: break
            if i not in used_g and j not in used_p:
                used_g.add(i); used_p.add(j)
        counts['bỏ sót GT']+=len(gt)-len(used_g)
        for i,g in enumerate(gt):
            bb=g['bbox']
            areas[g['label']]['matched' if i in used_g else 'missed'].append((bb[2]-bb[0])*(bb[3]-bb[1])/10000)
        for j,p in enumerate(pred):
            if j in used_p: continue
            same=max((iou_xyxy(g['bbox'],p['bbox']) for g in gt if g['label']==p['label']),default=0)
            other=max((iou_xyxy(g['bbox'],p['bbox']) for g in gt if g['label']!=p['label']),default=0)
            category='box trùng' if same>=.5 else ('sai lớp với IoU ≥ 0.5' if other>=.5 else ('sai định vị, IoU cùng lớp 0.1–0.5' if same>=.1 else 'FP có IoU cùng lớp < 0.1'))
            counts[category]+=1
    medians={c:{kind:statistics.median(values) if values else None for kind,values in groups.items()} for c,groups in areas.items()}
    return {"counts":counts,"median_area_percent_by_match":medians,"note":"FN and FP categories can describe the same wrong prediction; counts are not disjoint events."}


def build_report():
    required=("e0","e1","e2","e3","e4_data_size_0.1","e4_data_size_1.0","e5")
    evidence={}
    for name in required:
        source=locate(f"report/evidence/{name}.json")
        evidence[name]=json.loads(source.read_text(encoding="utf-8"))
        if evidence[name].get("missing_predictions") != 0 or evidence[name].get("split") != "test":
            raise ValueError(f"Incomplete official evidence: {name}")
        write_json(f"report/evidence/{name}.json",evidence[name])
    meta=json.loads(Path("data/splits/meta.json").read_text(encoding="utf-8"))
    if {v["dataset_version"] for v in evidence.values()} != {meta["dataset_version"]}:
        raise ValueError("Different dataset versions in report sources.")
    preflight=json.loads(locate("report/evidence/preflight.json").read_text(encoding="utf-8"))
    write_json("report/evidence/preflight.json",preflight)
    student=yaml.safe_load(Path("configs/student.yaml").read_text(encoding="utf-8"))
    test=read_jsonl("data/splits/test.jsonl")
    predictions={name:{r["image_id"]:r["raw"] for r in read_jsonl(locate(f"work/{name}.jsonl"))} for name in required if name != "e4_data_size_1.0"}
    baseline=evidence["e4_data_size_1.0"]["meta"]["base_setup"]
    predictions["e4_data_size_1.0"]=predictions[baseline]
    rows=[]
    setups={"e0":"Zero-shot","e1":"LoRA LLM","e2":"LLM + projector","e3":"LLM + projector + LoRA ViT","e4_data_size_0.1":f"{baseline.upper()}, 10% train","e4_data_size_1.0":f"{baseline.upper()}, 100% train","e5":"GRPO from best validation SFT"}
    for name,value in evidence.items():
        info=value["meta"]
        lo,hi=value["f1_ci95"]
        memory=f"{info['peak_vram_gb']:.2f}" if 'peak_vram_gb' in info else 'không đo'
        if name=='e0' and Path('report/evidence/e0_resource_probe.json').exists():
            resource=json.loads(Path('report/evidence/e0_resource_probe.json').read_text(encoding='utf-8'))
            memory=f"{resource['peak_vram_gb']:.2f}†"
        rows.append(f"| {name.upper()} | {setups[name]} | {info.get('trainable_params',0):,.0f} | {memory} | {info.get('train_minutes',0):.2f} | {info.get('test_inference_minutes',0):.2f} | {fmt(value['micro']['f1'])} | [{fmt(lo)}, {fmt(hi)}] | {fmt(value['fp_per_image'])} | " + " | ".join(fmt(value["per_class"][c]["f1"]) for c in CLASSES)+" |")
        if name != "e0":
            for filename in ("run.json","train.log.json","grpo.log.json","diagnostics.json","validation.json"):
                try: source=locate(f"artifacts/{name}/{filename}")
                except FileNotFoundError: continue
                target=Path("report/logs")/name/filename
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
    wrong=evaluate(test,predictions[baseline],coord="pixel")
    write_json("report/evidence/wrong_coordinate_scale.json",wrong)
    comparisons={}
    for first,second in (("e0","e1"),("e1","e2"),("e2","e3"),("e4_data_size_0.1","e4_data_size_1.0"),(baseline,"e5")):
        comparisons[f"{second}-{first}"]={"delta_f1":evidence[second]["micro"]["f1"]-evidence[first]["micro"]["f1"], "paired_image_ci95":paired_interval(test,predictions[first],predictions[second]), "paired_session_ci95":paired_interval(test,predictions[first],predictions[second],by_session=True)}
    write_json("report/evidence/paired_comparisons.json",comparisons)
    comparison_lines=[]
    for key,comparison in comparisons.items():
        lo,hi=comparison["paired_session_ci95"]
        conclusion="CI chứa 0: chưa đủ bằng chứng về chênh lệch" if lo<=0<=hi else "CI không chứa 0: có bằng chứng về chênh lệch trên tập này"
        comparison_lines.append(f"- {key}: ΔF1={fmt(comparison['delta_f1'])}; CI95 bootstrap theo phiên=[{fmt(lo)}, {fmt(hi)}]. {conclusion}.")
    examples=illustrate(test,predictions[baseline],"report/figures")
    weakest=min(CLASSES,key=lambda c:evidence[baseline]["per_class"][c]["f1"])
    areas={c:[] for c in CLASSES}
    for entry in read_jsonl("data/splits/train.jsonl"):
        for obj in entry["objects"]:
            areas[obj["label"]].append(obj["yolo"][2]*obj["yolo"][3]*100)
    medians={c:statistics.median(areas[c]) for c in CLASSES}
    largest=max(("e1","e2","e3"),key=lambda n:evidence[n]["meta"]["peak_vram_gb"])
    output_samples=list(predictions["e0"].items())[:5]
    write_json("report/evidence/e0_raw_samples.json",[{"image_id":k,"raw":v} for k,v in output_samples])
    hacks={}
    for name in (baseline,"e5"):
        parsed=[parse_output(predictions[name][e["image_id"]]) for e in test]
        hacks[name]={"empty_rate":sum(p==[] for p in parsed)/len(test), "invalid_rate":sum(p is None for p in parsed)/len(test), "giant_box_rate":sum(any((b['bbox'][2]-b['bbox'][0])*(b['bbox'][3]-b['bbox'][1])>=500000 for b in (p or [])) for p in parsed)/len(test), "mean_iou_reward":statistics.mean(iou_reward(predictions[name][e["image_id"]],_gt_boxes(e)) for e in test)}
    write_json("report/evidence/reward_hacking.json",hacks)
    train_losses={}
    for name in ("e1","e2","e3"):
        history=json.loads(locate(f"artifacts/{name}/train.log.json").read_text(encoding="utf-8"))
        train_losses[name]={"first":history[0]["loss"],"last":history[-1]["loss"],"steps":len(history)}
    counts="\n".join(f"| {s} | {meta['counts'][s]['images']} | " + " | ".join(str(meta['counts'][s]['boxes'][c]) for c in CLASSES)+" |" for s in ("train","val","test"))
    images="\n\n".join(f"![GT xanh, dự đoán đỏ: {e['image_id']}]({e['file']})\n\nẢnh `{e['image_id']}`: {e['errors']} lỗi FP+FN tại IoU≥0,5." for e in examples)
    arch=preflight["architecture"]
    gain=evidence["e1"]["micro"]["f1"]-evidence["e0"]["micro"]["f1"]
    e0stats=evidence["e0"]["micro"]
    breakdown={name:error_breakdown(test,predictions[name]) for name in ('e0',baseline)}
    write_json('report/evidence/error_breakdown.json',breakdown)
    top_errors=sorted(breakdown['e0']['counts'].items(),key=lambda item:item[1],reverse=True)[:2]
    report=f'''Họ tên: {student['name']}
GitHub: {student['github_username']}; Kaggle: {student['kaggle_username']}
`dataset_version`: {meta['dataset_version']}

## Môi trường và cách tái lập

Chạy trên Kaggle, GPU {preflight['gpu']} ({preflight['gpu_memory_gb']:.2f} GiB), chỉ một GPU được nhìn thấy qua CUDA_VISIBLE_DEVICES=0. Đề gốc dùng Colab T4; đây là thay đổi nền tảng và được ghi rõ. Không chạy model trên máy cá nhân. Phiên bản thư viện thực tế nằm trong `report/evidence/preflight.json`. Mã của bài ở `solution/`; `scripts/` và `tests/` giữ nguyên. SFT dùng batch=1, tích lũy=8, 300 bước, LR=2e-4, seed=42; đây là phương án giảm VRAM được GUIDE cho phép. Checkpoint chọn theo 40 ảnh validation cố định; báo cáo thêm kết quả trên toàn bộ validation. Mỗi setup suy luận test một lần. Thời gian huấn luyện đo riêng các bước tối ưu, không gồm validation; tổng thời gian chạy sẽ lớn hơn.

## Bảng tổng hợp

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh GiB | Phút huấn luyện | Phút suy luận test | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

† E0 không huấn luyện; VRAM được đo riêng trên bốn ảnh validation với batch=4 và max_new_tokens=512. Đây là đỉnh cấp phát của lượt đo này, không khẳng định là đỉnh của toàn bộ test. Số đo thật và image_id trong `e0_resource_probe.json`. Thời gian suy luận test của E0 lấy từ lượt test gốc; không chạy lại suy luận test.

## Câu 1 — Kiến trúc

Đếm từ model thật trên meta device: tổng {arch['total']:,} tham số; language_model {arch['language_model']:,}; visual {arch['visual']:,}, trong đó merger {arch['visual.merger']:,} và deepstack_merger_list {arch['visual.deepstack_merger_list']:,}. Tổng projector là {arch['visual.merger']+arch['visual.deepstack_merger_list']:,}; phần visual ngoài projector là {arch['visual']-arch['visual.merger']-arch['visual.deepstack_merger_list']:,}. Projector nằm trong visual, không cộng lại lần thứ hai. ViT tạo đặc trưng ảnh, projector đưa đặc trưng sang không gian biểu diễn của LLM, LLM sinh JSON. E1 cập nhật adapter của 252 module LLM. E2 thêm toàn bộ projector. E3 thêm 96 module ViT, tổng 348 module LoRA. Các tham số nền khác đóng băng. Số tham số thực sự huấn luyện được ghi trong bảng và nhật ký.

## Câu 2 — E0

Tỷ lệ parse hợp lệ: {evidence['e0']['parse_valid_rate']:.4f}. Tổng TP={e0stats['tp']}, FP={e0stats['fp']}, FN={e0stats['fn']}. Hai nhóm lỗi đếm được nhiều nhất là {top_errors[0][0]} ({top_errors[0][1]}) và {top_errors[1][0]} ({top_errors[1][1]}). Phân loại FP dựa trên IoU với GT cùng/khác lớp; FN và FP có thể mô tả cùng một dự đoán sai, không xem là các sự kiện độc lập. Tỷ lệ JSON hỏng là {1-evidence['e0']['parse_valid_rate']:.4f}; không mặc định quy lỗi JSON khi parser vẫn chấp nhận đầu ra. Năm output thô thật nằm trong `report/evidence/e0_raw_samples.json`. Ví dụ ảnh `{output_samples[0][0]}`:

```text
{output_samples[0][1]}
```

## Câu 3 — Thang tọa độ

Nhãn YOLO (xc,yc,w,h) chuyển thành (xc−w/2,yc−h/2,xc+w/2,yc+h/2), nhân 1000, làm tròn và kẹp. Quy ngược về pixel bằng x·640/1000 và y·360/1000. Unit test kiểm tra [375,250,625,750] tương ứng [240,90,400,270]. Với output thật của {baseline.upper()}, chấm đúng thang cho F1={fmt(evidence[baseline]['micro']['f1'])}; diễn giải nhầm là pixel cho F1={fmt(wrong['micro']['f1'])}. Đây là chấm lại cùng output, không chạy lại suy luận. Hai hệ số trục khác nhau khiến box sai vị trí/kích thước và IoU giảm; số liệu nằm trong `wrong_coordinate_scale.json`.

## Câu 4 — Mask loss

Mỗi ví dụ được encode hai lần để xác định tiền tố prompt và kiểm tra tiền tố bằng nhau. Labels của prompt và padding đặt -100; chỉ JSON trả lời và token kết thúc tham gia loss. Chẩn đoán thực tế trong `report/logs/e1/diagnostics.json` kiểm tra 220 token ảnh và decode phần được giám sát đúng bằng đáp án. Nếu tính cả prompt/token ảnh, mục tiêu còn yêu cầu dự đoán lại đầu vào, làm phần loss không liên quan chiếm trọng số và làm sai mục tiêu học grounding. Loss đầu/cuối thực tế: {json.dumps(train_losses,ensure_ascii=False)}.

## Câu 5 — Lượng tử hóa và projector

NF4 áp dụng cho phần LLM; visual và lm_head nằm trong skip_modules. Code kiểm tra không có Linear4bit trong visual. Trọng số 4-bit là dạng lưu trữ lượng tử, không phải tham số float có thể cập nhật đầy đủ bằng AdamW như projector. E2/E3 giữ projector trainable ở float32; autocast vẫn dùng fp16 cho tính toán phù hợp. Float32 giúp cập nhật ổn định và tránh vấn đề unscale gradient fp16. Projector được lưu riêng bằng safetensors vì adapter LoRA không chứa các trọng số này. Khi nạp lại, code kiểm tra toàn bộ khóa và SHA256 của projector.

## Câu 6 — Chia theo phiên quay

Frame lân cận từ cùng video gần giống nhau: chia ngẫu nhiên sẽ đưa cảnh gần trùng vào train và test, khiến F1 có thể cao giả. Dùng split cố định của đề và kiểm tra session không giao nhau; không đổi split để cải thiện điểm.

| Split | Ảnh | pothole | crack | manhole |
|---|---|---|---|---|
{counts}

## Câu 7 — So sánh E1/E2/E3

Bảng trên ghi cả lợi ích F1 và chi phí thực tế. E1 cải thiện so với E0 ΔF1={gain:.4f}. Setup chọn cho E4/E5 là {baseline.upper()}, dựa trên validation, không dựa trên test. Setup dùng VRAM nhiều nhất trong E1–E3 là {largest.upper()}, đạt {evidence[largest]['meta']['peak_vram_gb']:.2f} GiB. Huấn luyện projector cần gradient và trạng thái optimizer; LoRA ViT cần truyền gradient qua bộ mã hóa ảnh. Giá trị F1 cao nhất chưa tự nó chứng minh phương án đáng dùng nếu chi phí tăng và CI chênh lệch còn chứa 0.

{chr(10).join(comparison_lines[:3])}

## Câu 8 — Khoảng tin cậy

Test có {len(test)} ảnh thuộc {len({e['session'] for e in test})} phiên. CI95 của từng setup trong bảng do score.py bootstrap theo ảnh. Frame trong cùng phiên có thể tương quan, nên còn tính bootstrap cặp theo phiên cho chênh lệch F1, giữ cặp dự đoán trên cùng ảnh và lấy lại mẫu theo phiên. CI giao nhau của từng model không phải phép kiểm định trực tiếp chênh lệch; dùng CI của ΔF1 và hạn chế kết luận ngoài tập dữ liệu nhỏ này. File `paired_comparisons.json` chứa cả bootstrap theo ảnh và theo phiên. Những CI chứa 0 được nêu rõ ở trên.

## Câu 9 — Phân tích theo lớp

Lớp có F1 thấp nhất của {baseline.upper()} là `{weakest}`: {evidence[baseline]['per_class'][weakest]['f1']:.4f}, TP={evidence[baseline]['per_class'][weakest]['tp']}, FP={evidence[baseline]['per_class'][weakest]['fp']}, FN={evidence[baseline]['per_class'][weakest]['fn']}. Trung vị diện tích box theo phần trăm ảnh trên train: {json.dumps(medians)}. Với lớp này trên test, trung vị diện tích GT được khớp/bị bỏ sót là {json.dumps(breakdown[baseline]['median_area_percent_by_match'][weakest])}. Box nhỏ nhạy với sai lệch tọa độ, lượng mẫu giữa lớp không đều, và bề mặt đường có thể gây nhầm ngữ nghĩa; đây là các khả năng giải thích, chưa phải quan hệ nhân quả được chứng minh. Sáu ảnh dưới chọn trường hợp ít/nhiều lỗi cho từng lớp, có GT xanh và dự đoán đỏ.

{images}

## Câu 10 — E4

Chỉ đổi lượng dữ liệu: 10% và 100% train. Giữ setup {baseline.upper()}, rank=16, alpha=32, seed=42, LR, batch hiệu dụng và 300 bước. Điều kiện 100% tái sử dụng đúng baseline, không huấn luyện hoặc chấm test lần nữa; điều kiện 10% lấy mẫu cố định theo seed. Với cùng số bước, ảnh ở tập 10% được lặp lại nhiều lần hơn: đây là giới hạn cần lưu ý khi diễn giải dữ liệu ít. {comparison_lines[3]}

## Bonus E5 — GRPO

Khởi tạo từ checkpoint SFT chọn bằng validation, dùng reward IoU cùng lớp, ghép tham lam với IoU≥0,5, phạt box thừa qua mẫu số. Các test gồm hoàn hảo, JSON hỏng, sai lớp, box thừa, rỗng và box lớn. GRPO dùng 4 generations, tối đa 128 token, beta=0, LR=5e-6, 60 bước, use_vllm=False. Log thật ở `report/logs/e5/grpo.log.json`. {comparison_lines[4]}

Kiểm tra reward hacking trên output test thật: {json.dumps(hacks,ensure_ascii=False)}. Tỷ lệ trả [] và box có diện tích tương đương ít nhất 50% khung hình được đối chiếu với SFT; không kết luận chỉ từ reward tăng vì model có thể tận dụng thành phần thưởng JSON hợp lệ. Max completion 128 còn có thể cắt đầu ra có nhiều đối tượng; tỷ lệ parse hợp lệ và FP/FN giúp phát hiện giới hạn đó.

## Nhìn lại

Kết quả phụ thuộc dữ liệu dashcam với vật thể nhỏ và split theo phiên có phân bố lớp khác nhau. Tập test nhỏ và số phiên hạn chế giới hạn mức chắc chắn của kết luận. Báo cáo dùng số đo từ các lần chạy thật; mọi output thiếu, sai phiên bản hoặc không phủ đủ test khiến bộ tạo báo cáo dừng. Điểm cuối cùng còn phụ thuộc đánh giá của giảng viên và việc chấp nhận nền tảng Kaggle thay Colab.
'''
    Path("REPORT.md").write_text(report,encoding="utf-8")
    write_json("report/evidence/report_manifest.json",{"student":student,"dataset_version":meta["dataset_version"],"evidence":list(evidence),"figures":len(examples),"selection_split":"val"})
    print("Report generated from measured evidence.",flush=True)
