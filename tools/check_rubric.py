"""Stricter audit of actual submission artifacts; does not change the grader."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_submission import run_checks


def audit(root=ROOT):
    root=Path(root)
    checks=run_checks(root)
    def add(ok,name,detail):
        checks.append(("PASS" if ok else "FAIL",name,detail))
    evidence={}
    for name in ("e0","e1","e2","e3","e4_data_size_0.1","e4_data_size_1.0","e5"):
        path=root/"report/evidence"/f"{name}.json"
        if path.exists():
            evidence[name]=json.loads(path.read_text(encoding="utf-8"))
    if "e1" in evidence:
        rate=evidence["e1"]["parse_valid_rate"]
        add(rate>=.95,"E1 valid output >= 95%",f"{rate:.4f}")
    if "e0" in evidence and "e1" in evidence:
        gain=evidence["e1"]["micro"]["f1"]-evidence["e0"]["micro"]["f1"]
        add(gain>=.10,"E1 F1 improvement >= 0.10",f"{gain:.4f}")
    add(all(n in evidence for n in ("e4_data_size_0.1","e4_data_size_1.0")),"E4 both data sizes","10% and 100%")
    images=list((root/"report/figures").glob("example_*.png"))
    add(len(images)>=6,"Annotated examples >= 6",str(len(images)))
    add((root/"report/logs/e5/grpo.log.json").exists() and "e5" in evidence,"E5 bonus logs and official test evidence","")
    add((root/"report/evidence/paired_comparisons.json").exists(),"Paired confidence intervals","")
    return checks


def main():
    checks=audit()
    for status,name,detail in checks:
        print(f"[{status}] {name}"+(f" — {detail}" if detail else ""))
    failures=sum(s=="FAIL" for s,_,_ in checks)
    print(f"{failures} FAIL")
    raise SystemExit(bool(failures))


if __name__=="__main__":
    main()
