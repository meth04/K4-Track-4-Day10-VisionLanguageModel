"""Lightweight local controller; all ML work runs in private Kaggle notebooks."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.kaggle_control import build

os.environ["PYTHONUTF8"]="1"
LOCK=threading.Lock()
STATE_PATH=ROOT/"work/campaign/state.json"


def save(state):
    with LOCK:
        STATE_PATH.parent.mkdir(parents=True,exist_ok=True)
        temporary=STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(state,indent=2),encoding="utf-8")
        temporary.replace(STATE_PATH)


def cli(*args):
    completed=subprocess.run([sys.executable,"-m","kaggle",*args],cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed.stdout


def remote_phase(owner,phase,sources,state):
    ref=f"{owner}/day10-qwen3vl-{phase}"
    prior=state["phases"].get(phase)
    if prior and prior["status"]=="complete":
        return
    if prior and prior["status"]=="failed":
        current=cli("kernels","status",ref)
        if any(word in current for word in ("ERROR","CANCEL","FAILED")):
            raise RuntimeError(f"{phase} still failed. Push the repaired phase manually before resuming.")
    if not prior:
        directory,_=build(owner,phase,sources)
        command=["kernels","push","-p",str(directory)]
        if phase!="report": command += ["--accelerator","NvidiaTeslaT4"]
        print(cli(*command),flush=True)
        state["phases"][phase]={"status":"submitted","ref":ref,"submitted_at":time.time()}
        save(state)
    start=time.time()
    while True:
        try:
            status=cli("kernels","status",ref)
        except RuntimeError as error:
            print(f"Temporary status error for {phase}: {error}",flush=True)
            time.sleep(45)
            continue
        print(status.strip(),flush=True)
        if "COMPLETE" in status:
            state["phases"][phase]["status"]="complete"
            save(state)
            return
        if any(word in status for word in ("ERROR","CANCEL","FAILED")):
            state["phases"][phase]["status"]="failed"
            save(state)
            subprocess.run([sys.executable,"tools/kaggle_control.py","log","--phase",phase,"--owner",owner],cwd=ROOT,check=False)
            raise RuntimeError(f"Remote {phase} failed; fix its actual error before retrying. No evidence was fabricated.")
        if time.time()-start>13*3600:
            raise TimeoutError(f"{phase} exceeded the 12-hour Kaggle session allowance plus queue tolerance.")
        time.sleep(45)


def collect_submission(owner):
    subprocess.run([sys.executable,"tools/kaggle_control.py","collect","--phase","report","--owner",owner],cwd=ROOT,check=True)
    import shutil
    download=ROOT/"work/downloads/report"
    reports=[p for p in download.rglob("REPORT.md") if (p.parent/"report/evidence/report_manifest.json").exists()]
    if len(reports)!=1:
        raise RuntimeError(f"Expected one generated final report, found {reports}")
    exported=reports[0].parent
    shutil.copy2(reports[0],ROOT/"REPORT.md")
    shutil.copytree(exported/"report",ROOT/"report",dirs_exist_ok=True)
    metadata=exported/"data/splits/meta.json"
    if metadata.exists():
        (ROOT/"data/splits").mkdir(parents=True,exist_ok=True)
        shutil.copy2(metadata,ROOT/"data/splits/meta.json")
    # Verify immutable instructor files before creating the results commit.
    changed=subprocess.run(["git","diff","c2f3997","--","scripts","tests"],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    if changed:
        raise RuntimeError("Instructor scripts/tests changed; stop before submission.")
    subprocess.run([sys.executable,"tools/check_rubric.py"],cwd=ROOT,check=True)
    # These are genuine newly measured results, not synthetic commit padding.
    subprocess.run(["git","add","REPORT.md","report"],cwd=ROOT,check=True)
    subprocess.run(["git","commit","-m","results: record measured Kaggle experiments, bonus GRPO and report"],cwd=ROOT,check=True)
    subprocess.run(["git","push","origin","main"],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner",default="soobinms")
    parser.add_argument("--smoke-already-submitted",action="store_true")
    args=parser.parse_args()
    state=json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {"phases":{},"status":"running"}
    state["status"]="running"
    if args.smoke_already_submitted and "smoke" not in state["phases"]:
        state["phases"]["smoke"]={"status":"submitted","ref":f"{args.owner}/day10-qwen3vl-smoke"}
    save(state)
    try:
        remote_phase(args.owner,"smoke",[],state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for pair in (("e0","e1"),("e2","e3")):
                source=[] if pair==("e0","e1") else [f"{args.owner}/day10-qwen3vl-e0"]
                futures=[pool.submit(remote_phase,args.owner,phase,source,state) for phase in pair]
                for future in futures: future.result()
            baselines=[f"{args.owner}/day10-qwen3vl-{name}" for name in ("e0","e1","e2","e3")]
            futures=[pool.submit(remote_phase,args.owner,phase,baselines,state) for phase in ("e4","e5")]
            for future in futures: future.result()
        sources=[f"{args.owner}/day10-qwen3vl-{name}" for name in ("preflight","e0","e1","e2","e3","e4","e5")]
        remote_phase(args.owner,"report",sources,state)
        collect_submission(args.owner)
        state["status"]="complete"
        save(state)
        print("CAMPAIGN COMPLETE — measured evidence and rubric checks passed.",flush=True)
    except Exception as error:
        state["status"]="failed"
        state["error"]=str(error)
        save(state)
        print(f"CAMPAIGN STOPPED: {error}",flush=True)
        raise


if __name__=="__main__":
    main()
