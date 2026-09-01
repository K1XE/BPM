#!/usr/bin/env python3
"""Harvest CORRECT teacher trajectories on the training set, for a student-SFT arm.

For each training prompt, the teacher generates up to --n-samples completions; each is
graded with the SAME graders the training pipeline uses (math: reward_adapter's killable
math_verify path; code: score_taco on metadata.tests), and the first --keep correct ones
are written as SFT records. Think-tag conventions are normalized to the STUDENT side
(Qwen3.5 prefill-think: the SFT target starts mid-think and contains </think>).

Usage (one free 8-GPU node):

  python examples/bpm/data/gen_teacher_trajectories.py --launch \
      --model-path /path/to/GLM-Z1-9B-0414/hf_checkpoint \
      --data /path/to/mix-20k.jsonl --out /path/to/seqkd_glm_z1_9b.jsonl
  # (a gen-time-think teacher's leading <think> is stripped automatically.)

--data and --out are required. Serving config defaults to tp=8 ep=1 mem-fraction=0.85
and can be overridden with --tp/--ep/--context-length/--mem-fraction.

Resume-safe: re-running with the same --out skips prompts that already have --keep
correct trajectories. Output log doubles as the progress record.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest

# repo root = 3 levels up (examples/bpm/data/<this file>), so the reward adapter is importable
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)


def http_json(url, payload, timeout=3600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_health(port, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def sanitize_output(path):
    """Truncate a trailing partial line left by a hard kill mid-write, so (a) resume
    counting stays exact and (b) the downstream converter's bare json.loads cannot
    crash on it. Only the LAST line is ever at risk (single writer, per-line flush)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        # read the tail region to find the last newline
        back = min(size, 4 * 1024 * 1024)
        f.seek(size - back)
        tail = f.read()
        if tail.endswith(b"\n"):
            # complete final line -- still verify it parses (torn write + newline is
            # possible only if the row itself was cut, which per-line write prevents,
            # but verify cheaply anyway)
            last = tail.rstrip(b"\n").rsplit(b"\n", 1)[-1]
            try:
                json.loads(last.decode("utf-8"))
                return
            except Exception:
                cut = size - len(last) - 1
        else:
            last = tail.rsplit(b"\n", 1)[-1]
            cut = size - len(last)
        f.truncate(cut)
        sys.stderr.write(f"[gen] sanitize: dropped {size - cut}B partial trailing line in {path}\n")


def detect_prefill_think(tok):
    """True if the teacher chat template prefills the think-open tag (GLM-Z1, MiniMax);
    False for gen-time think teachers (Qwen3), whose output BEGINS with <think>."""
    txt = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    return txt.rstrip().endswith("<think>")


def normalize_for_student(text, teacher_prefill_think):
    """Map the teacher's raw completion onto the student's (Qwen3.5 prefill-think)
    convention: target starts mid-think, contains </think>. Returns None for
    malformed/unterminated traces."""
    t = text
    if not teacher_prefill_think:
        # gen-time think teacher: strip exactly one leading <think>\n
        s = t.lstrip()
        if s.startswith("<think>"):
            t = s[len("<think>"):]
            if t.startswith("\n"):
                t = t[1:]
    if "</think>" not in t:
        return None
    return t.strip("\n") + "\n" if t.endswith("\n") else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="teacher HF checkpoint")
    ap.add_argument("--out", required=True, help="output JSONL of teacher trajectories")
    ap.add_argument("--data", required=True, help="prompt set to sample the teacher on")
    ap.add_argument("--domain", choices=["all", "math", "code"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="0 = all prompts (caps prompts scanned)")
    ap.add_argument("--total", type=int, default=0,
                    help="target TOTAL kept records, split evenly math/code (0 = harvest every "
                         "prompt). e.g. --total 5000 -> 2500 math + 2500 code; --total 8000 -> "
                         "4000+4000. With --domain math|code the whole budget goes to that domain. "
                         "Keeps generating until each domain's quota is met (not the first N prompts).")
    ap.add_argument("--n-samples", type=int, default=4, help="max attempts per prompt")
    ap.add_argument("--keep", type=int, default=1, help="correct trajectories kept per prompt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=27648)  # == training/eval max_resp_len (27k)
    ap.add_argument("--concurrency", type=int, default=64, help="concurrent generate requests")
    ap.add_argument("--grade-workers", type=int, default=16, help="concurrent graders (code sandboxes are CPU-heavy)")
    ap.add_argument("--math-timeout", type=float, default=8.0)
    ap.add_argument("--code-timeout", type=float, default=180.0)
    ap.add_argument("--chat-template-kwargs", default='{"enable_thinking": true}')
    # server
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--port", type=int, default=30217)
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--mem-fraction", type=float, default=None)
    args = ap.parse_args()

    args.tp = args.tp if args.tp is not None else 8
    args.ep = args.ep if args.ep is not None else 1
    args.mem_fraction = args.mem_fraction if args.mem_fraction is not None else 0.85

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prefill_think = detect_prefill_think(tok)
    tmpl_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else {}
    print(f"[gen] teacher={args.model_path}  prefill_think={prefill_think}")

    # graders: the SAME code paths training rewards use (no re-implementation).
    import importlib
    ra = importlib.import_module("examples.bpm.reward.reward_adapter")

    def grade(rec, text):
        md = rec.get("metadata") or {}
        rm = (md.get("rm_type") or "").strip()
        if rm == "code_sandbox":
            if not md.get("has_tests", md.get("tests")):
                return False
            r = ra._run_killable_sync(
                ra._score_taco_kw, (text, md.get("tests")), timeout=args.code_timeout, default=False
            )
            return bool(r if isinstance(r, bool) else r.get("acc", False) if isinstance(r, dict) else False)
        acc, _pred = ra._mv_verify_sync(str(rec.get("label", "")), text, timeout=args.math_timeout)
        return bool(acc)

    # data + resume
    records = []
    with open(args.data, encoding="utf-8") as f:
        for uid, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            dom = (r.get("domain") or ("code" if (r.get("metadata") or {}).get("rm_type") == "code_sandbox" else "math"))
            if args.domain != "all" and dom != args.domain:
                continue
            records.append((uid, dom, r))
    if args.limit > 0:
        records = records[: args.limit]

    kept_by_uid: dict[int, int] = {}
    dom_kept = {"math": 0, "code": 0}   # already-kept per domain (for --total budgeting)
    sanitize_output(args.out)
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    rr = json.loads(line)
                except Exception:
                    continue
                kept_by_uid[rr.get("uid")] = kept_by_uid.get(rr.get("uid"), 0) + 1
                if rr.get("domain") in dom_kept:
                    dom_kept[rr["domain"]] += 1
        print(f"[gen] resume: {len(kept_by_uid)} prompts already have kept trajectories "
              f"(math={dom_kept['math']} code={dom_kept['code']})")

    # --total: per-domain quota (even split; --domain math|code sends all to one domain).
    # dom_target=None -> harvest every prompt (original behaviour).
    if args.total > 0:
        if args.domain == "all":
            dom_target = {"math": args.total // 2, "code": args.total - args.total // 2}
        else:
            dom_target = {"math": 0, "code": 0}
            dom_target[args.domain] = args.total
    else:
        dom_target = None

    todo = [(u, d, r) for (u, d, r) in records if kept_by_uid.get(u, 0) < args.keep]
    if dom_target is not None:
        # interleave math/code so both quotas fill in parallel (not all-math-then-code)
        mq = [x for x in todo if x[1] == "math"]
        cq = [x for x in todo if x[1] == "code"]
        todo = [x for pair in zip_longest(mq, cq) for x in pair if x is not None]
        print(f"[gen] prompts: total={len(records)}  todo={len(todo)}  quota "
              f"math={dom_target['math']} code={dom_target['code']}  still-need "
              f"math={max(0, dom_target['math'] - dom_kept['math'])} "
              f"code={max(0, dom_target['code'] - dom_kept['code'])}")
    else:
        print(f"[gen] prompts: total={len(records)}  todo={len(todo)}")

    # server
    proc = None
    if args.launch:
        cmd = [sys.executable, "-m", "sglang.launch_server",
               "--model-path", args.model_path, "--trust-remote-code",
               "--tp", str(args.tp), "--mem-fraction-static", str(args.mem_fraction),
               "--port", str(args.port), "--host", "127.0.0.1"]
        if args.ep > 1:
            cmd += ["--ep-size", str(args.ep)]
        if args.context_length:
            cmd += ["--context-length", str(args.context_length)]
        print("[gen] launching:", " ".join(cmd), flush=True)
        srv_log_path = time.strftime(os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                                  "teacher_server_%Y%m%d_%H%M%S.log"))
        srv_log = open(srv_log_path, "w")
        print(f"[gen] server log: {srv_log_path}  (cold-load of the fp8 MoE checkpoint can take "
              "10-30 min from shared storage; watch 'Loading safetensors checkpoint shards')", flush=True)
        proc = subprocess.Popen(cmd, stdout=srv_log, stderr=srv_log,
                                preexec_fn=os.setsid)
        if not wait_health(args.port):
            print(f"[gen] server failed to become healthy; see {srv_log_path}"); sys.exit(1)

    out_lock = threading.Lock()
    dom_lock = threading.Lock()   # guards dom_kept for --total quota enforcement
    grade_sem = threading.Semaphore(args.grade_workers)
    stats = {"attempted": 0, "len_trunc": 0, "malformed": 0, "wrong": 0, "kept": 0}
    stats_lock = threading.Lock()
    out_f = open(args.out, "a", encoding="utf-8")

    def bump(k, n=1):
        with stats_lock:
            stats[k] += n

    def run_prompt(item):
        uid, dom, rec = item
        if dom_target is not None:
            with dom_lock:
                if dom_kept[dom] >= dom_target[dom]:
                    return   # this domain's quota already met
        need = args.keep - kept_by_uid.get(uid, 0)
        prompt_text = tok.apply_chat_template(
            rec["prompt"], tokenize=False, add_generation_prompt=True, **tmpl_kwargs
        )
        for s_idx in range(args.n_samples):
            if need <= 0:
                return
            if dom_target is not None:
                with dom_lock:
                    if dom_kept[dom] >= dom_target[dom]:
                        return   # quota filled by another worker mid-flight
            bump("attempted")
            try:
                out = http_json(f"http://127.0.0.1:{args.port}/generate", {
                    "text": prompt_text,
                    "sampling_params": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    },
                })
            except Exception as e:
                sys.stderr.write(f"[gen] uid={uid} s={s_idx} generate err {type(e).__name__}\n")
                continue
            fr = ((out.get("meta_info") or {}).get("finish_reason") or {})
            if (fr.get("type") if isinstance(fr, dict) else fr) == "length":
                bump("len_trunc"); continue
            text = normalize_for_student(out.get("text", ""), prefill_think)
            if text is None:
                bump("malformed"); continue
            with grade_sem:
                ok = grade(rec, text)
            if not ok:
                bump("wrong"); continue
            row = {
                "uid": uid, "domain": dom,
                "source": rec.get("source"), "label": rec.get("label"),
                "prompt": rec["prompt"], "metadata": rec.get("metadata"),
                "response": text,
                # build_sft_warmup_data.py compatibility: it reads (reasoning, answer)
                # and its fallback splits `answer` at </think> when reasoning is empty.
                "reasoning": "", "answer": text,
                "teacher": args.model_path,
                "gen": {"temperature": args.temperature, "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens, "sample_idx": s_idx},
                "acc": True,
            }
            with out_lock:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n"); out_f.flush()
            bump("kept"); need -= 1
            if dom_target is not None:
                with dom_lock:
                    dom_kept[dom] += 1

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            done = 0
            for _ in ex.map(run_prompt, todo):
                done += 1
                if done % 200 == 0:
                    with stats_lock:
                        s = dict(stats)
                    extra = f"  dom={dom_kept}" if dom_target is not None else ""
                    print(f"[gen] {done}/{len(todo)} prompts  {s}{extra}  {time.time()-t0:.0f}s", flush=True)
                if dom_target is not None and all(dom_kept[d] >= dom_target[d] for d in dom_target):
                    print(f"[gen] all domain quotas met (math={dom_kept['math']} "
                          f"code={dom_kept['code']}); stopping.", flush=True)
                    break
    finally:
        out_f.close()
        if proc is not None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            print("[gen] server terminated")
    print(f"[gen] DONE {stats}  out={args.out}")


if __name__ == "__main__":
    main()
