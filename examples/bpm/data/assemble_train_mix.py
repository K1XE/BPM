#!/usr/bin/env python3
r"""Build the paper training mix = 10000 math + 10000 taco (1:1).

PROVENANCE ONLY -- this script documents how the released mix was assembled; it is
NOT runnable from a fresh clone, because its four input artifacts (the decontaminated
math set and the curated code pools) are frozen release data rather than something
rebuilt from public sources at run time. To reproduce training, download the assembled
mix directly from https://huggingface.co/datasets/K1zE/BPM .

Design (deterministic, seed 0):
  MATH 10000: from dapo_math_17k.jsonl (14065; the paper's decontam-v2 math set) ->
    ctrl/moji screen -> re-verify decontam vs the CURRENT math suite (aime2026/hmmt2026/
    math500: 13-gram containment 0.7 + NFKC-norm 6-gram 0.55, house thresholds) DROPPING
    any hit -> body dedup -> length-stratified stride to 10000 (same as assemble_final_18k).
  CODE 10000: a frozen curated 8000 (fully curated + LLM-judged; the
    WS-mask TACO verdict ran on exactly these) + 2000 top-up:
      first from _code_cand.jsonl (drop judge-rejects {616,1848,4312}, apply proxies A/B/D
      + repair C, drop rows already frozen), then if short, mine _taco_pool.jsonl through
      the same curate gates (clean_gradeable / ctrl / img / mixed-contract / dedup) —
      every top-up row additionally decontamed vs lcb_v6_full(=lcb2025 superset),
      humanevalplus AND bcbhard (8g containment 0.6 + norm 6g 0.60).
Verify: whole-file decontam re-check vs all 6 paper sets, split counts, md5.
"""
import json, re, random, unicodedata, hashlib
from collections import defaultdict
random.seed(0)

import os
D = os.environ.get("BPM_DATA_DIR", "./data")
OUT = f"{D}/train_math10k_taco10k.jsonl"
MATH_N = CODE_N = 10000
CODE_FRAC = {"VERY_HARD": 0.16, "HARD": 0.19, "MEDIUM_HARD": 0.19,
             "MEDIUM": 0.19, "UNKNOWN_DIFFICULTY": 0.19, "EASY": 0.08}
FILL = ["HARD", "MEDIUM_HARD", "MEDIUM", "UNKNOWN_DIFFICULTY", "VERY_HARD", "EASY"]
JUDGE_REJECT_CODE = {616, 1848, 4312}

# ---------- helpers (verbatim from assemble_final_18k / curate_18k) ----------
_LEAD = [r"where \$Answer is the answer to the problem\.\s*",
         r"Do not put a period after the answer\.\s*",
         r"read from standard input and write to standard output\.\s*"]
_TRAIL = [r"\s*Remember to put your answer on its own line after \"Answer:\"\.?"]

def full_content(rec):
    return "\n".join(m.get("content", "") for m in rec["prompt"])

def body(rec):
    t = " ".join(m.get("content", "") for m in rec["prompt"])
    for p in _LEAD:
        parts = re.split(p, t, maxsplit=1)
        if len(parts) == 2:
            t = parts[1]
    for p in _TRAIL:
        t = re.split(p, t, maxsplit=1)[0]
    return t.strip()

def words(s): return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()
def ngrams(ws, n): return set(tuple(ws[i:i+n]) for i in range(len(ws)-n+1)) if len(ws) >= n else set()
def load(p): return [json.loads(l) for l in open(p)]
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
MOJI_RE = re.compile(r"Ã.|â€|â‚")

def norm_words(s):
    t = unicodedata.normalize("NFKC", s).lower()
    t = re.sub(r"[\x00-\x1f]", " ", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("$", " ").replace("{", " ").replace("}", " ")
    return re.findall(r"[a-z0-9]+", t)

def build_norm_index(paths, n):
    ev = []
    for p in paths:
        for r in load(p):
            g = ngrams(norm_words(body(r)), n)
            if g: ev.append(g)
    inv = defaultdict(list)
    for k, g in enumerate(ev):
        for gg in g: inv[gg].append(k)
    return ev, inv

def norm_contam(row_body, ev, inv, n, thr):
    tg = ngrams(norm_words(row_body), n)
    if not tg: return False
    cand = defaultdict(int)
    for gg in tg:
        for k in inv[gg]: cand[k] += 1
    return any(ev[k] and sh/len(ev[k]) >= thr for k, sh in cand.items())

def eval_index(paths, n):
    idx = []
    for path in paths:
        for r in load(path):
            g = ngrams(words(body(r) if isinstance(r.get("prompt"), list) else str(r.get("prompt",""))), n)
            if g: idx.append(g)
    return idx

def contaminated(row_body, checks):
    for _idx, n, thr in checks:
        tn = ngrams(words(row_body), n)
        if not tn: continue
        for e in _idx:
            if len(tn & e)/len(e) >= thr: return True
    return False

# clean_gradeable / GfG-mislabel (verbatim curate_18k)
_ARR_DISPLAY = re.compile(r"\[\s*\]\s*=\s*\{")
_VAR_DISPLAY = re.compile(r"^\s*[A-Za-z_]\w*\s*(\[\s*\])?\s*=\s*\S")
_NOREAD = re.compile(r"you don'?t need to read (input|or print)|don'?t need to read input"
                     r"|do not need to read input", re.I)
def _first_line_var_display(inp):
    if not isinstance(inp, str): return False
    for ln in inp.splitlines():
        if ln.strip(): return bool(_VAR_DISPLAY.match(ln))
    return False
def is_gfg_mislabel(r):
    t = r.get("tests", {})
    if not isinstance(t, dict) or t.get("type") != "stdin": return False
    is_gfg = r.get("source", "") == "taco:geeksforgeeks"
    for c in t.get("cases", []):
        inp = c.get("input", "")
        if not isinstance(inp, str): continue
        if _ARR_DISPLAY.search(inp): return True
        if is_gfg and _first_line_var_display(inp): return True
    return bool(_NOREAD.search(full_content(r)))
def clean_gradeable(r):
    t = r.get("tests", {})
    return isinstance(t, dict) and bool(t.get("cases")) and not is_gfg_mislabel(r)

_IMG = re.compile(r"<image>|as shown in the (figure|picture|image|diagram)|shown in the (figure|picture|image)"
                  r"|see the (figure|picture|image)|refer to the (figure|picture|image)"
                  r"|in the (figure|image|diagram) (below|above)|\[image\]", re.I)
_MIX_STDIN = re.compile(r"first line of (the )?input|input contains.*test cases|read.*from stdin", re.I)
_MIX_BENIGN = re.compile(r"do(es)? not (need to )?read", re.I)
def mixed_contract(txt):
    return ("Starter code:" in txt) and bool(_MIX_STDIN.search(txt)) and not _MIX_BENIGN.search(txt)

# proxies (verbatim assemble_final_18k; operate on tests DICT)
_RESIDUE = re.compile(r"^\s*(case\s*)?\d+\s*:\s*$", re.I)
_FLOAT = re.compile(r"^-?\d+\.\d+$")
_FRAC = re.compile(r"^-?\d+/\d+$")
_POW2CTX = re.compile(r"2\^|2\\\^|power of two|power-of-two", re.I)
def first_nonempty(inp):
    if isinstance(inp, list):
        for ln in inp:
            if isinstance(ln, str) and ln.strip(): return ln
        return ""
    if isinstance(inp, str):
        for ln in inp.splitlines():
            if ln.strip(): return ln
    return ""
def is_pow2(x): return x >= 1 and (x & (x-1)) == 0
def proxy_A(t):
    return any(_RESIDUE.match(first_nonempty(c.get("input",""))) for c in t.get("cases", []))
def _str_outs(t):
    return [c["output"].strip() for c in t.get("cases", [])
            if isinstance(c.get("output"), str) and c["output"].strip()]
def proxy_B(t):
    if t.get("type") != "stdin": return False
    outs = _str_outs(t)
    if len(outs) < 2 or len(outs) != len(t.get("cases", [])): return False
    return len(set(outs)) == 1 and bool(_FLOAT.match(outs[0]))
def proxy_D(t, ptxt):
    outs = _str_outs(t)
    if len(outs) < 1 or not all(_FRAC.match(o) for o in outs): return False
    if not re.search(r"probabilit|expected", ptxt, re.I): return False
    if not _POW2CTX.search(ptxt): return False
    return any(not is_pow2(int(o.split("/")[1])) for o in outs)
def proxy_C_repair(t):
    ch = False
    for c in t.get("cases", []):
        o = c.get("output")
        if isinstance(o, str) and o[:1] == "\n":
            c["output"] = o.lstrip("\n"); ch = True
    return ch

# ---------- decontam indexes ----------
print("[gate] building decontam indexes ...", flush=True)
MATH_SUITE = [f"{D}/eval/aime2026.jsonl", f"{D}/eval/hmmt2026.jsonl", f"{D}/eval/math500.jsonl"]
CODE_SUITE = [f"{D}/eval/lcb_v6_full.jsonl", f"{D}/eval/humanevalplus.jsonl", f"{D}/eval/bcbhard.jsonl"]
MATH_EVAL = eval_index(MATH_SUITE, 13)
CODE_EVAL13 = eval_index([f"{D}/eval/lcb_v6_full.jsonl"], 13)
CODE_EVAL8 = eval_index([f"{D}/eval/humanevalplus.jsonl", f"{D}/eval/bcbhard.jsonl"], 8)
MATH_NORM, MATH_NORM_INV = build_norm_index(MATH_SUITE, 6)
CODE_NORM, CODE_NORM_INV = build_norm_index(CODE_SUITE, 6)
def math_contam(b):
    return contaminated(b, [(MATH_EVAL, 13, 0.7)]) or norm_contam(b, MATH_NORM, MATH_NORM_INV, 6, 0.55)
def code_contam(b):
    return contaminated(b, [(CODE_EVAL13, 13, 0.7), (CODE_EVAL8, 8, 0.6)]) or \
           norm_contam(b, CODE_NORM, CODE_NORM_INV, 6, 0.60)

bkey = lambda r: re.sub(r"\s+", " ", body(r).lower())[:400]

# ================= MATH 10000 =================
math = load(f"{D}/dapo_math_17k.jsonl")
n0 = len(math)
math = [r for r in math if not (CTRL_RE.search(full_content(r)) or MOJI_RE.search(full_content(r)))]
n_ctrl = len(math)
math = [r for r in math if not math_contam(body(r))]
n_dec = len(math)
seen = set()
math = [r for r in math if (k := bkey(r)) not in seen and not seen.add(k)]
n_dup = len(math)
math.sort(key=lambda r: (len(body(r)), bkey(r)))
step = len(math) / MATH_N
frozen_math = [math[int(i*step)] for i in range(MATH_N)] if len(math) > MATH_N else math
print(f"[math] {n0} -> ctrl {n_ctrl} -> decontam(v-suite) {n_dec} -> dedup {n_dup} -> select {len(frozen_math)}")
assert len(frozen_math) == MATH_N
assert all(str(r["label"]).lstrip("-").isdigit() for r in frozen_math)

# ================= CODE 10000 = frozen 8000 + 2000 top-up =================
frozen8k = [r for r in load(f"{D}/curated_code_pool.jsonl") if r.get("domain") == "code"]
print(f"[code] frozen from 18k: {len(frozen8k)}")
frozen_keys = {bkey(r) for r in frozen8k}

# top-up A: _code_cand remainder
cand = load(f"{D}/_code_cand.jsonl")
cand = [r for r in cand if r.get("idx") not in JUDGE_REJECT_CODE]
topA, drops = [], defaultdict(int)
for r in cand:
    t = json.loads(r["metadata"]["tests"])
    ptxt = full_content(r)
    if bkey(r) in frozen_keys: drops["already_frozen"] += 1; continue
    if proxy_A(t): drops["proxy_A"] += 1; continue
    if proxy_B(t): drops["proxy_B"] += 1; continue
    if proxy_D(t, ptxt): drops["proxy_D"] += 1; continue
    if code_contam(body(r)): drops["contam_new_suite"] += 1; continue
    if proxy_C_repair(t):
        r["metadata"]["tests"] = json.dumps(t, ensure_ascii=False)
        r["metadata"]["has_tests"] = bool(t.get("cases"))
    r["_src"] = "code_cand"
    topA.append(r)
print(f"[code] top-up A from _code_cand: {len(topA)} usable (drops {dict(drops)})")

need = CODE_N - len(frozen8k) - len(topA)
topB = []
if need > 0:
    pool = load(f"{D}/_taco_pool.jsonl")
    seen2 = set(frozen_keys) | {bkey(r) for r in topA}
    pdrop = defaultdict(int)
    for r in pool:
        if len(topB) >= need + 500: break            # small buffer for tier balance
        if not clean_gradeable(r): pdrop["ungradeable"] += 1; continue
        fc = full_content(r)
        if CTRL_RE.search(fc) or MOJI_RE.search(fc): pdrop["ctrl"] += 1; continue
        b = body(r)
        if _IMG.search(b): pdrop["img"] += 1; continue
        if mixed_contract(fc): pdrop["mixed"] += 1; continue
        k = bkey(r)
        if k in seen2: pdrop["dup"] += 1; continue
        t = r["tests"]
        if proxy_A(t) or proxy_B(t) or proxy_D(t, fc): pdrop["proxy"] += 1; continue
        if code_contam(b): pdrop["contam"] += 1; continue
        proxy_C_repair(t)
        seen2.add(k)
        topB.append({"prompt": r["prompt"], "label": r.get("label", ""), "domain": "code",
                     "source": r.get("source", ""), "difficulty": r.get("difficulty", ""),
                     "has_starter": bool(r.get("has_starter")),
                     "_src": "taco_pool",
                     "metadata": {"domain": "code", "rm_type": "code_sandbox",
                                  "tests": json.dumps(t, ensure_ascii=False),
                                  "has_tests": bool(t.get("cases"))}})
    print(f"[code] top-up B mined from _taco_pool: {len(topB)} (drops {dict(pdrop)})")

# tier-balanced pick of exactly 2000 from topA+topB
pool2 = topA + topB
for r in pool2:
    r["_tier"] = r.get("difficulty", "") or "UNKNOWN_DIFFICULTY"
by = defaultdict(list)
for r in pool2: by[r["_tier"]].append(r)
for v in by.values():
    v.sort(key=lambda r: bkey(r))
    random.shuffle(v)
TOP_N = CODE_N - len(frozen8k)
avail = {k: len(v) for k, v in by.items()}
take = {k: min(round(CODE_FRAC.get(k, 0.0) * TOP_N), avail.get(k, 0)) for k in CODE_FRAC}
short = TOP_N - sum(take.values())
for k in FILL:
    if short <= 0: break
    room = avail.get(k, 0) - take.get(k, 0)
    add = min(room, short); take[k] = take.get(k, 0) + add; short -= add
topup = []
for k, n in take.items(): topup.extend(by[k][:n])
print(f"[code] top-up take per tier {take} -> {len(topup)}")
assert len(topup) == TOP_N, f"top-up {len(topup)} != {TOP_N}"
src_mix = defaultdict(int)
for r in topup: src_mix[r.pop("_src", "?")] += 1
print(f"[code] top-up source mix: {dict(src_mix)}")
frozen_code = frozen8k + [{k: v for k, v in r.items() if k != "_tier"} for r in topup]

# ================= WRITE + VERIFY =================
def out_math(r):
    return {"prompt": r["prompt"], "label": r["label"], "domain": "math",
            "source": r.get("source", ""), "metadata": {"domain": "math", "rm_type": "dapo"}}
rows = [out_math(r) for r in frozen_math] + frozen_code
random.shuffle(rows)
with open(OUT, "w") as w:
    for o in rows:
        w.write(json.dumps(o, ensure_ascii=False) + "\n")
md5 = hashlib.md5(open(OUT, "rb").read()).hexdigest()
print(f"[write] {OUT}: {len(frozen_math)} math + {len(frozen_code)} code = {len(rows)} | md5 {md5}")

print("\n==================== FINAL VERIFY ====================")
rows = load(OUT)
m = [r for r in rows if r["domain"] == "math"]; c = [r for r in rows if r["domain"] == "code"]
print(f"split: math {len(m)} code {len(c)}")
print(f"math contam(current suite): {sum(1 for r in m if math_contam(body(r)))}")
print(f"code contam(lcb+he+bcb):    {sum(1 for r in c if code_contam(body(r)))}")
print(f"code gradeable:             {sum(1 for r in c if json.loads(r['metadata']['tests']).get('cases'))}/{len(c)}")
print(f"math integer labels:        {sum(1 for r in m if str(r['label']).lstrip('-').isdigit())}/{len(m)}")
import collections
print("code sources:", dict(collections.Counter(r['source'].split(':')[0] if ':' in str(r['source']) else r['source'] for r in c).most_common(5)))
