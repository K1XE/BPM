#!/usr/bin/env python3
"""Convert the Path-B code eval jsonl (lcb) into the Path-A code_sandbox
metadata schema so `train.py --debug-rollout-only` can score them through a custom RM.

WHY this exists
---------------
Path A (train.py eval) scores each sample through the rollout reward hub. With
`--custom-rm-path examples.bpm.eval.eval_reward.reward_func`, a sample whose
`metadata.rm_type == "code_sandbox"` is graded by executing code in the killable
sandbox (examples/bpm/reward/code_eval.py). The training reward adapter's code_sandbox path
only calls `score_taco` (TACO {inputs,outputs,fn_name}); MBPP+ (assert test body) and
LCB (stdin/functional) need `score_mbpp` / `score_lcb`. eval_reward.reward_func routes
on `metadata.code_grader`, and the grader reads its payload from `metadata`:

  mbpp : score_mbpp(response, metadata["test"])                       -> assert test body
  lcb  : score_lcb(response, {testtype, tests, func_name})            -> stdin / functional

The stock benchmark files carry those fields at TOP LEVEL with label=None; this script
moves them under `metadata` and stamps rm_type/code_grader/has_tests. Originals are left
untouched (they still drive Path B / eval_ckpt.py).

Usage:
  python3 prepare_code_eval_data.py \
     [--eval-dir ./data/eval] \
     [--out-dir  ./data/eval]
Outputs: <out-dir>/lcb_v6_full_codesandbox.jsonl
"""
import argparse
import json
import os
import sys


def _convert_mbpp(row: dict) -> dict:
    test_body = row.get("test") or ""
    entry = row.get("entry_point") or ""
    return {
        "prompt": row.get("prompt"),
        "label": entry,  # unused for grading; keeps --eval-label-key happy
        "domain": "code",
        "source": row.get("source", "mbppplus"),
        "metadata": {
            "domain": "code",
            "rm_type": "code_sandbox",
            "code_grader": "mbpp",
            "has_tests": bool(test_body.strip()),
            "test": test_body,
            "entry_point": entry,
        },
    }


def _convert_bcb(row: dict) -> dict:
    test_body = row.get("test") or ""
    return {
        "prompt": row.get("prompt"),
        "label": row.get("entry_point") or "task_func",
        "domain": "code",
        "source": row.get("source", "bigcodebench-hard"),
        "metadata": {
            "domain": "code",
            "rm_type": "code_sandbox",
            "code_grader": "bcb",
            "has_tests": bool(test_body.strip()),
            "test": test_body,
            "entry_point": row.get("entry_point") or "task_func",
        },
    }


def _convert_lcb(row: dict) -> dict:
    tests = row.get("tests") or []
    return {
        "prompt": row.get("prompt"),
        "label": row.get("func_name") or row.get("question_id") or "",
        "domain": "code",
        "source": row.get("source", "lcb"),
        "metadata": {
            "domain": "code",
            "rm_type": "code_sandbox",
            "code_grader": "lcb",
            "has_tests": bool(tests),
            # score_lcb(generation, problem) reads exactly these three keys.
            "problem": {
                "testtype": row.get("testtype"),
                "tests": tests,
                "func_name": row.get("func_name") or "",
            },
        },
    }


def _convert_taco(row: dict) -> dict:
    """TACO rows carry their cases in `input_output` (or an already-JSON-string `tests`).

    ``score_taco(generation, tests)`` consumes the decoded dict directly, so the payload is
    stored as a JSON STRING under ``metadata.tests`` -- TACO case values can exceed the range
    of a JSON number consumer that re-parses them eagerly, and keeping the raw text defers
    decoding to the grader. Both the stdin form ({"inputs", "outputs"}) and the functional
    form (adds "fn_name") pass through unchanged.
    """
    raw = row.get("input_output") or row.get("tests") or ""
    if isinstance(raw, (dict, list)):
        raw = json.dumps(raw, ensure_ascii=False)
    has_tests = False
    if raw:
        try:
            decoded = json.loads(raw)
            has_tests = bool(decoded.get("inputs")) and bool(decoded.get("outputs"))
        except (ValueError, AttributeError):
            has_tests = False
    return {
        "prompt": row.get("prompt"),
        "label": row.get("fn_name") or row.get("question_id") or "",
        "domain": "code",
        "source": row.get("source", "taco"),
        "metadata": {
            "domain": "code",
            "rm_type": "code_sandbox",
            "code_grader": "taco",
            "has_tests": has_tests,
            "tests": raw,
            "difficulty": row.get("difficulty", ""),
        },
    }


# EvalPlus HumanEval/32 (find_zero) ships a BROKEN special oracle in its `test`
# (`_poly(*candidate(*inp), inp)` unpacks the scalar root -> TypeError -> EVERY submission
# fails, a uniform -0.61pp on all models). Its correct oracle can't be reconstructed 1:1, so
# we drop it and report HumanEval+ on 163/164; note this in the paper.
_SKIP_TASK_IDS = {"HumanEval/32"}


def _run(src: str, dst: str, convert) -> bool:
    """Convert one benchmark file. Returns False (and skips) when the source is absent,
    so a user who only downloaded a subset of the benchmarks still gets the rest."""
    if not os.path.isfile(src):
        print(f"[prepare] skipping absent source: {src}", file=sys.stderr)
        return False
    n_in = n_out = n_notests = n_skip = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") in _SKIP_TASK_IDS:
                n_skip += 1
                continue
            n_in += 1
            out = convert(row)
            if not out["metadata"]["has_tests"]:
                n_notests += 1
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[prepare] {os.path.basename(src)} -> {os.path.basename(dst)}: "
          f"{n_out}/{n_in} rows ({n_notests} without recoverable tests -> acc 0)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", dest="eval_dir", default=os.environ.get("BPM_EVAL_DATA_DIR", "./data/eval"))
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.eval_dir
    os.makedirs(out_dir, exist_ok=True)
    # HumanEval+ rows already bake `check(<entry_point>)` into `test` (build_humanevalplus.py),
    # so the plain assert-body converter applies unchanged.
    # The two code benchmarks used in the paper. Each is skipped (with a warning) when its
    # source file is absent, so a partial download still converts whatever is present.
    done = [
        _run(os.path.join(args.eval_dir, "humanevalplus.jsonl"),
             os.path.join(out_dir, "humanevalplus_codesandbox.jsonl"), _convert_mbpp),
        # official LCB date window 25.01-25.04
        _run(os.path.join(args.eval_dir, "lcb2025.jsonl"),
             os.path.join(out_dir, "lcb2025_codesandbox.jsonl"), _convert_lcb),
        _run(os.path.join(args.eval_dir, "taco_test_em.jsonl"),
             os.path.join(out_dir, "taco_test_em_codesandbox.jsonl"), _convert_taco),
    ]
    if not any(done):
        print(
            "[prepare] no benchmark source found under "
            f"{args.eval_dir}; see README.md for where to obtain them.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
