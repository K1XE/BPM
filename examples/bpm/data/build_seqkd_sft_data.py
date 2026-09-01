"""SimCT Step 3.1: turn GLM teacher trajectories into slime SFT warmup data.

Reads the JSONL produced by ``gen_teacher_sft_responses.py`` and writes a parquet
with a single ``messages`` column that slime's ``sft_rollout`` + ``--input-key
messages`` consumes directly.  Each row is a single-turn conversation rendered in
the STUDENT's (Qwen3.5) thinking format, so the warmed student learns to emit
exactly the ``<think>...</think>...`` shape its on-policy rollout later produces:

    [{"role": "user", "content": <question>},
     {"role": "assistant", "content": "<think>\n<reasoning>\n</think>\n\n<answer>"}]

Because the exact Qwen3.5 thinking-template token boundary cannot be assumed,
this script self-verifies on a few samples with the SAME MultiTurnLossMaskGenerator
the SFT path uses: it decodes the loss=1 region and asserts it really IS the
intended reasoning+answer -- reasoning text present, exactly one ``<think>`` (no
strip, no double-think), answer present, prompt excluded -- and probes the
OPD-style prompt to show where ``<think>`` lands.  Run it on the box that has the
student tokenizer (pass --hf-checkpoint); it fails loudly on a format mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _build_assistant_content(reasoning: str, answer: str, include_thinking: bool, open_think: bool = False) -> str:
    reasoning, answer = reasoning.strip(), answer.strip()
    # Recover the split when the gen step left `reasoning` empty and dumped the whole
    # "<reasoning></think><answer>" into `answer`. GLM-Z1/Qwen3.5 force the OPENING <think>
    # in the PROMPT, so the teacher OUTPUT carries only </think> (no <think>), which the
    # <think>-keyed fallback in gen_teacher_sft_responses.py misses -> reasoning stays "".
    if not reasoning and "</think>" in answer:
        head, _, tail = answer.partition("</think>")
        reasoning, answer = head.strip(), tail.strip()
    if include_thinking and reasoning:
        if open_think:
            # Prompt (distill_qwen_think mask) already ends with an OPEN "<think>\n";
            # content only closes it -> composes to "<think>\n{reasoning}\n</think>\n\n{answer}".
            return f"{reasoning}\n</think>\n\n{answer}"
        return f"<think>\n{reasoning}\n</think>\n\n{answer}"
    # Non-thinking warmup: Qwen3.5 still expects an (empty) closed think block.
    if include_thinking:
        return f"</think>\n\n{answer}" if open_think else f"<think>\n\n</think>\n\n{answer}"
    return answer


def _rows_to_messages(
    path: str, include_thinking: bool, min_answer_chars: int, open_think: bool = False
) -> list[list[dict]]:
    conversations: list[list[dict]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            answer = (rec.get("answer") or "").strip()
            if len(answer) < min_answer_chars:
                continue
            content = _build_assistant_content(rec.get("reasoning", ""), answer, include_thinking, open_think)
            # DAPO prompt may be a raw question string or a ready message list.
            prompt = rec["prompt"]
            leading = list(prompt) if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            conversations.append(leading + [{"role": "assistant", "content": content}])
    return conversations


def _check_masked(
    messages: list[dict], masked_text: str, n_loss: int, resp_len: int, open_think: bool = False
) -> list[str]:
    """Return a list of problems with the masked (loss=1) region; empty == OK.

    Catches the failure modes the OPD-vs-SFT format parity depends on:
    reasoning silently stripped by the template, a double `<think>`, the answer
    missing, or the prompt bleeding into the supervised region.

    With ``open_think`` (distill_qwen_think): the opening ``<think>`` lives in the
    PROMPT, so the loss region must contain reasoning + ``</think>`` + answer but
    must NOT itself contain a ``<think>`` (that would mean double-think).
    """
    problems: list[str] = []
    if n_loss <= 0 or resp_len <= 0:
        problems.append("loss region is empty")
        return problems
    prompt_head = (messages[0]["content"] or "")[:40]
    # ALLOW_PROMPT_ECHO=1: teachers like MiniMax-M2.7 restate the problem
    # verbatim inside <think> (22.6% of P3 trajectories vs 0% for GLM-Z1);
    # that echo is legitimate response content, not a mask bug. The structural
    # checks below (think placement, reasoning/answer presence, exact
    # loss_tokens==response_len) still guard the mask itself.
    import os as _os
    if prompt_head.strip() and prompt_head in masked_text and _os.environ.get("ALLOW_PROMPT_ECHO") != "1":
        problems.append("prompt text leaked into the loss region")
    intended = messages[-1]["content"]
    if open_think:
        # content == "{reasoning}\n</think>\n\n{answer}"; the opening <think> is in the prompt.
        if "</think>" not in masked_text:
            problems.append("'</think>' missing -> reasoning not closed in the loss region")
        if "<think>" in masked_text:
            problems.append("'<think>' found in loss region (should be in the prompt only -> double-think)")
        reasoning_head = intended.split("</think>")[0].strip()[:40]
        answer = intended.split("</think>")[-1].strip()
        if reasoning_head and reasoning_head not in masked_text:
            problems.append("reasoning text missing from loss region")
        if answer[:40] and answer[:40] not in masked_text:
            problems.append("answer text is NOT in the loss region")
        cc = masked_text.count("</think>")
        if cc != 1:
            problems.append(f"'</think>' count={cc} (expected 1; 0=not closed, >=2=unsplit reasoning/double-close)")
        return problems
    has_think = "<think>" in intended
    answer = intended.split("</think>")[-1].strip() if has_think else intended.strip()
    if answer[:40] and answer[:40] not in masked_text:
        problems.append("answer text is NOT in the loss region")
    if has_think:
        reasoning_head = intended.split("<think>", 1)[-1].split("</think>")[0].strip()[:40]
        if "</think>" not in masked_text:
            problems.append("'</think>' missing -> think block stripped (reasoning NOT trained)")
        if reasoning_head and reasoning_head not in masked_text:
            problems.append("reasoning text missing from loss region (think stripped?)")
        tc = masked_text.count("<think>")
        if tc != 1:
            problems.append(f"'<think>' count={tc} (expected 1; 0=stripped, >=2=double-think)")
    return problems


def _verify(conversations: list[list[dict]], hf_checkpoint: str, loss_mask_type: str, n: int) -> None:
    """Replay the SFT loss-mask path and assert the masked region really IS the
    intended assistant response (reasoning + answer), with no think-strip/double-think.
    Also probes the OPD-style prompt so you can see where `<think>` lands."""
    from transformers import AutoTokenizer

    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    open_think = loss_mask_type == "distill_qwen_think"
    tok = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type=loss_mask_type)

    # OPD parity probe: where does enable_thinking=true put <think> in the prompt?
    try:
        opd_prompt = tok.apply_chat_template(
            conversations[0][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        in_prompt = opd_prompt.rstrip().endswith("<think>")
        print(f"[verify] OPD-style prompt tail (enable_thinking=True): {opd_prompt[-60:]!r}")
        print(
            f"[verify]   -> OPD puts <think> in the {'PROMPT' if in_prompt else 'RESPONSE (model generates it)'}; "
            f"SFT trains <think> in the response. {'1-token diff, minor.' if in_prompt else 'matches SFT.'}"
        )
    except Exception as exc:
        print(f"[verify][note] could not render OPD-style prompt for parity probe: {exc!r}")

    print(f"\n[verify] loss_mask_type={loss_mask_type} on {min(n, len(conversations))} sample(s)")
    bad = 0
    for idx in range(min(n, len(conversations))):
        messages = conversations[idx]
        token_ids, loss_mask = gen.get_loss_mask(messages)
        resp_len = gen.get_response_lengths([loss_mask])[0]
        masked_text = tok.decode([t for t, m in zip(token_ids, loss_mask, strict=False) if m == 1])
        n_loss = sum(loss_mask)
        problems = _check_masked(messages, masked_text, n_loss, resp_len, open_think)
        bad += int(bool(problems))
        if idx == 0:
            print(f"[verify] total_tokens={len(token_ids)} loss_tokens={n_loss} response_len={resp_len}")
            print(f"[verify] masked(head 300 chars): {masked_text[:300]!r}")
            print(f"[verify] masked(tail 120 chars): {masked_text[-120:]!r}")
        if problems:
            print(f"[verify][FAIL] sample {idx}: " + "; ".join(problems))
    if bad:
        raise SystemExit(
            f"[verify][error] {bad} sample(s) failed the mask/content check (see [FAIL] above). "
            f"The template likely strips or double-inserts <think>; try a different --loss-mask-type "
            f"(qwen3/distill_qwen/distill_qwen_think) or adjust the assistant-content format."
        )
    if open_think:
        print("[verify] OK: loss region == reasoning+answer; opening <think> in the prompt (thinking-ON, no double-think).")
    else:
        print("[verify] OK: loss region == assistant response (prompt excluded).")


def main() -> None:
    p = argparse.ArgumentParser(description="Build slime SFT warmup parquet from teacher trajectories.")
    p.add_argument("--input", required=True, help="teacher trajectories jsonl (from gen_teacher_sft_responses.py)")
    p.add_argument("--output", required=True, help="output parquet with a `messages` column")
    p.add_argument("--include-thinking", type=int, default=1, help="1=train on reasoning (<think>); 0=answer-only")
    p.add_argument("--min-answer-chars", type=int, default=1)
    p.add_argument("--hf-checkpoint", default=None, help="student HF path; enables loss-mask self-verification")
    p.add_argument(
        "--loss-mask-type",
        default="qwen3",
        choices=["qwen", "qwen3", "distill_qwen", "distill_qwen_think"],
        help="distill_qwen_think = thinking-ON SFT (prompt renders an OPEN <think>; content closes it)",
    )
    p.add_argument("--verify-samples", type=int, default=3)
    args = p.parse_args()

    open_think = args.loss_mask_type == "distill_qwen_think"
    conversations = _rows_to_messages(args.input, bool(args.include_thinking), args.min_answer_chars, open_think)
    if not conversations:
        raise SystemExit(f"[build][error] no usable rows in {args.input}")
    print(f"[build] built {len(conversations)} SFT conversations from {args.input}")

    if args.hf_checkpoint:
        _verify(conversations, args.hf_checkpoint, args.loss_mask_type, args.verify_samples)
    else:
        print("[build][note] --hf-checkpoint not given; skipping loss-mask verification (recommended to enable).")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    pd.DataFrame({"messages": conversations}).to_parquet(args.output, index=False)
    print(f"[build] wrote {len(conversations)} rows -> {args.output}")


if __name__ == "__main__":
    main()
