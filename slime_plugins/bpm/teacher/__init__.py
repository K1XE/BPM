"""BPM teacher rollout stack: prefill-only forwards on a separate SGLang engine,
returning hidden states that the loss projects through the frozen teacher lm_head.
Only the first three modules are import-light; the rest are imported lazily.

bpm_teacher_tokens     student token/EOS normalization and alignment text
bpm_teacher_request    teacher-tokenizer prompt/prefill construction and loss masks
bpm_teacher_writeback  payload validation, sample injection, post-process hook
bpm_teacher_payload    teacher lifecycle and RPC orchestration
bpm_teacher_rollout    Ray placement/startup bound onto the manager
bpm_teacher_engine     subprocess SGLang teacher engine service
bpm_teacher_handlers   subprocess request handlers
bpm_sglang_patch       scheduler hidden-states extraction patch
"""
