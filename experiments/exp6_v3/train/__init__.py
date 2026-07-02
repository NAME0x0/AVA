"""AVA v3.0 training pipeline — the staged-scope shippable path.

Modules (docs/v3/REVIEW_2026-07.md sections 5, 9, 11):

    sandbox_exec  subprocess-isolated code execution for eval + verification
    data          streaming mixture, masking, FIM, hashline dialect, decontam
    c1_eval       donor baseline eval (HumanEval+/MBPP+/floors, dual mode)
    sft           resumable QLoRA shard trainer (CheckpointSync-backed)
    gate          phase-gate regression check (<= 2 pp rule, CODE_PIVOT section 8)

Everything here is v3.0: donor + specialization. v3.1 lanes (ternary QAT,
HRM mount, self-play RL) are separate, gated, and intentionally NOT stubbed
here — they require research iteration, not scaffolding.
"""
