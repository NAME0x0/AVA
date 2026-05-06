"""AVA v3 student architecture modules.

Structure:
  - ternary_linear.py    : ternary {-1, 0, +1} weight quantization-aware Linear
  - mote_ffn.py          : Mixture of Ternary Experts with shared FP expert
  - subln.py             : SubLN normalization (BitNet-style)
  - gated_deltanet.py    : FLA Gated DeltaNet wrapper for the 3:1 hybrid stack
  - student_model.py     : top-level student model class
  - distill.py           : teacher KL + MiniLM-attn distillation losses

All modules are stubs at P0. Implementation begins at P2.
"""
