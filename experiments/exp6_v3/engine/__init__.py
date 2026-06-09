"""AVA v3 student architecture modules.

Structure:
  - ternary_linear.py    : ternary {-1, 0, +1} group-256 QAT Linear (implemented, P2)
  - mote_ffn.py          : Mixture of Ternary Experts + shared expert (implemented, P2)
  - mamba3_block.py      : Mamba-3 L-block mixer — FLA kernel + PyTorch reference (implemented, P2)
  - gated_deltanet.py    : FLA Gated DeltaNet fallback wrapper (implemented, P2)
  - hrm_core.py          : HRM refinement recurrence + convergence-aware halting (implemented, P2)
  - student_model.py     : V3Config + AVAv3ForCausalLM full assembly (implemented, P2)
  - distill.py           : teacher KL + MiniLM-attn distillation losses (stub, P4/P5)

Design source: docs/v3/ (June 2026 revision). Smoke tests: tests/test_engine_smoke.py.
"""
from .hrm_core import HaltingHead, HRMOutput, HRMRepeatUnit
from .mamba3_block import ReferenceMamba3Mixer, build_l_mixer
from .mote_ffn import MoTEFFN, TernaryExpert
from .student_model import AVAv3ForCausalLM, V3Config, V3ModelOutput, count_full_size_params
from .ternary_linear import TernaryLinear

__all__ = [
    "AVAv3ForCausalLM",
    "HaltingHead",
    "HRMOutput",
    "HRMRepeatUnit",
    "MoTEFFN",
    "ReferenceMamba3Mixer",
    "TernaryExpert",
    "TernaryLinear",
    "V3Config",
    "V3ModelOutput",
    "build_l_mixer",
    "count_full_size_params",
]
