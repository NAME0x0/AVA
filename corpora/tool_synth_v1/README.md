# Tool Synth V1

Deterministic synthetic tool-supervision packet for AVA's calculator branch.

Contents:
- exact compact calculator traces
- direct calculator-answer prompts
- no-tool abstention prompts
- calculator boundary refusals

Generation:
- source generator: `src/ava/synthetic.py`
- protocol: `compact_tags`
- tracked manifest: `manifest.json`

This packet is intended to test whether a larger, cleaner trace curriculum improves exact tool behavior before moving to tool RL or tokenizer changes.
