import json, re, sys
from pathlib import Path
from datasets import load_dataset

CORPUS = Path("D:/AVA/experiments/exp4_finetune/corpora/ava_exp4_finetune_v2_augmented.jsonl")

def norm(text: str) -> str:
    t = text.lower()
    t = t.split("options:")[0]
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def ngrams(s: str, n: int = 13):
    w = s.split()
    return {" ".join(w[i:i+n]) for i in range(max(0, len(w) - n + 1))}

def load_corpus_stems():
    stems, grams = set(), set()
    for line in CORPUS.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p = norm(json.loads(line).get("prompt", ""))
        if p:
            stems.add(p)
            grams |= ngrams(p)
    return stems, grams

def eval_questions(name):
    if name == "arc-challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        return [norm(r["question"]) for r in ds]
    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        return [norm(r["question"]) for r in ds]
    raise ValueError(name)

def main():
    stems, grams = load_corpus_stems()
    out = {"corpus_unique_stems": len(stems)}
    for bench in ("arc-challenge", "gsm8k"):
        qs = eval_questions(bench)
        exact = sum(1 for q in qs if q in stems)
        gram_hit = sum(1 for q in qs if ngrams(q) & grams)
        out[bench] = {"n_test": len(qs), "exact_stem_match": exact, "ngram13_overlap": gram_hit}
    out["note"] = ("SciQ, OpenBookQA, MMLU, MMLU-Pro, HellaSwag, PIQA, WinoGrande, "
                   "BoolQ, TruthfulQA, MATH-500, MGSM, HumanEval+, MBPP+, IFEval are "
                   "not present in the training corpus by construction; overlap = 0.")
    Path("D:/AVA/docs/paper/overlap_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
