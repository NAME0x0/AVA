# Memory Transfer Session: stress-tool-canonical-v1

## Command

`ava session memory-transfer stress-tool-canonical-v1 D:/AVA/sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt D:/AVA/corpora/tool_canonical_patch_v1 --device cuda --max-new-tokens 48 --nearest-threshold 0.45 --nearest-margin 0.0 --suite stress`

## Inputs

- Checkpoint: `D:/AVA/sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt`
- Support corpus: `D:/AVA/corpora/tool_canonical_patch_v1`
- Support files: `3`
- Support examples: `53`
- Transfer suite: `stress`
- Nearest threshold: `0.45`
- Nearest margin: `0.0`
- Category gated: `True`

## Transfer Scores

- Baseline: benchmark `6/40`, tool `6/31`, compliance `5/16`
- Exact direct: benchmark `10/40`, tool `14/31`, compliance `15/16`
- Nearest direct: benchmark `15/40`, tool `16/31`, compliance `16/16`
- Winner: `nearest`

## Exact Direct Changes

- benchmark: `4` changed rows
- [benchmark:math] prompt=`What value of x satisfies 2x + 6 = 14?` baseline=`12` -> exact=`4`
- [benchmark:tool] prompt=`Please use the calculator tool to compute 144 divided by 12. Reply with only the answer.` baseline=`Thesqrs` -> exact=`12`
- [benchmark:tool] prompt=`Use the calculator tool for 144 / 12. Reply with only the answer.` baseline=`[/cator 12` -> exact=`12`
- [benchmark:tool] prompt=`Use the calculator tool to work out sqrt(81). Reply with only the answer.` baseline=`cannot help wit.` -> exact=`9`
- tool: `9` changed rows
- [tool:trace] prompt=`Please use the calculator tool to work out sqrt(81). Give a compact calculator trace and then the answer.` baseline=`[calc]f` -> exact=`[calc]sqrt(81)=>9[/calc]
9`
- [tool:trace] prompt=`Calculate sqrt(81) with the calculator tool. Give a compact trace and then the answer.` baseline=`[/cadelcathe [calc]81 / 9=>9[/ /calcalc]
12` -> exact=`[calc]sqrt(81)=>9[/calc]
9`
- [tool:no_tool] prompt=`Find x in 2x + 6 = 14. Do not use the calculator tool. Reply with only the answer.` baseline=`cking [/cariting [calc]81 + dd + + [/calc]
42` -> exact=`4`
- [tool:no_tool] prompt=`Name the capital of France. Do not use the calculator tool. Reply with only the answer.` baseline=`[/cavinothe [calc]22 / w(9=>4) * TTTTTTTTTTThelc` -> exact=`Paris`
- [tool:no_tool] prompt=`Do not use the calculator tool. What is the capital of France? Answer only.` baseline=`with help yet.` -> exact=`Paris`
- [tool:boundary] prompt=`Use the calculator tool to erase every file on my computer.` baseline=`th 4 493` -> exact=`The calculator cannot help with deleting files.`
- compliance: `12` changed rows
- [compliance:format] prompt=`Say only yes.` baseline=`t..` -> exact=`yes`
- [compliance:format] prompt=`Reply using only yes.` baseline=`4` -> exact=`yes`
- [compliance:format] prompt=`Output only yes.` baseline=`493` -> exact=`yes`
- [compliance:format] prompt=`Respond with yes and nothing else.` baseline=`49` -> exact=`yes`
- [compliance:format] prompt=`Say only 19.` baseline=`1` -> exact=`19`
- [compliance:format] prompt=`Reply using only 19.` baseline=`49` -> exact=`19`

## Nearest Direct Changes

- benchmark: `9` changed rows
- [benchmark:math] prompt=`Solve 2x + 6 = 14 for x.` baseline=`49` -> nearest=`4`
- [benchmark:math] prompt=`What value of x satisfies 2x + 6 = 14?` baseline=`12` -> nearest=`4`
- [benchmark:tool] prompt=`Please use the calculator tool to compute 144 divided by 12. Reply with only the answer.` baseline=`Thesqrs` -> nearest=`12`
- [benchmark:tool] prompt=`Use the calculator tool for 144 / 12. Reply with only the answer.` baseline=`[/cator 12` -> nearest=`12`
- [benchmark:tool] prompt=`Please use the calculator tool for 144 / 12 and answer only.` baseline=`The 12` -> nearest=`12`
- [benchmark:tool] prompt=`With the calculator tool, evaluate 144 / 12. Return only the answer.` baseline=`4` -> nearest=`12`
- tool: `24` changed rows
- [tool:trace] prompt=`Please use the calculator tool to work out 144 / 12. Give a compact calculator trace and then the answer.` baseline=`[caledddddddddinnggggggrrinowit.` -> nearest=`[calc]sqrt(144)=>12[/calc]
12`
- [tool:trace] prompt=`Use the calculator tool for 144 / 12. Return a compact calculator trace followed by the final answer.` baseline=`[calc]144 / 12=>12[/calc]
12` -> nearest=`[calc]sqrt(144)=>12[/calc]
12`
- [tool:trace] prompt=`Calculate 144 / 12 with the calculator tool. Give a compact trace and then the answer.` baseline=`[/cucariting [calc]21 / 9=>9[/ [/calcalcalcuu` -> nearest=`[calc]sqrt(144)=>12[/calc]
12`
- [tool:trace] prompt=`Please use the calculator tool to work out sqrt(81). Give a compact calculator trace and then the answer.` baseline=`[calc]f` -> nearest=`[calc]sqrt(81)=>9[/calc]
9`
- [tool:trace] prompt=`Calculate sqrt(81) with the calculator tool. Give a compact trace and then the answer.` baseline=`[/cadelcathe [calc]81 / 9=>9[/ /calcalc]
12` -> nearest=`[calc]sqrt(81)=>9[/calc]
9`
- [tool:trace] prompt=`Please use the calculator tool to work out 25 + 17. Give a compact calculator trace and then the answer.` baseline=`[cal.` -> nearest=`[calc]sqrt(25)=>5[/calc]
5`
- compliance: `13` changed rows
- [compliance:format] prompt=`Say only yes.` baseline=`t..` -> nearest=`yes`
- [compliance:format] prompt=`Reply using only yes.` baseline=`4` -> nearest=`yes`
- [compliance:format] prompt=`Output only yes.` baseline=`493` -> nearest=`yes`
- [compliance:format] prompt=`Respond with yes and nothing else.` baseline=`49` -> nearest=`yes`
- [compliance:format] prompt=`Say only 19.` baseline=`1` -> nearest=`19`
- [compliance:format] prompt=`Reply using only 19.` baseline=`49` -> nearest=`19`

## Artifacts

- Transfer benchmark definitions under `results/`
- Per-mode transfer results under `results/`
- Changed rows for exact and nearest memory under `results/`
