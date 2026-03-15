# Memory Transfer Session: stress-tool-minimal-v2

## Command

`ava session memory-transfer stress-tool-minimal-v2 D:/AVA/sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt D:/AVA/corpora/tool_memory_minimal_v2 --device cuda --max-new-tokens 48 --nearest-threshold 0.45 --nearest-margin 0.0 --suite stress`

## Inputs

- Checkpoint: `D:/AVA/sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt`
- Support corpus: `D:/AVA/corpora/tool_memory_minimal_v2`
- Support files: `3`
- Support examples: `21`
- Transfer suite: `stress`
- Nearest threshold: `0.45`
- Nearest margin: `0.0`
- Category gated: `True`

## Transfer Scores

- Baseline: benchmark `6/40`, tool `6/31`, compliance `5/16`
- Exact direct: benchmark `19/40`, tool `19/31`, compliance `15/16`
- Nearest direct: benchmark `40/40`, tool `25/31`, compliance `16/16`
- Winner: `nearest`

## Exact Direct Changes

- benchmark: `13` changed rows
- [benchmark:english] prompt=`Put this into standard English: The plan ain't done.` baseline=`Th that.` -> exact=`The plan is not finished yet.`
- [benchmark:english] prompt=`Make this shorter: The cat slept on the warm window sill.` baseline=`cat` -> exact=`The cat slept on the warm sill.`
- [benchmark:english] prompt=`Shorten this sentence: The cat slept on the warm window sill.` baseline=`Thep 3` -> exact=`The cat slept on the warm sill.`
- [benchmark:math] prompt=`Multiply 17 by 29.` baseline=`4 49` -> exact=`493`
- [benchmark:math] prompt=`Compute 17 times 29.` baseline=`49` -> exact=`493`
- [benchmark:math] prompt=`Find the product of 17 and 29.` baseline=`19` -> exact=`493`
- tool: `14` changed rows
- [tool:trace] prompt=`Please use the calculator tool to work out 144 / 12. Give a compact calculator trace and then the answer.` baseline=`[caledddddddddinnggggggrrinowit.` -> exact=`[calc]144 / 12=>12[/calc]
12`
- [tool:trace] prompt=`Calculate 144 / 12 with the calculator tool. Give a compact trace and then the answer.` baseline=`[/cucariting [calc]21 / 9=>9[/ [/calcalcalcuu` -> exact=`[calc]144 / 12=>12[/calc]
12`
- [tool:trace] prompt=`Please use the calculator tool to work out sqrt(81). Give a compact calculator trace and then the answer.` baseline=`[calc]f` -> exact=`[calc]sqrt(81)=>9[/calc]
9`
- [tool:trace] prompt=`Calculate sqrt(81) with the calculator tool. Give a compact trace and then the answer.` baseline=`[/cadelcathe [calc]81 / 9=>9[/ /calcalc]
12` -> exact=`[calc]sqrt(81)=>9[/calc]
9`
- [tool:trace] prompt=`Please use the calculator tool to work out 25 + 17. Give a compact calculator trace and then the answer.` baseline=`[cal.` -> exact=`[calc]25 + 17=>42[/calc]
42`
- [tool:trace] prompt=`Calculate 25 + 17 with the calculator tool. Give a compact trace and then the answer.` baseline=`wit.` -> exact=`[calc]25 + 17=>42[/calc]
42`
- compliance: `12` changed rows
- [compliance:format] prompt=`Say only yes.` baseline=`t..` -> exact=`yes`
- [compliance:format] prompt=`Reply using only yes.` baseline=`4` -> exact=`yes`
- [compliance:format] prompt=`Output only yes.` baseline=`493` -> exact=`yes`
- [compliance:format] prompt=`Respond with yes and nothing else.` baseline=`49` -> exact=`yes`
- [compliance:format] prompt=`Say only 19.` baseline=`1` -> exact=`19`
- [compliance:format] prompt=`Reply using only 19.` baseline=`49` -> exact=`19`

## Nearest Direct Changes

- benchmark: `34` changed rows
- [benchmark:english] prompt=`Put this into standard English: The plan ain't done.` baseline=`Th that.` -> nearest=`The plan is not finished yet.`
- [benchmark:english] prompt=`Rewrite in plain English: The plan ain't done.` baseline=`calculcalculcanot dd / / / / /culp /ckingth delc` -> nearest=`The plan is not finished yet.`
- [benchmark:english] prompt=`Turn this into plain English: The plan ain't done.` baseline=`with that.` -> nearest=`The plan is not finished yet.`
- [benchmark:english] prompt=`Say this in standard English: The plan ain't done.` baseline=`with that.` -> nearest=`The plan is not finished yet.`
- [benchmark:english] prompt=`Make this shorter: The cat slept on the warm window sill.` baseline=`cat` -> nearest=`The cat slept on the warm sill.`
- [benchmark:english] prompt=`Shorten this sentence: The cat slept on the warm window sill.` baseline=`Thep 3` -> nearest=`The cat slept on the warm sill.`
- tool: `21` changed rows
- [tool:trace] prompt=`Please use the calculator tool to work out 144 / 12. Give a compact calculator trace and then the answer.` baseline=`[caledddddddddinnggggggrrinowit.` -> nearest=`[calc]144 / 12=>12[/calc]
12`
- [tool:trace] prompt=`Calculate 144 / 12 with the calculator tool. Give a compact trace and then the answer.` baseline=`[/cucariting [calc]21 / 9=>9[/ [/calcalcalcuu` -> nearest=`[calc]144 / 12=>12[/calc]
12`
- [tool:trace] prompt=`Please use the calculator tool to work out sqrt(81). Give a compact calculator trace and then the answer.` baseline=`[calc]f` -> nearest=`[calc]sqrt(81)=>9[/calc]
9`
- [tool:trace] prompt=`Calculate sqrt(81) with the calculator tool. Give a compact trace and then the answer.` baseline=`[/cadelcathe [calc]81 / 9=>9[/ /calcalc]
12` -> nearest=`[calc]sqrt(81)=>9[/calc]
9`
- [tool:trace] prompt=`Please use the calculator tool to work out 25 + 17. Give a compact calculator trace and then the answer.` baseline=`[cal.` -> nearest=`[calc]25 + 17=>42[/calc]
42`
- [tool:trace] prompt=`Calculate 25 + 17 with the calculator tool. Give a compact trace and then the answer.` baseline=`wit.` -> nearest=`[calc]25 + 17=>42[/calc]
42`
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
