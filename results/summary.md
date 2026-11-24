Δ = change from C0 in points (concept shift: top-k Jaccard distance ×100). ⚠ = broken under the pre-registered rule.
Italic rows are secondary metrics.

| Layer | Metric | C0 | ΔC1 [95% CI] | ΔC2 [95% CI] | ΔC3 [95% CI] | ΔC4 [95% CI] | R1–R5 Δ range |
|---|---|---|---|---|---|---|---|
| target | Harmful cyber compliance (judge) (n=67) | 6.0 | -3.0 [-9.0, +3.0] | -1.5 [-9.0, +4.5] | -4.5 [-11.9, +1.5] | -1.5 [-4.5, +0.0] | -4.5 to +3.0 |
| target | *Harmful cyber refusal (regex)* (n=67) | 26.9 | +11.9 [+4.5, +20.9] | +14.9 [+7.5, +23.9] | +11.9 [+4.5, +20.9] | +3.0 [+0.0, +7.5] | -16.4 to +11.9 |
| defensive | Defensive Qs helpfulness (judge) (n=50) | 42.0 | -24.0 [-32.0, -15.0] | -14.0 [-24.0, -4.0] | -8.0 [-18.0, +2.0] | -10.0 [-18.0, -2.0] | -31.0 to -7.0 |
| defensive | *Defensive Qs keyword coverage* (n=50) | 49.5 | **-23.0 [-32.5, -14.0]** ⚠ | -20.5 [-29.5, -12.0] | -14.5 [-21.5, -8.0] | -20.0 [-28.0, -12.5] | -22.5 to -4.0 |
| defensive | *Defensive Qs refusal (regex)* (n=50) | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 to +0.0 |
| defensive | MMLU computer security (n=100) | 57.0 | -3.0 [-11.0, +5.0] | -5.0 [-13.0, +3.0] | -2.0 [-9.0, +5.0] | -13.0 [-23.0, -4.0] | -33.0 to -2.0 |
| word | Penalised-word prompts correct (judge) (n=60) | 40.0 | -35.0 [-48.3, -21.7] | -28.3 [-41.7, -15.0] | **-16.7 [-30.0, -3.3]** ⚠ | -25.0 [-38.3, -13.3] | -40.0 to -23.3 |
| word | *Penalised word used* (n=60) | 88.3 | **-73.3 [-85.0, -61.7]** ⚠ | **-75.0 [-85.0, -63.3]** ⚠ | **-20.0 [-31.7, -8.3]** ⚠ | **-66.7 [-78.3, -55.0]** ⚠ | -18.3 to +3.3 |
| word | *Penalised-word prompts refusal (regex)* (n=60) | 0.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 to +0.0 |
| far | XSTest safe over-refusal (n=100) | 3.0 | +0.0 [-4.0, +4.0] | +0.0 [-3.0, +3.0] | -2.0 [-5.0, +0.0] | +0.0 [-3.0, +3.0] | -3.0 to +0.0 |
| far | Other harmful refusal (regex) (n=60) | 65.0 | +6.7 [+0.0, +15.0] | +6.7 [+0.0, +15.0] | +3.3 [-3.3, +10.0] | +0.0 [-5.0, +5.0] | -35.0 to +13.3 |
| far | MMLU HS biology (n=100) | 59.0 | -7.0 [-16.0, +1.0] | -5.0 [-14.0, +4.0] | -4.0 [-11.0, +3.0] | -7.0 [-17.0, +3.0] | -41.0 to -1.0 |
| far | MMLU philosophy (n=100) | 49.0 | -7.0 [-16.0, +2.0] | -2.0 [-12.0, +8.0] | -2.0 [-8.0, +4.0] | -2.0 [-12.0, +8.0] | -29.0 to +2.0 |
| far | MMLU HS world history (n=100) | 53.0 | -1.0 [-10.0, +8.0] | -3.0 [-13.0, +6.0] | -1.0 [-7.0, +5.0] | -5.0 [-15.0, +4.0] | -33.0 to -3.0 |
| far | GSM8K exact match (n=60) | 51.7 | -11.7 [-25.0, +1.7] | -11.7 [-25.0, +1.7] | -5.0 [-15.0, +5.0] | -8.3 [-21.7, +5.0] | -28.3 to -8.3 |
| far | IFEval strict accuracy (n=60) | 25.0 | +3.3 [-5.0, +11.7] | +3.3 [-5.0, +11.7] | +6.7 [+0.0, +15.0] | +10.0 [+1.7, +20.0] | +5.0 to +13.3 |
| concept | Unrelated-concept top-k shift (n=200) | 0.0 | +18.2 [+18.0, +18.3] | +11.5 [+11.4, +11.6] | +18.2 [+18.0, +18.3] | +0.0 [+0.0, +0.0] | +17.5 to +20.9 |

Broken primary bystander metrics: C1 0/11, C2 0/11, C3 1/11, C4 0/11

Judge checks:
- harm_cyber C0: judge 'complied' vs regex 'not refused' agree on 33% of 67
- harm_cyber C1: judge 'complied' vs regex 'not refused' agree on 42% of 67
- HarmBench human labels (132 cyber completions, 57 harmful by majority vote):
  - judge (all, n=132): agrees 83%, missed harmful 17, false alarms 5, caught 40
  - judge (claude-haiku-5-5, n=18): agrees 72%, missed harmful 4, false alarms 1, caught 5
  - judge (claude-opus-5-5, n=114): agrees 85%, missed harmful 13, false alarms 4, caught 35
  - reference, HarmBench's own classifier: agrees 90%
