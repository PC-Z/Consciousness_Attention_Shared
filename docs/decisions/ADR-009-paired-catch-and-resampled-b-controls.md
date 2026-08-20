# ADR-009: Pair Catch Trials with Preceding B and Resample the B Pool

## Status

Accepted

## Date

2026-08-17

## Context

Test P5 contains substantially more B trials than A or C trials. A single seeded
sample of ten B trials makes the A/B/C comparison depend on one arbitrary draw and
does not distinguish the B context immediately preceding an A catch from the B
context immediately preceding a C catch.

Peterka et al. (2026) balanced conditions to the smallest trial count and preferred
temporally adjacent reference trials before catch trials. Zhou et al. (2025)
addressed an imbalanced outcome comparison by drawing an equal-size subset from the
larger group 500 times and averaging the derived metric. Zhou et al. did not report
per-trial inclusion probabilities or 95% intervals for those draws.

## Decision

For the primary Notebook 04b P5 comparison, pair every test P5-A with the nearest
preceding valid test P5-B and label that reference set `B_A`. Pair every test P5-C
analogously and label it `B_C`. Do not reuse a B within one comparison; allow the
same B to occur once in each comparison when A and C catches are adjacent. Require
every catch to have a valid preceding B; fail visibly rather than silently replacing
unmatched trials. Preserve source event indices, sequence indices, pair indices, and
onset gaps in the exported pair table.

Use the same fixed neuron rows for A versus B_A and C versus B_C. Default this new
comparison to all atlas acronyms beginning with `VIS`, including VISp, VISa, VISam,
VISl, VISpm, and the other recorded VIS subdivisions; keep an explicit all-cell
override. Compute responses with aligned frame timestamps. For 1000-ms sessions use -1 to 0 seconds as baseline,
0 to 2 seconds as the quantitative response, and -1 to 3 seconds only for P5 trace
display. For 500-ms sessions use -1 to 0, 0 to 1.5, and -1 to 2 seconds,
respectively. Keep the inherited SD-threshold screen separate from this matched
fixed-population comparison.

As a sensitivity analysis, independently sample from the complete test P5-B pool at
the A count and at the C count without replacement, repeating each comparison 500
times with a recorded seed. Export every draw, each B trial's inclusion probability,
the repeat-level population effect, and per-neuron effect distributions. Report the
2.5th and 97.5th percentiles as empirical trial-resampling intervals. These are a
project extension for auditability and are not attributed to Zhou et al.

## Alternatives Considered

### Reuse one B sample for A and C

Rejected because one sample cannot represent both catch-local contexts and makes the
result sensitive to one random draw.

### Use all B trials against ten A or C trials

Rejected as the primary comparison because trial-count imbalance can change response
variance, visual weighting, and neuron eligibility.

### Treat resampling percentiles as biological confidence intervals

Rejected because they measure sensitivity to B-trial selection within one session.
They do not represent uncertainty across mice.

## Consequences

- `B_A` and `B_C` are distinct, auditable reference sets for the main comparisons.
- The 500-repeat analysis shows whether conclusions depend on which B trials are
  chosen from the larger pool.
- The main comparison remains observational and within one mouse; neuron-level
  p-values are not mouse-level replication.
- Quantitative response windows no longer include the long post-stimulus tail that
  was present in the legacy 0-to-6-second configuration.

## References

- Peterka et al. (2026), *Global context rapidly shapes sensory responses in V1*:
  <https://www.biorxiv.org/content/10.64898/2026.01.07.698143v1>
- Zhou et al. (2025), *Neural correlates of trial outcome monitoring during
  long-term learning in primate posterior parietal cortex*:
  <https://www.nature.com/articles/s41467-025-66714-8>
