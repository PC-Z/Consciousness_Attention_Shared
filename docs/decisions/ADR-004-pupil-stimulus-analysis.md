# ADR-004: Use Auditable Train/Test Pupil Windows

## Status

Superseded by ADR-010

The archived implementation remains available as
`archive/pupil03-fractional-v1.0.0` and tag
`pupil03-fractional-v1.0.0`.

## Date

2026-08-15

## Context

The experiment contains 100 AAAAB train sequences followed by 100 test sequences.
The test phase contains predictable local-deviant AAAAB, global-deviant AAAAA, and
local-plus-global-deviant AAAAC sequences. Condition labels `500 ms` and `1000 ms`
do not equal the measured stripe durations: their confirmed stripe/gray pairs are
1.0/0.5 seconds and 2.0/1.0 seconds respectively.

Westerberg et al. used P4-minus-P3 for a four-item sequence. The present design has
five items and places the critical X at P5, so its corresponding within-sequence
contrast is P5-minus-P4. Peterka et al. compared the first and last seven repeats of
training, while exploratory review of 100 trials benefits from configurable bins.

## Decision

Infer windows from exported event metadata and use measured onsets as time zero.
Single-item windows include one within-sequence gray interval before onset and the
full stimulus duration. Full-sequence windows include the same pre-onset baseline and
all five stimuli with four internal gray intervals.

Use equivalent pupil diameter as the primary geometry and compute trial-baseline
fractional change. Also retain session z-scores for the literature-style P5-minus-P4
trace. Do not interpolate across gaps longer than 0.5 seconds. A trial is accepted
only when baseline and response coverage pass an explicit threshold.

Keep `TRIALS_PER_GROUP` as a notebook parameter used by all grouped train/test plots.
Show gray individual trials with a black mean, and show full-sequence first/last group
means with a selectable SEM or 95% confidence error band. Preserve the first-7 versus
last-7 comparison and add first-30 versus last-30 as a sensitivity analysis.

Analyze test AAAAB, AAAAA, and AAAAC separately. For P5-minus-P4, align P4 and P5 to
their own onsets and subtract matched relative-time samples within the same sequence.
Retain chronological test indices so rare conditions are not presented as contiguous
when they were interleaved.

Save derived traces, metrics, QC, parameters, and figures beneath the session output
directory. Never write to source `data`, `analysis`, or `stimuli` directories.

## Alternatives Considered

### Fix every plot to five-trial groups

Rejected because five trials can be too noisy for pupil measurements. A single group
parameter provides visual sensitivity checks without changing scientific windows.

### Compare only the first 30 and last 30 trials

Rejected as the sole analysis because it discards 40 trials and does not match the
paper figure. It remains a prespecified sensitivity analysis.

### Treat P5 responses as independent of P4

Insufficient for the literature-style contrast because P4 provides the immediately
preceding, position-matched response within the same sequence. Direct P5 traces are
still retained because pupil responses are slow and subtraction is not a complete
physiological model.

## Consequences

- Group size changes plotting aggregation, not event extraction or trial identity.
- Single-mouse results are descriptive; population inference must use mouse as the
  independent unit after all sessions pass QC.
- Phase and trial coverage failures stop analysis instead of producing sparse curves.
- All numerical figure inputs remain available as Parquet/CSV outputs.
