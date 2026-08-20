# Attention Experiment Shared Data

This directory contains authorized experimental data and derived analysis state
for collaborators working with the private `attention-alignment-code` repository.
It is intentionally separate from GitHub source code. Access is granted per
session; do not redistribute files outside the approved collaboration group.

## Layout

Keep the layout compatible with the analysis pipeline:

```text
attention-experiment-data/
├── stimuli/
│   ├── oddball_stimuli_train_test_500ms.mp4
│   ├── oddball_stimuli_train_test_1000ms.mp4
│   ├── test_stimuli_500ms_order_list.txt
│   └── order_list_1000ms.txt
├── data/<session_id>/
│   ├── eeg/01.txt
│   ├── eeg/02.txt
│   ├── eeg/01.mp4
│   ├── eeg/02.mp4
│   ├── wholebrain_output_after_rejection.mat
│   ├── denoise/whole_brain_trace_denoised_loose.npy
│   └── valid_area_5.mat
├── annotations/<session_id>/
│   └── rois_<date>_<author>.yaml
└── outputs/<session_id>/
    ├── calcium_frames.parquet
    ├── stimulus_events.parquet
    ├── video_frames_01.parquet
    ├── video_frames_02.parquet
    ├── behavior_01.parquet
    ├── behavior_02.parquet
    ├── pupil_analysis_v2_session_z/
    └── neural_analysis/
```

The denoised trace and `valid_area_5.mat` are required only when Notebook 04 is
within the collaborator's scope. Raw camera videos are required when a
collaborator must visually review or improve an ROI/manual pupil anchor.

## Files required by each notebook

### Notebook 01

Provide the two timestamp text files, both camera videos, the calcium source,
the stimulus order file and the corresponding reference stimulus video. To
continue from an existing alignment, also provide the session's
`outputs/<session_id>/` alignment files:

```text
stimulus_events.parquet
calcium_frames.parquet
video_frames_01.parquet
video_frames_02.parquet
manifest.json
qc.json
ephys_02_manifest.json
```

Files that are not present for a session should not be invented; record them as
missing in the manifest.

### Notebook 02

Provide `01.mp4` and/or `02.mp4`, the corresponding current annotation file, and
the aligned `video_frames_*.parquet`. The annotation file must preserve:

- eye and movement polygons;
- pupil seed polygon and reference frame;
- eye-adjacent landmarks;
- threshold settings;
- every `pupil_manual_anchors` polygon saved by P12/P13.

### Notebook 03

Provide the latest `behavior_*.parquet`, `stimulus_events.parquet`, and the
complete `pupil_analysis_v2_session_z/` directory, including parameters, QC,
train/test traces and metrics, neighboring-B pairing, resampling tables, and
figures. The older `pupil_analysis/` directory is optional historical output.

### Notebook 04

Provide `calcium_frames.parquet`, the configured calcium trace source, the atlas
MAT file, and the session's `neural_analysis/` output directory when existing
results need review. Calcium matrices and denoised traces must remain in the
controlled data store, never in the code repository.

## Annotation versioning

Do not overwrite a shared ROI file. Use one file per authorized session and
annotator/date, for example:

```text
annotations/<session_id>/
├── rois_owner_20260819.yaml
├── rois_collaboratorA_20260825.yaml
└── annotation_notes.md
```

The local pipeline prefers one ROI file per session:
`configs/rois/<session_id>.yaml`. If that file is absent, it falls back to the
aggregate `configs/rois.local.yaml`. P9/P13 create the per-session file on the
first save by copying only that session's entry from the aggregate file; an
existing per-session file is never overwritten automatically. This keeps
annotation changes isolated as more sessions and collaborators are added.

When a new ROI version is used, rerun P10 and P11 and save the resulting
behavior output under a clearly named analysis version. Record the annotation
filename, code tag, and rerun date in `annotation_notes.md` or the data
manifest. Keep `configs/rois.local.yaml` as a controlled fallback/migration
source rather than editing it for unrelated sessions.

## Provenance

Each shared file should be listed in `data-manifest.yaml` with:

- session ID and artifact kind;
- repository-relative path;
- file size and SHA256;
- code tag;
- annotation version, when relevant;
- date and author.

Never put passwords, cloud tokens, or local absolute paths in the manifest.
