# Collaboration and Data Sharing

This project uses a private code repository and a separate controlled data
repository. The code repository is safe to clone; the data repository is shared
only with collaborators who are authorized to access the corresponding sessions.
The data repository may contain approved raw data and approved derived artifacts
such as alignment indexes, behavior Parquet files, pupil analysis outputs, and
ROI/manual-anchor annotations.

## Repository boundary

The code repository contains Python modules, notebooks 01-04, configuration
templates, tests, documentation, and release metadata. It must not contain raw
videos, calcium matrices, electrophysiology files, stimulus videos, derived
outputs, real ROI coordinates, manual pupil anchors, or local absolute paths.

The data repository contains session directories, stimulus files, and the
authorized analysis state. Keep it on the approved cloud drive and record
metadata in a copy of `docs/data-manifest.template.yaml`. The manifest may
contain file hashes and repository-relative paths, but no credentials or local
absolute paths.

Do not share the whole current `configs/rois.yaml` when it contains sessions the
collaborator is not authorized to inspect. Export only the approved session
entries, preserving all ROI vertices, landmarks, pupil seeds, and manual pupil
anchors. Keep a versioned copy rather than overwriting the canonical annotation.

`04b_neural_activity_overview.ipynb` is an optional legacy exploratory notebook.
It should be published under a clearly marked `legacy/` directory rather than
being treated as the primary 04 workflow.

## Cloud data layout

Use one directory per authorized session. The following layout is the minimum
for continuing the current workflow:

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
│   └── valid_area_5.mat              # only when 04 is in scope
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
    └── neural_analysis/              # only when 04 is in scope
```

This mirrors the current project layout: `data_root` points to `data/`, while
alignment and downstream results are copied into the local
`alignment_pipeline/outputs/<session_id>/` directory before running 03/04. The
`outputs/<session_id>/` cloud folder should include the existing
`calcium_frames.parquet`, `stimulus_events.parquet`, `video_frames_01.parquet`,
`video_frames_02.parquet`, `manifest.json`, `qc.json`, and
`ephys_02_manifest.json` when available. It should also include the latest
reviewed `behavior_*.parquet`.

The pupil directory should include `analysis_parameters.json`, QC tables,
trace/metric Parquet or CSV files, figures, and the resampling outputs. For 04,
also provide the configured calcium matrix, denoised trace, atlas MAT file, and
the aligned `calcium_frames.parquet`.

Raw videos remain necessary for visual QC and further ROI/manual-anchor edits;
derived Parquet files alone are not sufficient for that purpose.

## Local setup

1. Clone the private code repository.
2. Copy `configs/sessions.template.yaml` to `configs/sessions.local.yaml`.
3. Copy `configs/rois.template.yaml` to `configs/rois.local.yaml` as the
   aggregate fallback, and create `configs/rois/<session_id>.yaml` for each
   annotated session.
4. Set the local data and stimulus roots, then add only authorized sessions.
5. Run `setup_env.ps1` and start Jupyter with `start_jupyter.ps1`.

The release notebooks load the local session configuration explicitly. ROI
loading prefers `configs/rois/<session_id>.yaml` and falls back to
`configs/rois.local.yaml`; P9/P13 migrate only the requested session on first
save and never overwrite an existing session file automatically. Keep local
configuration and per-session ROI files untracked.

## Analysis/output boundary

Run notebooks in this order:

```text
01 alignment -> 02 pupil/behavior -> 03 pupil stimulus analysis
01 alignment -> 04 calcium overview and neural analysis
```

Outputs remain in the local `outputs/<session_id>/` directory and are not
committed to the code repository. The owner copies authorized alignment,
behavior, pupil, and neural outputs to the corresponding cloud session folder.
Each shared result should carry the code tag, manifest version, configuration
hash, session ID, and date.

## Two-repository Git workflow

The existing `Consciousness_Attention` repository remains the private full
analysis archive. Create a second private repository, for example
`Consciousness_Attention-code`, containing only the sanitized release. This is a
privacy boundary, not a requirement to mirror every local commit.

Use the full archive for day-to-day personal work. Publish only stable,
shareable changes to the code repository as a release commit or tag. Pull
Requests from collaborators are reviewed in the code repository and then
cherry-picked or manually ported back to the full archive when appropriate.
Likewise, do not automatically copy every exploratory local change into the
shared repository. A one-way release process avoids two live copies of every
Notebook while keeping the full history private.

GitHub permissions are repository-wide; branch protection or sparse checkout
does not hide other branches or historical files. Therefore a separate private
repository is required when collaborators must not see the full archive.

## Git workflow

- `main` is the reviewed branch and is protected from direct pushes.
- Use `analysis/<topic>` or `feature/<topic>` branches.
- Open a Pull Request with a short description, affected notebooks/modules,
  validation performed, and whether outputs changed.
- Create a version tag for stable analysis releases, for example
  `v0.3.0-peterka-neural`.
- Never commit `*.local.yaml`, raw data, videos, calcium matrices, outputs, or
  notebook execution dumps.

## Release checklist

Before publishing a code release, work from an isolated clean copy and verify:

- no tracked file contains `D:\`, `C:\Users\`, `analysis_data`, or a real local
  session path;
- notebook outputs and execution counts are cleared;
- only template configs are present;
- `git ls-files` contains no raw-data extension or output artifact;
- `python -m pytest` passes the synthetic/unit tests;
- the exact Python/dependency lock information is included;
- the release README states which data files are required for each notebook.

The current complete project remains a private analysis archive. Do not derive
the collaborator repository by pushing the full history of that archive.
