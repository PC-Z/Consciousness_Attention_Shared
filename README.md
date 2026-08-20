# Attention Alignment Shared Code

This repository contains the sanitized Python package and Notebooks 01-04 for
the attention oddball analysis. It does not contain raw data, stimulus videos,
calcium matrices, real ROI coordinates, manual pupil anchors, or derived
output. Authorized data and the current analysis state are distributed
separately in the controlled data directory described in
`docs/shared-data/README.md`.

## Quick start

```powershell
.\setup_env.ps1
.\start_jupyter.ps1
```

Copy the configuration templates before opening a notebook:

```powershell
Copy-Item configs/sessions.template.yaml configs/sessions.local.yaml
Copy-Item configs/rois.template.yaml configs/rois.local.yaml
New-Item -ItemType Directory -Force configs/rois
```

Set the local `data_root`, `stimuli_root`, and authorized session list. Keep
the aggregate `configs/rois.local.yaml` as a fallback, and put reviewed
annotations in `configs/rois/<session_id>.yaml`; the session-specific file is
preferred automatically. Both local configuration paths are ignored by Git.

Run the notebooks in this order:

```text
01 alignment -> 02 pupil/behavior -> 03 pupil stimulus analysis
01 alignment -> 04 calcium overview and neural analysis
```

## Minimum environment

Python 3.11 is required. Core dependencies are declared in `pyproject.toml`;
Jupyter dependencies are in the `notebook` extra and test dependencies in the
`dev` extra. `environment.yml` is provided as a Conda alternative.

## Collaboration rules

- `main` is reviewed through Pull Requests.
- Use `analysis/<topic>` or `feature/<topic>` branches.
- Never commit `*.local.yaml`, per-session ROI YAML files, data, videos,
  calcium matrices, outputs, or notebook execution dumps.
- Every result report records the code tag, session ID, data-manifest version,
  and annotation version.

The owner's complete analysis archive is maintained separately. This
repository receives stable, shareable code releases only. See
`docs/collaboration.md` for the release and data workflow.
