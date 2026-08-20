# ADR-011: Code-only collaboration repository with separate data storage

## Status

Accepted

## Date

2026-08-19

## Context

The project needs multiple collaborators to review and extend notebooks 01-04,
but the full local project contains session-specific ROI polygons, manual pupil
anchors, local paths, historical outputs, and experiment-specific configuration.
Raw videos and calcium matrices are also too large and sensitive for an ordinary
GitHub repository.

The collaboration workflow therefore needs to separate code versioning from
authorized data access while retaining reproducible session metadata.

## Decision

Maintain a private, sanitized GitHub repository containing code, notebooks,
templates, tests, documentation, and release tags. Store raw data, stimulus
videos, and derived results in a separate controlled cloud repository. Connect
the two through collaborator-local configuration files and a metadata-only data
manifest with relative paths, file sizes, and hashes.

The complete current project remains a private archive and is not pushed as the
collaborator repository. Real `sessions.yaml` and `rois.yaml` remain local.

## Alternatives considered

### Push the current repository directly

Rejected because the current history and tracked configuration expose project
context, real session identifiers, ROI coordinates, manual anchors, and local
paths. A GitHub pull request does not provide data access isolation.

### Cloud ZIP only

Useful for an immediate release, but rejected as the long-term source of truth:
parallel edits and version comparison become difficult.

### Put raw data in Git LFS

Rejected for the current experiment because videos and calcium matrices are
large, access-controlled, and not code-version artifacts. A separate storage
system is easier to permission and back up.

## Consequences

- Collaborators can review and merge code without receiving unauthorized data.
- Each collaborator must maintain a private local config and have the correct
  cloud-data permission.
- Release notebooks need an explicit local-config parameter rather than relying
  on the current repository-default config.
- A lock file and small synthetic fixtures should be added before the first
  external release.
- Selected results must record the code tag and data-manifest version to remain
  traceable.
