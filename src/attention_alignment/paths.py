from __future__ import annotations

from pathlib import Path

from .errors import ConfigurationError


class PathPolicy:
    """Resolve generated paths while enforcing the project output boundary."""

    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root).resolve()

    def resolve_output(self, *parts: str) -> Path:
        candidate = self.output_root.joinpath(*parts).resolve()
        if candidate != self.output_root and self.output_root not in candidate.parents:
            raise ConfigurationError(f"Refusing output outside project: {candidate}")
        return candidate

    def ensure_output_dir(self, *parts: str) -> Path:
        candidate = self.resolve_output(*parts)
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate


def inventory_signature(paths: list[Path]) -> list[dict[str, int | str]]:
    """Return size/mtime metadata used to prove source directories stayed read-only."""

    rows: list[dict[str, int | str]] = []
    for root in paths:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            stat = path.stat()
            rows.append(
                {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return rows
