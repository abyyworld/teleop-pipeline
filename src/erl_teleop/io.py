"""Read/write helpers for the canonical episode store.

One episode = one parquet file (timeseries) + one JSON sidecar (metadata).
Parquet rather than HDF5 because DVC dedupes it well, pandas/pyarrow read it
without a C extension dance, and it stays columnar for the stats passes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .schema import EpisodeMeta, SessionMeta


def content_hash(df: pd.DataFrame) -> str:
    """Stable sha256 of the timeseries content.

    Used as the episode's identity for dedupe and for lineage. Hashing the
    parquet bytes directly would be unstable — pyarrow embeds a writer version
    and compression can differ between machines — so hash the sorted-column
    values instead.
    """
    h = hashlib.sha256()
    for col in sorted(df.columns):
        h.update(col.encode())
        h.update(df[col].to_numpy().tobytes())
    return h.hexdigest()


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_session_meta(session_dir: Path, meta: SessionMeta) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "meta.json"
    path.write_text(meta.model_dump_json(indent=2))
    return path


def read_session_meta(session_dir: Path) -> SessionMeta:
    return SessionMeta.model_validate_json((Path(session_dir) / "meta.json").read_text())


def write_episode(session_dir: Path, meta: EpisodeMeta, df: pd.DataFrame) -> Path:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    data_path = session_dir / f"{meta.episode_id}.parquet"
    df.to_parquet(data_path, index=False, compression="zstd")
    (session_dir / f"{meta.episode_id}.meta.json").write_text(meta.model_dump_json(indent=2))
    return data_path


def read_episode(session_dir: Path, episode_id: str) -> tuple[EpisodeMeta, pd.DataFrame]:
    session_dir = Path(session_dir)
    meta = EpisodeMeta.model_validate_json((session_dir / f"{episode_id}.meta.json").read_text())
    df = pd.read_parquet(session_dir / f"{episode_id}.parquet")
    return meta, df


def read_episode_meta(meta_path: Path) -> EpisodeMeta:
    return EpisodeMeta.model_validate_json(Path(meta_path).read_text())


def iter_episode_metas(episode_root: Path) -> Iterator[EpisodeMeta]:
    """Yield every episode's metadata under the store, in a stable order."""
    for meta_path in sorted(Path(episode_root).glob("*/*.meta.json")):
        yield read_episode_meta(meta_path)


def iter_episodes(episode_root: Path) -> Iterator[tuple[EpisodeMeta, pd.DataFrame]]:
    for meta_path in sorted(Path(episode_root).glob("*/*.meta.json")):
        meta = read_episode_meta(meta_path)
        yield meta, pd.read_parquet(meta_path.parent / f"{meta.episode_id}.parquet")


def episode_path(episode_root: Path, meta: EpisodeMeta) -> Path:
    return Path(episode_root) / meta.session_id / f"{meta.episode_id}.parquet"


def update_episode_meta(episode_root: Path, meta: EpisodeMeta) -> Path:
    path = Path(episode_root) / meta.session_id / f"{meta.episode_id}.meta.json"
    path.write_text(meta.model_dump_json(indent=2))
    return path


def write_json(path: Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
