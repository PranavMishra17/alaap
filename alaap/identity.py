"""
The identity store.

This is the contract between the ML track and the platform track
(GRAND-PLAN section 3), and the thing that must not change late.

Invariant I2, in code:

    EVERY identity stores BOTH a Tier-1 vector AND a Tier-2 seed clip,
    plus backend_version and the full generation parameters.

Why both, always -- not "Tier 1 with Tier 2 as fallback":

  * The Zonos v0.1 -> ZONOS2 encoder change (256 -> 2048 dims, LDA 128 ->
    1024) orphaned every stored Tier-1 vector in 16 months. The waveform is
    what survives a version bump.
  * 100% of shipping commercial voice-design products are Tier 2
    (RESEARCH/11). Cartesia shipped Tier-1 192-d vectors with weighted
    mixing and withdrew the capability entirely on 2026-06-01.
  * Tier 3 (description + seed) is dead, not merely weak: VoxCPM2's
    retry_badcase (default True) silently does current_seed += 1, so a seed
    is not reproducible even on one machine.

E4 adds a fourth field that turns out to be load-bearing: `space_ref`.
A Tier-1 vector is meaningless without the SpeakerSpace transform that
produced it, and that transform is a per-DOMAIN artefact (an in-domain fit
halves EER vs a cross-domain one). Storing the vector without the space is
the same class of mistake as storing it without the backend version.

Backed by SQLite locally. The schema is deliberately Postgres+pgvector
shaped so S6 is a driver swap, not a migration.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS identity (
    identity_id      TEXT PRIMARY KEY,
    character_id     TEXT NOT NULL,
    language         TEXT NOT NULL,
    description      TEXT NOT NULL,

    -- Tier 1
    embedding        BLOB,          -- float32 little-endian
    embedding_dim    INTEGER,
    space_ref        TEXT,          -- SpeakerSpace id; a vector is meaningless without it

    -- Tier 2 (ALWAYS populated -- invariant I2)
    seed_audio_ref   TEXT NOT NULL,
    seed_audio_sha   TEXT NOT NULL,
    seed_audio_sr    INTEGER,

    -- reproduction recipe (Hume's precedent: speech AND prompt AND params)
    generation_params TEXT NOT NULL,

    -- provenance / migration safety
    backend_id       TEXT NOT NULL,
    backend_version  TEXT NOT NULL,
    identity_tier    INTEGER NOT NULL,
    minted_at        TEXT NOT NULL,

    -- search
    desc_embedding   BLOB,
    tags             TEXT,

    schema_version   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_identity_character ON identity(character_id);
CREATE INDEX IF NOT EXISTS idx_identity_language  ON identity(language);
CREATE INDEX IF NOT EXISTS idx_identity_backend   ON identity(backend_id, backend_version);

CREATE TABLE IF NOT EXISTS render_log (
    render_id        TEXT PRIMARY KEY,
    identity_id      TEXT NOT NULL,
    text             TEXT NOT NULL,
    direction        TEXT,
    backend_id       TEXT NOT NULL,
    backend_version  TEXT NOT NULL,
    audio_ref        TEXT,
    watermarked      INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_render_identity ON render_log(identity_id);
"""


def _blob(a: Optional[np.ndarray]) -> Optional[bytes]:
    return None if a is None else np.asarray(a, dtype=np.float32).tobytes()


def _unblob(b: Optional[bytes]) -> Optional[np.ndarray]:
    return None if b is None else np.frombuffer(b, dtype=np.float32)


@dataclass
class Identity:
    character_id: str
    language: str
    description: str
    seed_audio_ref: str
    seed_audio_sha: str
    backend_id: str
    backend_version: str
    generation_params: dict
    embedding: Optional[np.ndarray] = None
    space_ref: Optional[str] = None
    seed_audio_sr: int = 24000
    desc_embedding: Optional[np.ndarray] = None
    tags: list[str] = field(default_factory=list)
    identity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    minted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def tier(self) -> int:
        """1 if a usable vector is present, else 2. Tier 3 does not exist."""
        return 1 if (self.embedding is not None and self.space_ref) else 2

    def __post_init__(self):
        # I2 enforcement: a seed clip is mandatory, no exceptions.
        if not self.seed_audio_ref or not self.seed_audio_sha:
            raise ValueError(
                "invariant I2: every identity must carry a Tier-2 seed clip. "
                "A vector alone does not survive a backend version bump.")
        if self.embedding is not None and not self.space_ref:
            raise ValueError(
                "a Tier-1 vector requires a space_ref -- the SpeakerSpace that "
                "produced it. E4: the transform is a per-domain artefact.")


class IdentityStore:
    def __init__(self, path: str = "data/identities.db"):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(DDL)
        self.con.commit()

    # ------------------------------------------------------------------ write
    def put(self, idn: Identity) -> str:
        self.con.execute(
            """INSERT OR REPLACE INTO identity VALUES
               (:identity_id,:character_id,:language,:description,
                :embedding,:embedding_dim,:space_ref,
                :seed_audio_ref,:seed_audio_sha,:seed_audio_sr,
                :generation_params,:backend_id,:backend_version,
                :identity_tier,:minted_at,:desc_embedding,:tags,:schema_version)""",
            {"identity_id": idn.identity_id, "character_id": idn.character_id,
             "language": idn.language, "description": idn.description,
             "embedding": _blob(idn.embedding),
             "embedding_dim": None if idn.embedding is None else int(len(idn.embedding)),
             "space_ref": idn.space_ref,
             "seed_audio_ref": idn.seed_audio_ref, "seed_audio_sha": idn.seed_audio_sha,
             "seed_audio_sr": idn.seed_audio_sr,
             "generation_params": json.dumps(idn.generation_params),
             "backend_id": idn.backend_id, "backend_version": idn.backend_version,
             "identity_tier": idn.tier, "minted_at": idn.minted_at,
             "desc_embedding": _blob(idn.desc_embedding),
             "tags": json.dumps(idn.tags), "schema_version": SCHEMA_VERSION})
        self.con.commit()
        return idn.identity_id

    def log_render(self, identity_id: str, text: str, backend_id: str,
                   backend_version: str, audio_ref: str | None = None,
                   direction: dict | None = None, watermarked: bool = False) -> str:
        """
        Invariant I7: every render is provenance-logged, from render #1.
        EU AI Act Art. 50(2) has applied since 2026-08-02.
        """
        rid = str(uuid.uuid4())
        self.con.execute(
            "INSERT INTO render_log VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, identity_id, text, json.dumps(direction) if direction else None,
             backend_id, backend_version, audio_ref, int(watermarked),
             datetime.now(timezone.utc).isoformat()))
        self.con.commit()
        return rid

    # ------------------------------------------------------------------- read
    def get(self, identity_id: str) -> Optional[Identity]:
        r = self.con.execute("SELECT * FROM identity WHERE identity_id=?",
                             (identity_id,)).fetchone()
        return self._row(r) if r else None

    def by_character(self, character_id: str) -> list[Identity]:
        rs = self.con.execute("SELECT * FROM identity WHERE character_id=?",
                              (character_id,)).fetchall()
        return [self._row(r) for r in rs]

    def all(self, language: str | None = None) -> list[Identity]:
        q, p = "SELECT * FROM identity", ()
        if language:
            q += " WHERE language=?"; p = (language,)
        return [self._row(r) for r in self.con.execute(q, p).fetchall()]

    @staticmethod
    def _row(r: sqlite3.Row) -> Identity:
        idn = Identity(
            character_id=r["character_id"], language=r["language"],
            description=r["description"], seed_audio_ref=r["seed_audio_ref"],
            seed_audio_sha=r["seed_audio_sha"], backend_id=r["backend_id"],
            backend_version=r["backend_version"],
            generation_params=json.loads(r["generation_params"]),
            embedding=_unblob(r["embedding"]), space_ref=r["space_ref"],
            seed_audio_sr=r["seed_audio_sr"],
            desc_embedding=_unblob(r["desc_embedding"]),
            tags=json.loads(r["tags"] or "[]"))
        idn.identity_id = r["identity_id"]
        idn.minted_at = r["minted_at"]
        return idn

    # -------------------------------------------------------------- migration
    def version_audit(self, backend_id: str, new_version: str) -> dict:
        """
        PHASE-01 section 3: a backend upgrade is a MIGRATION, never an in-place
        swap. This reports which identities were minted against an older
        version and therefore need a re-render + diff before the upgrade is
        allowed to touch them.

        Getting this wrong silently changes every shipped character's voice,
        which is unrecoverable for anyone who already shipped a game.
        """
        rows = self.con.execute(
            "SELECT backend_version, identity_tier, COUNT(*) n FROM identity "
            "WHERE backend_id=? GROUP BY backend_version, identity_tier",
            (backend_id,)).fetchall()
        stale = [dict(r) for r in rows if r["backend_version"] != new_version]
        return {"backend_id": backend_id, "new_version": new_version,
                "affected": sum(r["n"] for r in stale),
                "needs_rerender_and_diff": stale,
                "tier1_at_risk": sum(r["n"] for r in stale if r["identity_tier"] == 1),
                "tier2_recoverable": sum(r["n"] for r in stale if r["identity_tier"] == 2)}

    def stats(self) -> dict:
        c = self.con.execute
        return {
            "identities": c("SELECT COUNT(*) n FROM identity").fetchone()["n"],
            "characters": c("SELECT COUNT(DISTINCT character_id) n FROM identity").fetchone()["n"],
            "languages": [r[0] for r in c("SELECT DISTINCT language FROM identity")],
            "tier1": c("SELECT COUNT(*) n FROM identity WHERE identity_tier=1").fetchone()["n"],
            "tier2": c("SELECT COUNT(*) n FROM identity WHERE identity_tier=2").fetchone()["n"],
            "renders": c("SELECT COUNT(*) n FROM render_log").fetchone()["n"],
            "renders_watermarked": c("SELECT COUNT(*) n FROM render_log WHERE watermarked=1").fetchone()["n"],
        }

    def close(self):
        self.con.close()


def sha256_audio(wav: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(wav, dtype=np.float32).tobytes()).hexdigest()
