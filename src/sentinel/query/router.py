"""Tier 9 -- the multi-modal query router.

A cheap pattern classifier in front of the backends, so the expensive path is
taken only when it is actually needed. "When was the last threat found?" is a
SQL query answered in single-digit milliseconds; sending it to a 3-billion-
parameter VLM would take seconds and answer worse.

Routing it correctly is itself worth demonstrating -- it shows the team knows
where the expensive component belongs.

    [1] STRUCTURED   dates, kinds, counts, "last"     -> SQLite       < 5 ms
    [2] IDENTITY     track ids, "who is"              -> OSNet gallery
    [3] GROUNDING    "where in this frame"            -> LocateAnything-3B, async
    [4] ARCHIVE      open-vocabulary over stored video -> NOT BUILT (Phase 2)

Route 4 is declared and returns an explicit "not implemented" rather than being
silently absent or, worse, quietly answered by one of the other routes. The
system should be able to say what it cannot do.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..config import SystemCfg
from ..storage import Repository
from .sql_route import SqlRoute
from .visual_route import VisualGroundingRoute

_GROUNDING_PAT = re.compile(
    r"\b(where (is|are|in)|locate|point (to|out)|find .* in (this|the) (frame|image)"
    r"|show me .* in (this|the) (frame|image)|highlight|which part)\b", re.I)

_IDENTITY_PAT = re.compile(
    r"\b(who is|identity of|track (person )?#?\d+|is .* friendly"
    r"|enrolled|gallery|match .* person)\b", re.I)

_ARCHIVE_PAT = re.compile(
    r"\b(last (tuesday|monday|wednesday|thursday|friday|saturday|sunday|week|month)"
    r"|find .* footage|search (the )?archive|any footage|clips? (of|showing))\b", re.I)


class QueryRouter:
    """Classifies an operator query and dispatches it to the cheapest backend."""

    def __init__(self, cfg: SystemCfg, repo: Repository) -> None:
        self.cfg = cfg
        self.repo = repo
        self.sql = SqlRoute(cfg.query, repo)
        self.visual = VisualGroundingRoute(cfg)

    def classify(self, text: str) -> str:
        q = text.strip()
        if _GROUNDING_PAT.search(q):
            return "grounding"
        if _ARCHIVE_PAT.search(q):
            return "archive"
        if _IDENTITY_PAT.search(q):
            return "identity"
        return "structured"

    def route(
        self,
        text: str,
        frame: np.ndarray | None = None,
        tracks: dict[int, Any] | None = None,
        force: str | None = None,
    ) -> dict[str, Any]:
        route = force or self.classify(text)

        if route == "grounding":
            if frame is None:
                return {"route": "grounding", "ok": False, "status": 400,
                        "error": "no keyframe supplied for a grounding query"}
            phrase = self._strip_grounding_verb(text)
            mode = "single" if re.search(r"\bthe (one|single)\b", text, re.I) else "multi"
            return self.visual.ground(frame, phrase, mode)

        if route == "identity":
            return self._identity(text, tracks or {})

        if route == "archive":
            # Declared and refused. Archive-wide semantic search needs a
            # continuously-built clip-embedding index, which is Phase 2. Saying
            # so is better than answering from the alert log and implying the
            # whole archive was searched.
            return {
                "route": "archive", "ok": False, "status": 501,
                "error": "archive-wide semantic search is not implemented in Phase 1",
                "detail": ("Searching all stored footage requires a continuously "
                           "built clip-embedding index. Only alerted footage is "
                           "searchable today -- try the structured route."),
                "fallback": self.sql.run(text),
            }

        return self.sql.run(text)

    def _identity(self, text: str, tracks: dict[int, Any]) -> dict[str, Any]:
        m = re.search(r"#?(\d+)", text)
        rows = []
        for tid, t in tracks.items():
            if m and int(m.group(1)) != tid:
                continue
            ident = getattr(t, "identity", None)
            rows.append({
                "track_id": tid,
                "class_name": t.latest.class_name,
                "group": t.group.value,
                "label": ident.label.value if ident else "UNKNOWN",
                "matched": ident.matched_enrollee if ident else None,
                "similarity": ident.similarity if ident else None,
                "threshold": ident.threshold_used if ident else None,
                "quality": ident.quality_score if ident else None,
                "note": t.meta.get("identity_note"),
            })
        if not rows:
            return {"route": "identity", "ok": True, "rows": [],
                    "answer": "No matching live track."}
        r = rows[0]
        if r["label"] == "FRIENDLY":
            ans = (f"Track #{r['track_id']} matches enrolled friendly "
                   f"'{r['matched']}' at similarity {r['similarity']:.3f} "
                   f"(threshold {r['threshold']:.2f}).")
        elif r["label"] == "UNIDENTIFIED":
            ans = (f"Track #{r['track_id']} is Unidentified -- best gallery "
                   f"similarity {r['similarity']:.3f} is below the "
                   f"{r['threshold']:.2f} threshold. This is not an assertion of "
                   f"hostility; it means the subject is not enrolled.")
        else:
            ans = (f"Track #{r['track_id']}: identity undetermined"
                   + (f" ({r['note']})" if r.get("note") else "") + ".")
        return {"route": "identity", "ok": True, "backend": "osnet_gallery",
                "rows": rows, "answer": ans}

    @staticmethod
    def _strip_grounding_verb(text: str) -> str:
        q = re.sub(r"^\s*(where\s+(is|are|in)|locate|find|show me|highlight|point to)\s+",
                   "", text.strip(), flags=re.I)
        q = re.sub(r"\s+in\s+(this|the)\s+(frame|image)\s*\??$", "", q, flags=re.I)
        return q.strip(" ?.") or text.strip()

    def describe(self) -> dict[str, Any]:
        return {
            "routes": {
                "structured": {"backend": "sqlite+fts5", "available": True},
                "identity": {"backend": "osnet_gallery", "available": True},
                "grounding": self.visual.describe(),
                "archive": {"backend": "clip index", "available": False,
                            "note": "Phase 2 -- not implemented"},
            }
        }
