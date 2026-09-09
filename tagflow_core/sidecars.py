"""Shared, indexed ownership rules for captions and TagFlow metadata."""
from __future__ import annotations

from collections.abc import Set


def resolve_sidecar(name: str, folded_stems: Set[str]) -> tuple[str, str] | None:
    """Return (case-folded owner, original suffix), preferring the longest stem.

    ``photo.en.txt`` belongs to ``photo.en.png`` rather than ``photo.png`` when
    both images exist. Lookups depend on dots in a filename, not dataset size.
    Slice the original name before casefolding: casefold can change its length.
    Metadata has an exact owner; only text sidecars accept custom suffixes.
    """
    folded = name.casefold()
    if folded.endswith(".tagflow.json"):
        boundary = len(name) - len(".tagflow.json")
        owner = name[:boundary].casefold()
        return (owner, name[boundary:]) if owner in folded_stems else None
    if not folded.endswith(".txt"):
        return None
    boundary = len(name) - len(".txt")
    while boundary > 0:
        owner = name[:boundary].casefold()
        if owner in folded_stems:
            return owner, name[boundary:]
        boundary = name.rfind(".", 0, boundary)
    return None
