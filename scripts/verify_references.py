#!/usr/bin/env python3
"""Verify every reference against Crossref and emit BibTeX from the response.

The manuscript may cite only formally published work, so a reference is not
accepted on the strength of a search result. Each DOI is resolved against
Crossref and the bibliography entry is built from what Crossref returns, not
from what was typed here. Anything that fails to resolve, or that resolves to a
posted-content record (a preprint), is reported and excluded.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

PREPRINT_TYPES = {"posted-content"}
PREPRINT_VENUES = ("arxiv", "biorxiv", "medrxiv", "preprints.org", "ssrn", "research square")


def fetch(doi: str, mailto: str) -> dict | None:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={mailto}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)["message"]
    except Exception:
        return None


def latex_escape(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    for char, repl in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(char, repl)
    return text


SMALL_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "is", "of",
    "on", "or", "the", "to", "via", "with",
}


def normalise_title(title: str) -> str:
    """Crossref stores some older records in block capitals; restore title case.

    Applied only when the record is almost entirely uppercase, so acronyms in
    normally-cased titles are left alone.
    """
    letters = [c for c in title if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return title
    words = title.split()
    out = []
    for index, word in enumerate(words):
        lowered = word.lower()
        if index and lowered.strip(".,:;") in SMALL_WORDS:
            out.append(lowered)
        else:
            out.append(lowered[:1].upper() + lowered[1:])
    return " ".join(out)


def normalise_name(name: str) -> str:
    """Some older Crossref records store author names in block capitals too."""
    letters = [c for c in name if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return name
    return " ".join(
        "-".join(q[:1].upper() + q[1:] for q in part.lower().split("-"))
        for part in name.split()
    )


def normalise_booktitle(venue: str) -> str:
    """The MDPI style prefixes "In Proceedings of the"; avoid saying it twice."""
    for prefix in ("Proceedings of the ", "Proceedings of "):
        if venue.startswith(prefix):
            return venue[len(prefix):]
    return venue


def to_bibtex(key: str, m: dict) -> str:
    authors = " and ".join(
        normalise_name(f"{a.get('family','')}, {a.get('given','')}".strip(", "))
        for a in m.get("author", [])
        if a.get("family")
    )
    title = latex_escape(normalise_title(m["title"][0])) if m.get("title") else ""
    venue = latex_escape(m.get("container-title", [""])[0] if m.get("container-title") else "")
    year = m.get("issued", {}).get("date-parts", [[None]])[0][0]
    kind = m.get("type", "")
    entry = "article" if kind == "journal-article" else "inproceedings"
    fields = [f"  author = {{{authors}}}", f"  title = {{{title}}}"]
    if entry == "article":
        fields.append(f"  journal = {{{venue}}}")
    else:
        fields.append(f"  booktitle = {{{normalise_booktitle(venue)}}}")
    if year:
        fields.append(f"  year = {{{year}}}")
    for src, dst in (("volume", "volume"), ("issue", "number"), ("page", "pages")):
        if m.get(src):
            fields.append(f"  {dst} = {{{m[src]}}}")
    if m.get("article-number") and not m.get("page"):
        fields.append(f"  pages = {{{m['article-number']}}}")
    if m.get("publisher"):
        fields.append(f"  publisher = {{{latex_escape(m['publisher'])}}}")
    fields.append(f"  doi = {{{m['DOI']}}}")
    return "@" + entry + "{" + key + ",\n" + ",\n".join(fields) + "\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--bibtex", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mailto", default="kadyrbek.nurgali@kaznu.kz")
    args = parser.parse_args()

    catalogue = json.loads(args.catalogue.read_text(encoding="utf-8"))
    accepted, rejected, manual_published, official_artifacts = [], [], [], []
    for entry in catalogue["references"]:
        key, doi = entry["key"], entry.get("doi")
        if not doi:
            # Keep a non-Crossref item only when the catalogue supplies its
            # complete BibTeX. Official artifacts are exact experimental inputs;
            # the remainder are published proceedings or journal records.
            target = (
                official_artifacts
                if entry.get("kind") == "official_artifact"
                else manual_published
            )
            target.append(entry)
            continue
        m = fetch(doi, args.mailto)
        time.sleep(0.2)
        if m is None:
            rejected.append({**entry, "reason": "DOI did not resolve at Crossref"})
            continue
        venue = (m.get("container-title") or [""])[0].lower()
        if m.get("type") in PREPRINT_TYPES or any(v in venue for v in PREPRINT_VENUES):
            rejected.append({**entry, "reason": f"preprint record ({m.get('type')}, {venue})"})
            continue
        accepted.append(
            {
                "key": key,
                "doi": m["DOI"],
                "title": (m.get("title") or [""])[0],
                "venue": (m.get("container-title") or [""])[0],
                "year": m.get("issued", {}).get("date-parts", [[None]])[0][0],
                "type": m.get("type"),
                "bibtex": to_bibtex(key, m),
            }
        )

    body = "".join(e["bibtex"] for e in accepted)
    body += "".join(e["bibtex"] for e in manual_published if "bibtex" in e)
    body += "".join(e["bibtex"] for e in official_artifacts if "bibtex" in e)
    args.bibtex.parent.mkdir(parents=True, exist_ok=True)
    args.bibtex.write_text(
        "% Generated by scripts/verify_references.py.\n"
        "% DOI-backed entries resolved at Crossref; non-DOI entries are explicit\n"
        "% published records or immutable first-party experimental artifacts.\n\n"
        + body,
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(
            {
                "accepted": [
                    {k: v for k, v in e.items() if k != "bibtex"} for e in accepted
                ],
                "manual_published_no_crossref": manual_published,
                "official_artifacts": official_artifacts,
                "rejected": rejected,
                "counts": {
                    "accepted": len(accepted),
                    "manual_published": len(manual_published),
                    "official_artifacts": len(official_artifacts),
                    "manual": len(manual_published) + len(official_artifacts),
                    "rejected": len(rejected),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "manual_published": len(manual_published),
                "official_artifacts": len(official_artifacts),
                "rejected": len(rejected),
            }
        )
    )
    for r in rejected:
        print(f"  REJECTED {r['key']}: {r['reason']}")


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  (used in fetch)

    main()
