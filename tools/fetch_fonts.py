"""Fetch and instantiate the corpus-generator fonts (dev-connected environments only).

Downloads variable TTFs from the google/fonts GitHub repo (all OFL-licensed) and uses
fonttools to instance the static weights that generate_test_corpus.py expects in
tools/assets/fonts/. Air-gapped deployments never run this — the fonts ship with the repo.

Does NOT handle: font subsetting, hinting preservation guarantees, or non-Google sources.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

try:  # corporate TLS interception: trust the Windows cert store (holds the proxy's root CA)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from fontTools import ttLib
from fontTools.varLib import instancer

FONT_DIR = Path(__file__).parent / "assets" / "fonts"
RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl"

# source path in google/fonts -> list of (output name, axis pins or None for static copy)
SOURCES: dict[str, list[tuple[str, dict[str, float] | None]]] = {
    f"{RAW}/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf": [
        ("NotoSans-Regular.ttf", {"wght": 400, "wdth": 100}),
        ("NotoSans-Medium.ttf", {"wght": 500, "wdth": 100}),
        ("NotoSans-Bold.ttf", {"wght": 700, "wdth": 100}),
    ],
    f"{RAW}/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf": [
        ("NotoSansDevanagari-Regular.ttf", {"wght": 400, "wdth": 100}),
        ("NotoSansDevanagari-Bold.ttf", {"wght": 700, "wdth": 100}),
    ],
    f"{RAW}/notosansmono/NotoSansMono%5Bwdth%2Cwght%5D.ttf": [
        ("NotoSansMono-Regular.ttf", {"wght": 400, "wdth": 100}),
    ],
    f"{RAW}/caveat/Caveat%5Bwght%5D.ttf": [
        ("Caveat-Regular.ttf", {"wght": 400}),
    ],
    f"{RAW}/kalam/Kalam-Regular.ttf": [
        ("Kalam-Regular.ttf", None),
    ],
}


def main() -> int:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    for url, outputs in SOURCES.items():
        src = FONT_DIR / ("_dl_" + url.rsplit("/", 1)[-1].replace("%5B", "[").replace("%2C", ",").replace("%5D", "]"))
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, src)
        for name, axes in outputs:
            out = FONT_DIR / name
            if axes is None:
                src.replace(src.with_name(name))
                print(f"  saved static {name}")
                continue
            font = ttLib.TTFont(src)
            instancer.instantiateVariableFont(font, axes, inplace=True, updateFontNames=True)
            font.save(out)
            print(f"  instanced {name} {axes}")
        if src.exists():
            src.unlink()
    missing = [n for outs in SOURCES.values() for n, _ in outs if not (FONT_DIR / n).exists()]
    if missing:
        print(f"MISSING: {missing}", file=sys.stderr)
        return 1
    print(f"all fonts present in {FONT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
