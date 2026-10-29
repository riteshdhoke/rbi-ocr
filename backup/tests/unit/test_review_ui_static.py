"""The reviewer pages are air-gap safe (WS-G, WS-I item 1).

Production is air-gapped: a page that quietly reaches for a CDN font works
perfectly on the developer's laptop and renders as a blank grey box on the node
that matters, months later, with nobody able to say what changed.  So the rule
is mechanical — **no absolute URL may appear in any review-UI asset** except a
single annotated loopback default — and this module is what enforces it.

It parses the HTML rather than pattern-matching it, so a URL smuggled in as an
attribute is caught even when the raw text scan would have been satisfied.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from app.review_ui import (
    REVIEW_PAGE_FILENAME,
    SHARED_ASSET_FILENAMES,
    STATIC_DIR,
    TEST_PAGE_FILENAME,
    review_page_path,
    static_assets,
    static_path,
)

# Aliased: pytest would otherwise collect ``test_page_path`` as a test case.
from app.review_ui import test_page_path as api_test_page_path

HTML_FILES = (TEST_PAGE_FILENAME, REVIEW_PAGE_FILENAME)
ALL_FILES = HTML_FILES + SHARED_ASSET_FILENAMES

#: The marker a line must carry to be allowed to contain an absolute URL.
API_BASE_MARKER = "API-BASE-PLACEHOLDER"

#: Attributes whose value is fetched by the browser.
URL_ATTRIBUTES = (
    "src",
    "href",
    "action",
    "formaction",
    "poster",
    "data",
    "srcset",
    "background",
    "cite",
    "manifest",
    "ping",
)

ABSOLUTE_URL = re.compile(r"https?://[^\s\"'<>)]*")
SCHEME_OR_PROTOCOL_RELATIVE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)")

#: Ways to put a string into the DOM as markup. None of them may appear: every
#: value rendered by these pages comes out of a complaint document.
FORBIDDEN_DOM_SINKS = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "eval(",
    "new Function(",
)

#: The published API surface. A reviewer page must not invent an endpoint.
PUBLISHED_PATHS = ("/v1/documents", "/v1/jobs/", "/healthz", "/metrics")


def read(name: str) -> str:
    return static_path(name).read_text(encoding="utf-8")


class _Page(HTMLParser):
    """Collects what the air-gap rules are stated in terms of."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.ids: set[str] = set()
        self.scripts: list[str] = []
        self.styles: list[str] = []
        self.comments: list[str] = []
        self._mode: str | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        mapping = dict(attrs)
        self.elements.append((tag, mapping))
        identifier = mapping.get("id")
        if identifier:
            self.ids.add(identifier)
        if tag in ("script", "style"):
            self._mode = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._mode = None

    def handle_data(self, data: str) -> None:
        if self._mode == "script":
            self.scripts.append(data)
        elif self._mode == "style":
            self.styles.append(data)

    def handle_comment(self, data: str) -> None:
        self.comments.append(data)


@pytest.fixture(scope="module")
def pages() -> dict[str, _Page]:
    parsed: dict[str, _Page] = {}
    for name in HTML_FILES:
        page = _Page()
        page.feed(read(name))
        page.close()
        parsed[name] = page
    return parsed


def _strip_comments(text: str) -> str:
    """Remove HTML, block and line comments, leaving code only.

    ``//`` preceded by ``:`` is left alone so that a URL inside a string is
    *not* mistaken for the start of a comment — which is the whole point: a
    stray absolute URL must survive this stripping and be caught.
    """
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return "\n".join(re.sub(r"(?<!:)//.*$", " ", line) for line in text.splitlines())


# --------------------------------------------------------------------------- #
# The files exist and the accessors work
# --------------------------------------------------------------------------- #


def test_every_asset_ships() -> None:
    for name in ALL_FILES:
        assert static_path(name).is_file(), name
    assert {path.name for path in static_assets()} == set(ALL_FILES)


def test_page_accessors_point_at_the_pages() -> None:
    assert api_test_page_path().name == TEST_PAGE_FILENAME
    assert review_page_path().name == REVIEW_PAGE_FILENAME
    assert api_test_page_path().parent == STATIC_DIR


@pytest.mark.parametrize(
    "name", ["../secrets.env", "sub/dir.html", "", ".", "..", "a\\b.html"]
)
def test_static_path_never_traverses(name: str) -> None:
    with pytest.raises(ValueError):
        static_path(name)


def test_missing_asset_is_loud() -> None:
    with pytest.raises(FileNotFoundError):
        static_path("not_shipped.html")


def test_assets_are_utf8() -> None:
    for name in ALL_FILES:
        static_path(name).read_bytes().decode("utf-8")


# --------------------------------------------------------------------------- #
# Air gap: no external URL, anywhere, in any form
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL_FILES)
def test_no_absolute_url_outside_comments_and_the_marker(name: str) -> None:
    body = read(name)
    kept = [
        line for line in body.splitlines() if API_BASE_MARKER not in line
    ]
    offenders = ABSOLUTE_URL.findall(_strip_comments("\n".join(kept)))
    assert offenders == [], f"{name} reaches out to {offenders}"


def test_the_only_absolute_url_is_an_annotated_loopback_default() -> None:
    marked = [
        line
        for name in ALL_FILES
        for line in read(name).splitlines()
        if API_BASE_MARKER in line
    ]
    urls = [url for line in marked for url in ABSOLUTE_URL.findall(line)]

    assert len(urls) == 1, urls
    host = urls[0].split("//", 1)[1].split(":", 1)[0].split("/", 1)[0]
    assert host in ("127.0.0.1", "localhost", "[::1]"), urls[0]


def test_every_url_attribute_is_a_local_relative_path(
    pages: dict[str, _Page]
) -> None:
    for name, page in pages.items():
        for tag, attrs in page.elements:
            for attribute in URL_ATTRIBUTES:
                value = attrs.get(attribute)
                if not value:
                    continue
                if value.startswith("#"):
                    continue
                assert not SCHEME_OR_PROTOCOL_RELATIVE.match(value), (
                    f"{name}: <{tag} {attribute}={value!r}> is not a local path"
                )
                assert ".." not in value, f"{name}: <{tag} {attribute}={value!r}>"
                assert static_path(value).is_file(), f"{name} references {value}"


def test_no_subresource_integrity_or_crossorigin(pages: dict[str, _Page]) -> None:
    """Both attributes only make sense for a third-party host."""
    for name, page in pages.items():
        for tag, attrs in page.elements:
            assert "integrity" not in attrs, name
            assert "crossorigin" not in attrs, name


def test_css_imports_and_fetches_nothing() -> None:
    css = _strip_comments(read("review_ui.css"))
    assert "@import" not in css
    assert "url(" not in css
    assert "@font-face" not in css


@pytest.mark.parametrize("name", ALL_FILES)
def test_no_markup_sinks(name: str) -> None:
    """Envelope text reaches the DOM through textContent or not at all."""
    body = _strip_comments(read(name))
    for sink in FORBIDDEN_DOM_SINKS:
        assert sink not in body, f"{name} uses {sink}"


@pytest.mark.parametrize("name", ALL_FILES)
def test_no_module_imports(name: str) -> None:
    """ES modules are blocked over file://; both pages must stay classic."""
    body = _strip_comments(read(name))
    assert 'type="module"' not in body
    assert not re.search(r"^\s*import\s", body, flags=re.MULTILINE)


# --------------------------------------------------------------------------- #
# Page shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", HTML_FILES)
def test_html_head_is_complete(name: str) -> None:
    body = read(name)
    page = _Page()
    page.feed(body)
    page.close()

    assert body.lstrip().lower().startswith("<!doctype html>")
    charsets = [
        attrs.get("charset")
        for tag, attrs in page.elements
        if tag == "meta" and "charset" in attrs
    ]
    assert charsets == ["utf-8"]
    assert any(tag == "title" for tag, _ in page.elements)
    assert any(
        tag == "html" and attrs.get("lang") for tag, attrs in page.elements
    )
    assert any(
        tag == "meta" and attrs.get("name") == "viewport"
        for tag, attrs in page.elements
    )


@pytest.mark.parametrize("name", HTML_FILES)
def test_both_pages_share_the_same_two_assets(
    name: str, pages: dict[str, _Page]
) -> None:
    page = pages[name]
    sources = {
        attrs.get("src") or attrs.get("href")
        for tag, attrs in page.elements
        if tag in ("script", "link")
    }
    assert "review_ui.js" in sources
    assert "review_ui.css" in sources


@pytest.mark.parametrize("name", HTML_FILES)
def test_every_element_the_script_looks_up_exists(
    name: str, pages: dict[str, _Page]
) -> None:
    """A typo in a selector is a silently dead button; catch it here."""
    script = "\n".join(pages[name].scripts)
    wanted = set(re.findall(r"RUI\.q\(\"#([A-Za-z0-9_-]+)\"\)", script))

    assert wanted, f"{name} looks nothing up"
    assert wanted <= pages[name].ids, sorted(wanted - pages[name].ids)


@pytest.mark.parametrize("name", HTML_FILES)
def test_pages_only_link_to_pages_that_exist(
    name: str, pages: dict[str, _Page]
) -> None:
    script = "\n".join(pages[name].scripts)
    for target in re.findall(r"\"([A-Za-z0-9_.-]+\.html)\"", script):
        assert static_path(target).is_file(), f"{name} opens missing {target}"


def test_test_page_has_the_controls_the_brief_asks_for(
    pages: dict[str, _Page]
) -> None:
    page = pages[TEST_PAGE_FILENAME]
    for identifier in (
        "api-base",      # configurable API base URL
        "file-input",    # upload a document
        "upload-btn",
        "requested-fields",
        "other-fields",
        "missing-fields",
        "banner",        # bounce banner
        "overview",
        "pages",         # per-page quality
        "fields",        # extracted fields table
        "lines",         # OCR text
        "page-canvas",   # evidence bbox over the page image
        "show-boxes",
        "raw",
        "review-btn",    # hand-off to the review page
    ):
        assert identifier in page.ids, identifier


def test_test_page_offers_an_offline_route(pages: dict[str, _Page]) -> None:
    """It must be usable with no API running: paste or load an envelope."""
    page = pages[TEST_PAGE_FILENAME]
    assert "paste-json" in page.ids
    assert "json-input" in page.ids
    assert "image-input" in page.ids


def test_review_page_is_keyboard_first(pages: dict[str, _Page]) -> None:
    page = pages[REVIEW_PAGE_FILENAME]
    script = "\n".join(page.scripts)
    body = read(REVIEW_PAGE_FILENAME)

    assert "RUI.bindKeys" in script
    for key in ('"j"', '"k"', '"a"', '"c"', '"s"', '"x"', '"e"', '"Escape"'):
        assert key in script, key
    assert "ArrowDown" in script and "ArrowUp" in script
    # the legend the reviewer reads
    assert body.count("<kbd>") >= 8


def test_review_page_shows_the_field_beside_its_evidence(
    pages: dict[str, _Page]
) -> None:
    page = pages[REVIEW_PAGE_FILENAME]
    for identifier in ("queue", "current", "crop-canvas", "correction", "decisions"):
        assert identifier in page.ids, identifier


def test_review_page_exports_rather_than_inventing_an_endpoint() -> None:
    body = read(REVIEW_PAGE_FILENAME)
    assert "downloadJSON" in body
    assert "/v1/review" not in body
    assert "PATCH" not in body
    assert "PUT" not in body


# --------------------------------------------------------------------------- #
# The shared script
# --------------------------------------------------------------------------- #


def test_shared_script_exposes_one_global() -> None:
    js = read("review_ui.js")
    globals_assigned = set(re.findall(r"global\.([A-Za-z_$][\w$]*)\s*=", js))
    assert globals_assigned == {"RUI"}


def test_shared_script_only_calls_published_endpoints() -> None:
    js = read("review_ui.js")
    paths = set(re.findall(r"\"(/[A-Za-z0-9_./{}-]*)\"", js))

    assert "/v1/documents" in paths
    assert "/healthz" in paths
    assert any(path.startswith("/v1/jobs/") for path in paths)
    for path in paths:
        assert any(path.startswith(known) for known in PUBLISHED_PATHS), path


def test_shared_script_renders_the_field_provenance_columns() -> None:
    js = read("review_ui.js")
    for column in (
        "value_en",
        "confidence",
        "source",
        "validator_passed",
        "flags",
        "evidence",
    ):
        assert column in js, column


def test_shared_script_knows_flagged_fields_and_bounce_reasons() -> None:
    js = read("review_ui.js")
    assert "bounce_decision" in js
    assert "quality_flags" in js
    assert "quality_score" in js
    assert "escalation_flag" in js
    assert "complaint_summary" in js


def test_shared_script_explains_the_file_protocol_trap() -> None:
    """Opened from disk the fetch fails on CORS; the page must say so."""
    js = read("review_ui.js")
    assert "file:" in js
    assert "Origin: null" in js


def test_shared_script_refuses_to_pretend_it_can_render_a_pdf() -> None:
    js = read("review_ui.js")
    assert "isPdf" in js
    assert "air-gapped" in js
