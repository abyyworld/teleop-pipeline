"""Defect-injection tests for the page checker.

A checker that passes on a good page has proved nothing. Each test here breaks
exactly one property of a known-good page and asserts the checker notices, which
is the same discipline the quality metrics are held to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import check_page  # noqa: E402

GOOD = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>T</title>
<style>
:root { --bg: #ffffff; --ink: #14171a; --muted: #5b6470; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg: #0f1215; --ink: #e8eaed; --muted: #9ba3ae; }
}
:root[data-theme="dark"] { --bg: #0f1215; --ink: #e8eaed; --muted: #9ba3ae; }
body { background: var(--bg); color: var(--ink); }
</style></head>
<body>
<h1>Title</h1>
<p>Some prose, with no dashes of the long kind.</p>
<svg role="img" aria-label="a chart"><line x1="0" y1="0" x2="1" y2="1"/></svg>
</body></html>
"""


def check(source: str) -> list[str]:
    return [
        str(p)
        for p in (
            check_page.check_offline(source)
            + check_page.check_structure(source)
            + check_page.check_contrast(source)
            + check_page.check_prose(source)
            + check_page.check_theme_agreement(source)
        )
    ]


def test_the_known_good_page_passes():
    """Without this the other tests could all pass on a checker that always fails."""
    assert check(GOOD) == []


# -- contrast ----------------------------------------------------------------


def test_contrast_is_computed_not_guessed():
    # 4.5:1 exactly at the boundary, and a known textbook pair.
    assert check_page.contrast("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
    assert check_page.contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)


def test_contrast_is_symmetric():
    assert check_page.contrast("#123456", "#abcdef") == check_page.contrast("#abcdef", "#123456")


def test_a_short_hex_is_expanded():
    assert check_page.luminance("#fff") == check_page.luminance("#ffffff")


def test_low_contrast_ink_is_caught():
    bad = GOOD.replace("--ink: #14171a;", "--ink: #b9bcc0;")
    assert any("contrast/light" in p for p in check(bad)), "unreadable body text passed"


def test_low_contrast_in_dark_mode_only_is_still_caught():
    """The mode the author does not use is the one that ships broken."""
    bad = GOOD.replace("--ink: #e8eaed;", "--ink: #2a2e33;")
    assert any("contrast/dark" in p for p in check(bad))


def test_a_bad_value_in_one_dark_scope_is_not_masked_by_the_other():
    """Dark is declared twice. Merging the two would hide a fault in either."""
    bad = GOOD.replace("--ink: #e8eaed;", "--ink: #2a2e33;", 1)
    assert any("contrast/dark" in p for p in check(bad))


def test_the_two_dark_scopes_must_agree():
    """Otherwise OS-dark and the toggle render differently, and only one is seen."""
    bad = GOOD.replace("--ink: #e8eaed;", "--ink: #d4d7db;", 1)
    assert any("theme" in p and "--ink" in p for p in check(bad))


# -- offline -----------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        '<script src="https://cdn.example.com/x.js"></script>',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=X">',
        "<style>@import url('https://fonts.example.com/f.css');</style>",
        "<style>@font-face { src: url(https://cdn.example.com/f.woff2); }</style>",
    ],
)
def test_every_shape_of_external_request_is_caught(injection):
    """Each is a dependency the lab machine may not be able to reach."""
    assert any("offline" in p for p in check(GOOD.replace("</head>", injection + "</head>")))


def test_a_relative_reference_is_not_flagged():
    ok = GOOD.replace("</head>", '<link rel="icon" href="favicon.svg"></head>')
    assert not any("offline" in p for p in check(ok))


# -- structure ---------------------------------------------------------------


def test_an_unclosed_tag_is_caught():
    assert any("structure" in p for p in check(GOOD.replace("</body>", "<div></body>")))


def test_a_crossed_tag_is_caught():
    bad = GOOD.replace("<h1>Title</h1>", "<h1><em>Title</h1></em>")
    assert any("structure" in p for p in check(bad))


def test_a_duplicate_id_is_caught():
    bad = GOOD.replace("<h1>Title</h1>", '<h1 id="x">Title</h1><p id="x">two</p>')
    assert any("used 2 times" in p for p in check(bad))


def test_a_missing_lang_is_caught():
    assert any("lang" in p for p in check(GOOD.replace('<html lang="en">', "<html>")))


def test_an_unlabelled_svg_is_caught():
    bad = GOOD.replace('<svg role="img" aria-label="a chart">', "<svg>")
    assert any("a11y" in p for p in check(bad))


def test_a_void_element_does_not_count_as_unclosed():
    """Self-closing SVG and HTML voids must not be reported as dangling."""
    ok = GOOD.replace("</body>", '<img src="x.png" alt="x"><br><hr></body>')
    assert not any("structure" in p for p in check(ok))


# -- prose and motion --------------------------------------------------------


def test_an_em_dash_is_caught():
    assert any("em dash" in p for p in check(GOOD.replace("prose,", "prose —")))


def test_animation_without_a_reduced_motion_guard_is_caught():
    bad = GOOD.replace("body {", "a { transition: color .2s; }\nbody {")
    assert any("prefers-reduced-motion" in p for p in check(bad))


def test_animation_with_a_guard_passes():
    ok = GOOD.replace(
        "body {",
        "a { transition: color .2s; }\n"
        "@media (prefers-reduced-motion: reduce) { a { transition: none; } }\nbody {",
    )
    assert not any("prefers-reduced-motion" in p for p in check(ok))


# -- theme splitting ---------------------------------------------------------


def test_dark_declarations_are_attributed_to_a_dark_scope():
    blocks = check_page.theme_blocks(GOOD)
    dark = "".join(v for k, v in blocks.items() if k.startswith("dark"))
    assert "#0f1215" in dark
    assert "#0f1215" not in blocks["light"], "a dark value leaked into the light palette"


def test_each_dark_scope_is_reported_separately():
    """Merged scopes let one mask a fault in the other."""
    scopes = [k for k in check_page.theme_blocks(GOOD) if k.startswith("dark")]
    assert len(scopes) == 2, scopes


def test_a_nested_media_block_does_not_swallow_the_rest_of_the_stylesheet():
    """Brace matching, not a greedy search to the next closing brace."""
    css = (
        "<style>@media (prefers-color-scheme: dark) { :root { --ink: #eee; }"
        " .x { color: red; } } .after { --bg: #123456; }</style>"
    )
    blocks = check_page.theme_blocks(css)
    dark = "".join(v for k, v in blocks.items() if k.startswith("dark"))
    assert "#123456" not in dark, "the dark block ran past its closing brace"
    assert "#123456" in blocks["light"]
