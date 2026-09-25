"""Checks the published page against the promises it makes.

The site is served straight from the default branch, so nothing stands between
a bad commit and the live page. These are the properties that cannot be left to
a glance:

* **No external requests.** The page has to render on a lab machine with no
  outbound network, and a third-party request from it would be a tracking
  vector as well as a dependency.
* **Readable contrast, in both themes.** Computed from the declared custom
  properties rather than eyeballed, because the failure is invisible to whoever
  wrote it and obvious to whoever cannot read it.
* **No horizontal overflow on a phone.** One long unbroken token is enough to
  push the whole page sideways.
Run it with no arguments from the repository root. `--render` adds the browser
checks, which need Playwright and are skipped otherwise.
"""

from __future__ import annotations

import argparse
import html.parser
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Elements that never take a closing tag, so a balance check must not expect one.
VOID = {
    "area",
    "base",
    "br",
    "circle",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "line",
    "link",
    "meta",
    "param",
    "path",
    "polygon",
    "polyline",
    "rect",
    "source",
    "stop",
    "track",
    "use",
    "wbr",
}

EXTERNAL = re.compile(r"""<(?:script|link)\b[^>]*\b(?:src|href)\s*=\s*["']https?://""", re.I)
IMPORT = re.compile(r"""@import\s+(?:url\()?["']?https?://""", re.I)
CSS_URL = re.compile(r"""url\(\s*["']?https?://""", re.I)
HEX = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")

#: Contrast floors from WCAG 2.1. Large text is 18.66px bold or 24px plain.
AA_TEXT = 4.5
AA_LARGE = 3.0


@dataclass
class Problem:
    where: str
    detail: str

    def __str__(self) -> str:
        return f"{self.where}: {self.detail}"


# -- colour ------------------------------------------------------------------


def _channel(value: float) -> float:
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def custom_properties(css: str) -> dict[str, str]:
    """Every `--name: #hex` declaration, last one winning as the cascade would."""
    found: dict[str, str] = {}
    for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;}]+)", css):
        match = HEX.search(value)
        if match:
            found[name] = match.group(0)
    return found


def theme_blocks(source: str) -> dict[str, str]:
    """Split the stylesheet into one entry per theme scope.

    Dark is usually declared twice, once under `prefers-color-scheme` for the OS
    setting and once under `[data-theme="dark"]` for a toggle. They are returned
    separately rather than merged: merging lets a bad value in one scope be
    masked by a good value in the other, so the OS path and the toggle path
    could render differently and the check would see only the last one.
    """
    styles = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", source, re.S | re.I))
    scopes: dict[str, str] = {}
    for i, match in enumerate(re.finditer(r"prefers-color-scheme\s*:\s*dark", styles, re.I)):
        scopes[f"dark[media{i or ''}]"] = _balanced_block(styles, match.end())
    for i, match in enumerate(re.finditer(r"""\[data-theme\s*=\s*["']?dark""", styles, re.I)):
        scopes[f"dark[attr{i or ''}]"] = _balanced_block(styles, match.end())
    light = styles
    for block in scopes.values():
        light = light.replace(block, "")
    return {"light": light, **scopes}


def check_theme_agreement(source: str) -> list[Problem]:
    """The OS-dark path and the toggle path must declare the same palette.

    When they drift, the page renders one way for someone whose system is dark
    and another for someone who pressed the toggle, and only one of the two ever
    gets looked at.
    """
    scopes = theme_blocks(source)
    dark = {k: custom_properties(v) for k, v in scopes.items() if k.startswith("dark")}
    if len(dark) < 2:
        return []
    names = sorted(dark)
    first, rest = names[0], names[1:]
    problems = []
    for other in rest:
        shared = set(dark[first]) & set(dark[other])
        for token in sorted(shared):
            if dark[first][token].lower() != dark[other][token].lower():
                problems.append(
                    Problem(
                        "theme",
                        f"{token} is {dark[first][token]} in {first} but "
                        f"{dark[other][token]} in {other}",
                    )
                )
    return problems


def _balanced_block(text: str, start: int) -> str:
    """The `{...}` region following `start`, braces matched."""
    open_at = text.find("{", start)
    if open_at == -1:
        return ""
    depth, i = 0, open_at
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at : i + 1]
        i += 1
    return text[open_at:]


def check_contrast(source: str) -> list[Problem]:
    """Every ink token against every surface token, in each theme.

    Pairs every foreground-looking property with every background-looking one,
    which over-reports slightly. That is the right direction to be wrong in: a
    pair flagged here is cheap to dismiss, one missed is shipped.
    """
    problems: list[Problem] = []
    blocks = theme_blocks(source)
    base = custom_properties(blocks["light"])
    for theme, css in blocks.items():
        # Each scope is resolved over the light defaults, exactly as the cascade
        # does, and never over another dark scope.
        tokens = dict(base)
        tokens.update(custom_properties(css))
        if not tokens:
            continue
        surfaces = {
            k: v
            for k, v in tokens.items()
            if re.search(r"bg|surface|plane|band|paper|canvas|backdrop", k)
        }
        inks = {
            k: v
            for k, v in tokens.items()
            if re.search(r"ink|text|fg|foreground|muted|accent|link|heading", k)
        }
        if not surfaces or not inks:
            continue
        for sk, sv in surfaces.items():
            for ik, iv in inks.items():
                ratio = contrast(iv, sv)
                floor = AA_LARGE if re.search(r"muted|accent|rule|border", ik) else AA_TEXT
                if ratio < floor:
                    problems.append(
                        Problem(
                            f"contrast/{theme}",
                            f"{ik} ({iv}) on {sk} ({sv}) is {ratio:.2f}:1, needs {floor}:1",
                        )
                    )
    return problems


# -- structure ---------------------------------------------------------------


class Balance(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.problems: list[Problem] = []
        self.ids: dict[str, int] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.ids[value] = self.ids.get(value, 0) + 1
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(
                Problem("structure", f"</{tag}> closes {self.stack[-1:] or ['nothing']}")
            )
            return
        self.stack.pop()


def check_structure(source: str) -> list[Problem]:
    parser = Balance()
    parser.feed(source)
    problems = list(parser.problems)
    if parser.stack:
        problems.append(Problem("structure", f"unclosed tags: {parser.stack}"))
    for name, count in parser.ids.items():
        if count > 1:
            problems.append(Problem("structure", f"id {name!r} used {count} times"))
    if not re.search(r"<html[^>]+\blang=", source, re.I):
        problems.append(Problem("structure", "<html> has no lang attribute"))
    if not re.search(r"<svg[^>]*(role=|aria-label=|<title)", source, re.I | re.S):
        problems.append(Problem("a11y", "an SVG carries no role, aria-label or <title>"))
    return problems


def check_offline(source: str) -> list[Problem]:
    problems = []
    for pattern, what in (
        (EXTERNAL, "script or stylesheet"),
        (IMPORT, "@import"),
        (CSS_URL, "css url()"),
    ):
        for match in pattern.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            problems.append(Problem("offline", f"line {line}: external {what}"))
    return problems


def check_prose(source: str) -> list[Problem]:
    problems = []
    for match in re.finditer("—", source):
        line = source.count("\n", 0, match.start()) + 1
        problems.append(Problem("prose", f"line {line}: em dash"))
    if re.search(r"@keyframes|transition\s*:", source, re.I) and not re.search(
        r"prefers-reduced-motion", source, re.I
    ):
        problems.append(Problem("a11y", "animation with no prefers-reduced-motion guard"))
    return problems


# -- browser -----------------------------------------------------------------


def _browser_kwargs() -> dict:
    """Point Playwright at a browser that is already on the machine.

    A pinned Playwright expects one exact build directory, and an image that
    ships a different one fails with "executable doesn't exist" even though a
    perfectly good Chromium is installed. Rather than download a second copy,
    look for one under PLAYWRIGHT_BROWSERS_PATH and use it.
    """
    import os

    root = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""))
    if not root.is_dir():
        return {}
    candidates = sorted(root.glob("chromium-*/chrome-linux/chrome"))
    candidates += sorted(root.glob("chromium*/chrome-linux/headless_shell"))
    candidates += sorted(root.glob("chromium*/chrome-mac/Chromium.app/Contents/MacOS/Chromium"))
    return {"executable_path": str(candidates[-1])} if candidates else {}


SQUEEZE_PROBE = """() => {
  const out = [];
  for (const el of document.querySelectorAll('p, li, td, th, dd, figcaption, blockquote')) {
    const text = (el.textContent || '').trim();
    if (text.length < 40) continue;
    // A layout container is not a text block: its children set their own
    // lines, so characters-per-line says nothing about it.
    let container = false;
    for (const child of el.children) {
      const d = getComputedStyle(child).display;
      if (d !== 'inline' && d !== 'inline-block' && d !== 'contents') { container = true; break; }
    }
    if (container) continue;
    const box = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    const size = parseFloat(style.fontSize) || 16;
    const lineHeight = parseFloat(style.lineHeight) || size * 1.4;
    if (box.width < 120 || box.height <= 0) continue;
    const lines = Math.max(box.height / lineHeight, 1);
    const actual = text.length / lines;
    // A rough average advance of half the font size is close enough across
    // proportional and monospace faces for an order-of-magnitude test.
    const expected = box.width / (size * 0.5);
    out.push({
      ratio: actual / expected,
      width: Math.round(box.width),
      actual: Math.round(actual),
      tag: el.tagName.toLowerCase(),
      text: text.slice(0, 60),
    });
  }
  return out;
}"""


def _squeezed_text(page, width: int) -> list[Problem]:
    """Catch text wrapping far narrower than its box allows.

    A broken flex or grid child keeps its full width while its text wraps to a
    few characters a line. Contrast and overflow checks both pass, the page
    looks catastrophic, and nothing notices. Comparing the characters per line
    actually achieved against what the box and font size predict separates that
    from a display heading deliberately broken over three lines, which is why
    the test is a ratio rather than a flat threshold.
    """
    problems = []
    for block in page.evaluate(SQUEEZE_PROBE):
        if block["ratio"] < 0.25:
            problems.append(
                Problem(
                    f"layout/{width}px",
                    f"<{block['tag']}> is {block['width']}px wide but wraps at about "
                    f"{block['actual']} characters a line: {block['text']!r}",
                )
            )
    return problems


def check_rendered(path: Path, widths=(320, 390, 768, 1200)) -> list[Problem]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  (skipping browser checks: playwright is not installed)")
        return []

    problems: list[Problem] = []
    url = path.resolve().as_uri()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"], **_browser_kwargs())
        for width in widths:
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.goto(url, wait_until="load")
            problems += _squeezed_text(page, width)
            measured = page.evaluate(
                """() => {
                  const d = document.documentElement;
                  const wide = [...document.querySelectorAll('body *')].filter(el => {
                    if (el.closest('[data-scroll], .scroll, .chart')) return false;
                    return el.getBoundingClientRect().right > d.clientWidth + 1;
                  }).map(el => el.tagName.toLowerCase() + (el.id ? '#' + el.id : ''));
                  return {scroll: d.scrollWidth, client: d.clientWidth, wide: [...new Set(wide)].slice(0, 5)};
                }"""
            )
            if measured["scroll"] > measured["client"] + 1:
                problems.append(
                    Problem(
                        f"overflow/{width}px",
                        f"page scrolls to {measured['scroll']} in a {measured['client']} viewport; "
                        f"widest: {measured['wide']}",
                    )
                )
            page.close()

        # Content must survive with scripting off, which is also how it behaves
        # for the first paint and for anyone with a blocker.
        context = browser.new_context(
            java_script_enabled=False, viewport={"width": 1200, "height": 900}
        )
        page = context.new_page()
        page.goto(url, wait_until="load")
        text = page.evaluate("() => document.body.innerText.length")
        if text < 2000:
            problems.append(
                Problem("no-js", f"only {text} characters render with JavaScript disabled")
            )
        context.close()
        browser.close()
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", nargs="?", default="index.html", type=Path)
    parser.add_argument("--render", action="store_true", help="also run the browser checks")
    args = parser.parse_args(argv)

    path = args.page
    if not path.is_file():
        print(f"no such page: {path}", file=sys.stderr)
        return 2
    source = path.read_text(encoding="utf-8")

    problems: list[Problem] = []
    problems += check_offline(source)
    problems += check_structure(source)
    problems += check_contrast(source)
    problems += check_prose(source)
    problems += check_theme_agreement(source)
    if args.render:
        problems += check_rendered(path)

    if problems:
        print(f"{len(problems)} problem(s) in {path}:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"{path}: {len(source)} bytes, all checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
