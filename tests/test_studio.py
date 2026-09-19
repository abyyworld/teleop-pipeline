"""Tests for the desktop app.

The app can start a training run, so the tests that matter most are the ones
about who is allowed to ask it to. Everything here runs against a real server
on a real loopback socket, because the guards being tested live in the HTTP
layer and a mocked request would not exercise them.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from teleop_pipeline.studio import pipeline as pipe
from teleop_pipeline.studio.server import make_server, url_for


@pytest.fixture
def studio_server():
    server, studio = make_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = url_for(server, studio).split("/?")[0]
    try:
        yield base, studio
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(base, path, *, token=None, host=None, payload=None, method=None):
    sep = "&" if "?" in path else "?"
    url = f"{base}{path}{sep}t={token or ''}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# -- who may drive the app ----------------------------------------------------


def test_the_page_needs_the_session_token(studio_server):
    """Otherwise any page you visit could drive the pipeline over localhost."""
    base, studio = studio_server
    assert request(base, "/", token=studio.token)[0] == 200
    assert request(base, "/", token="guessed")[0] == 403
    assert request(base, "/")[0] == 403


def test_every_api_route_needs_the_token(studio_server):
    base, studio = studio_server
    for path in ("/api/state", "/api/log"):
        assert request(base, path, token="guessed")[0] == 403, path
    assert request(base, "/api/run", token="guessed", payload={}, method="POST")[0] == 403


def test_a_request_addressed_to_another_hostname_is_refused(studio_server):
    """DNS rebinding: an attacker's name resolving to 127.0.0.1 still fails."""
    base, studio = studio_server
    code, _ = request(base, "/api/state", token=studio.token, host="attacker.example.com")
    assert code == 403


def test_loopback_hostnames_are_accepted(studio_server):
    base, studio = studio_server
    for host in ("localhost:1234", "127.0.0.1:1234", "localhost"):
        code, _ = request(base, "/api/state", token=studio.token, host=host)
        assert code == 200, host


# -- the API ------------------------------------------------------------------


def test_state_describes_every_stage(studio_server):
    base, studio = studio_server
    _, body = request(base, "/api/state", token=studio.token)
    state = json.loads(body)
    assert [s["key"] for s in state["stages"]] == [s.key for s in pipe.STAGES]
    assert "synth" not in state["default_run"], "generated data must not join a real corpus"


def test_unknown_stage_names_are_rejected(studio_server):
    base, studio = studio_server
    code, body = request(
        base, "/api/run", token=studio.token, payload={"stages": ["definitely-not"]}, method="POST"
    )
    assert code == 400
    assert "unknown stage" in json.loads(body)["error"]


def test_a_malformed_stage_list_is_rejected(studio_server):
    base, studio = studio_server
    for payload in (
        {"stages": "score"},
        {"stages": [1, 2]},
        {"options": []},
        # An explicit empty list asks for nothing. Treating it as "missing" would
        # start the entire pipeline instead.
        {"stages": []},
    ):
        code, _ = request(base, "/api/run", token=studio.token, payload=payload, method="POST")
        assert code == 400, payload


def test_unknown_routes_are_404(studio_server):
    base, studio = studio_server
    assert request(base, "/api/nope", token=studio.token)[0] == 404


# -- the runner ---------------------------------------------------------------


def test_only_one_run_at_a_time():
    """Two stages writing the episode store would race."""
    runner = pipe.Runner()
    started = threading.Event()
    release = threading.Event()

    def slow(cfg, options):
        started.set()
        release.wait(timeout=10)
        return {}

    runner_stages = dict(pipe.RUNNERS)
    try:
        pipe.RUNNERS["score"] = slow
        runner.start(["score"])
        assert started.wait(timeout=10)
        with pytest.raises(RuntimeError, match="already in progress"):
            runner.start(["score"])
    finally:
        release.set()
        pipe.RUNNERS.clear()
        pipe.RUNNERS.update(runner_stages)


def test_a_failing_stage_skips_the_ones_that_depend_on_it():
    """Training on the output of a failed dataset build is worse than stopping."""
    runner = pipe.Runner()
    original = dict(pipe.RUNNERS)

    def boom(cfg, options):
        raise RuntimeError("no episodes")

    try:
        pipe.RUNNERS["dataset"] = boom
        runner.start(["dataset", "train", "eval"])
        for _ in range(200):
            if not runner.busy:
                break
            threading.Event().wait(0.05)
        statuses = {r.key: r.status for r in runner.runs}
        assert statuses == {"dataset": "failed", "train": "skipped", "eval": "skipped"}
        assert "no episodes" in runner.last_error
    finally:
        pipe.RUNNERS.clear()
        pipe.RUNNERS.update(original)


def test_unknown_stage_never_starts_a_thread():
    runner = pipe.Runner()
    with pytest.raises(ValueError, match="unknown stage"):
        runner.start(["rm -rf /"])
    assert not runner.busy


# -- the log ------------------------------------------------------------------


def test_log_returns_only_what_the_caller_has_not_seen():
    log = pipe.LogBuffer()
    log.write("one\ntwo\n")
    lines, cursor = log.since(0)
    assert lines == ["one", "two"]
    assert log.since(cursor)[0] == []
    log.write("three\n")
    assert log.since(cursor)[0] == ["three"]


def test_an_unterminated_line_is_not_lost():
    """A stage printing a progress line without a newline still shows up."""
    log = pipe.LogBuffer()
    log.write("working")
    assert log.since(0)[0] == []
    log.promote_partial()
    assert log.since(0)[0] == ["working"]


def test_a_cursor_past_the_end_does_not_raise():
    log = pipe.LogBuffer()
    log.write("one\n")
    assert log.since(99) == ([], 1)


def test_rich_writes_plain_text_here():
    """isatty() is False, which is what stops escape codes reaching the browser."""
    assert pipe.LogBuffer().isatty() is False


# -- the packaged build -------------------------------------------------------


def test_self_test_passes_in_a_working_install():
    """CI runs this inside the binary, where a lost data file is easy to miss."""
    from teleop_pipeline.studio.launch import self_test

    assert self_test() == 0


def test_self_test_fails_when_the_page_is_missing(monkeypatch, tmp_path):
    from teleop_pipeline.studio import launch, server

    monkeypatch.setattr(server, "INDEX", tmp_path / "gone.html")
    assert launch.self_test() == 1


def test_every_stage_has_an_implementation():
    """A stage in the UI with no runner is a button that 500s."""
    assert {s.key for s in pipe.STAGES} == set(pipe.RUNNERS)


def test_every_stage_declares_where_its_output_lands():
    from teleop_pipeline.config import load_config

    assert set(pipe.stage_outputs(load_config(None))) == {s.key for s in pipe.STAGES}
