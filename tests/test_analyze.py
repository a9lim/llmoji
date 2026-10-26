"""``run_analyze`` end to end against a fake synthesizer, and the dry-run plan.

The fake answers from a hash of the prompt, so a result depends only on
what a cell sampled, never on call order. Nothing here reaches a network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from llmoji import analyze, paths
from llmoji.analyze import (
    INSTANCE_SAMPLE_CAP,
    AnalyzeError,
    _bucket,
    _cells,
    plan_analyze,
    run_analyze,
)
from llmoji.scrape import ScrapeRow
from llmoji.synth import Synthesizer, SynthesizerError, load_cache, validate_synthesis
from llmoji.synth_prompts import (
    CIRCUMPLEX_ANCHORS,
    DEFAULT_ANTHROPIC_MODEL_ID,
    DEFAULT_OPENAI_MODEL_ID,
    EXTENSION_AXES,
    LEXICON_VERSION,
)

BACKEND, BASE_URL, MODEL = "local", "http://fake/v1", "fake-model"
SMILE, ANGRY = "(◕‿◕)", "(>_<)"


def _row(face: str, body: str = "hi", user: str = "u", model: str | None = "m1",
         source: str = "test") -> ScrapeRow:
    return ScrapeRow(source=source, model=model, timestamp="2026-10-25T00:00:00Z",
                     cwd="/tmp", assistant_text=body, first_word=face, surrounding_user=user)


def _rows() -> list[ScrapeRow]:
    """Three cells: m1/smile (two rows, one decorated), m1/angry, m2/smile;
    plus a row whose face canonicalizes to nothing."""
    return [
        _row(SMILE, "one", "u1"), _row("٩(◕‿◕)۶", "two", "u2"), _row(ANGRY, "ow", "u3"),
        _row(SMILE, "hello", "u4", model="m2"), _row("", "no face", "u5"),
    ]


def _answer(prompt: str) -> dict[str, list[str]]:
    h = hashlib.sha256(prompt.encode()).digest()
    return {
        "primary_affect": [CIRCUMPLEX_ANCHORS[h[0] % len(CIRCUMPLEX_ANCHORS)]],
        "stance_modality_function": [
            EXTENSION_AXES[h[1] % len(EXTENSION_AXES)],
            EXTENSION_AXES[h[2] % len(EXTENSION_AXES)],
        ],
    }


class FakeSynth(Synthesizer):
    """Records every prompt; fails on prompts for which ``fail`` is true."""

    backend, base_url = BACKEND, BASE_URL

    def __init__(self, fail: Callable[[str], bool] | None = None) -> None:
        super().__init__(MODEL)
        self.prompts: list[str] = []
        self.fail = fail

    def call_structured(self, prompt: str, *, schema: dict[str, Any]) -> dict[str, list[str]]:
        with self._lock:
            self.prompts.append(prompt)
        if self.fail and self.fail(prompt):
            raise SynthesizerError("boom")
        return _answer(prompt)


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch) -> Callable[..., analyze.AnalyzeResult]:
    def _run(rows: list[ScrapeRow], synth: FakeSynth | None = None, **kw: Any):
        fake = synth or FakeSynth()
        monkeypatch.setattr(analyze, "make_synthesizer", lambda backend, **k: fake)
        return run_analyze(iter(rows), print_progress=False, **kw)

    return _run


def _bundle() -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(paths.bundle_dir().iterdir())}


def _cache_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in paths.cache_path().read_text().splitlines()]


def _plan(rows: list[ScrapeRow]) -> analyze.AnalyzePlan:
    return plan_analyze(rows, backend=BACKEND, base_url=BASE_URL, model_id=MODEL)


# --- bundle schema ----------------------------------------------------------

def test_bundle_schema(run: Callable[..., Any]) -> None:
    result = run(_rows(), notes="a note")
    files = _bundle()
    assert sorted(files) == ["m1.jsonl", "m2.jsonl", "manifest.json"]

    manifest = json.loads(files["manifest.json"])
    assert set(manifest) == {
        "llmoji_version", "lexicon_version", "synthesis_model_id", "synthesis_backend",
        "submitter_id", "generated_at", "providers_seen", "model_counts",
        "total_synthesized_rows", "notes",
    }
    assert manifest["lexicon_version"] == LEXICON_VERSION
    assert manifest["synthesis_model_id"] == MODEL and manifest["synthesis_backend"] == BACKEND
    assert len(manifest["submitter_id"]) == 32
    assert manifest["providers_seen"] == ["test"]
    assert manifest["model_counts"] == {"m1": 4, "m2": 1}  # the faceless row counts
    assert manifest["total_synthesized_rows"] == 3 and manifest["notes"] == "a note"

    rows = {name: [json.loads(line) for line in text.splitlines()]
            for name, text in files.items() if name.endswith(".jsonl")}
    assert [(r["kaomoji"], r["count"]) for r in rows["m1.jsonl"]] == [(ANGRY, 1), (SMILE, 2)]
    assert [(r["kaomoji"], r["count"]) for r in rows["m2.jsonl"]] == [(SMILE, 1)]
    for r in (r for rs in rows.values() for r in rs):
        assert set(r) == {"kaomoji", "count", "synthesis"}
        validate_synthesis(r["synthesis"])
    assert all(text.endswith("\n") for text in files.values())
    assert (result.total_rows, result.canonical_unique) == (5, 2)
    assert (result.calls_made, result.calls_cached) == (3, 0)
    assert result.providers_seen == ["test"] and result.bundle_dir == paths.bundle_dir()


def test_cache_rows_record_their_cell(run: Callable[..., Any]) -> None:
    run(_rows())
    rows = _cache_rows()
    assert len(rows) == len({r["key"] for r in rows}) == 3
    assert {(r["source_model"], r["kaomoji"]) for r in rows} == {
        ("m1", SMILE), ("m1", ANGRY), ("m2", SMILE)}
    assert all(r["model"] == MODEL and r["backend"] == BACKEND for r in rows)


def test_bucket_names_models_and_counts_every_row() -> None:
    rows = [
        _row(SMILE, model=" m1 "), _row("", model="m1"), _row(SMILE, model=None, source="s-hook"),
        _row(ANGRY, model="", source="other"),
    ]
    buckets, providers, counts = _bucket(rows)
    assert providers == ["other", "s-hook", "test"]
    assert counts == {"m1": 2, "s-hook": 1, "other": 1}
    assert {sm: sorted(cells) for sm, cells in buckets.items()} == {
        "m1": [SMILE], "s-hook": [SMILE], "other": [ANGRY]}


# --- determinism and the cache ----------------------------------------------

def test_output_is_byte_identical_across_cold_runs(run: Callable[..., Any]) -> None:
    """Concurrency and call order must not reach the bundle."""
    run(_rows(), concurrency=4)
    first = _bundle()
    paths.cache_path().unlink()
    run(_rows(), concurrency=1)
    second = _bundle()
    for name in first:
        if name == "manifest.json":
            first_m, second_m = json.loads(first[name]), json.loads(second[name])
            first_m.pop("generated_at"), second_m.pop("generated_at")
            assert first_m == second_m
        else:
            assert first[name] == second[name]


def test_warm_cache_makes_no_calls(run: Callable[..., Any]) -> None:
    run(_rows())
    before = _bundle()
    synth = FakeSynth()
    result = run(_rows(), synth)
    assert synth.prompts == []
    assert (result.calls_made, result.calls_cached) == (0, 3)
    assert len(_cache_rows()) == 3
    assert {k: v for k, v in _bundle().items() if k != "manifest.json"} == {
        k: v for k, v in before.items() if k != "manifest.json"}


@pytest.mark.parametrize("concurrency", [1, 4])
def test_partial_failure_keeps_successes_and_a_rerun_resumes(
    run: Callable[..., Any], concurrency: int,
) -> None:
    rows = [_row(f"(face{i})", f"body{i}", "FAILME" if i == 2 else f"u{i}") for i in range(4)]
    failing = FakeSynth(fail=lambda prompt: "FAILME" in prompt)
    with pytest.raises(AnalyzeError, match=r"1 of 4 .*failed.*re-run") as exc:
        run(rows, failing, concurrency=concurrency)
    assert isinstance(exc.value.__cause__, SynthesizerError)
    assert len(failing.prompts) == 4
    assert sorted(r["kaomoji"] for r in _cache_rows()) == ["(face0)", "(face1)", "(face3)"]
    assert not paths.bundle_dir().exists()  # nothing half-written

    resumed = FakeSynth()
    result = run(rows, resumed, concurrency=concurrency)
    assert len(resumed.prompts) == 1 and "FAILME" in resumed.prompts[0]
    assert (result.calls_made, result.calls_cached) == (1, 3)
    assert sorted(_bundle()) == ["m1.jsonl", "manifest.json"]


def test_a_failed_run_leaves_the_previous_bundle_alone(run: Callable[..., Any]) -> None:
    run([_row(SMILE)])
    before = _bundle()
    with pytest.raises(AnalyzeError):
        run([_row(ANGRY, "fresh")], FakeSynth(fail=lambda prompt: True))
    assert _bundle() == before


def test_invalid_cached_rows_are_dispatched_again(run: Callable[..., Any]) -> None:
    run(_rows())
    rows = _cache_rows()
    by_cell = {(r["source_model"], r["kaomoji"]): r for r in rows}
    by_cell[("m1", SMILE)]["synthesis"]["primary_affect"] = ["ecstatic"]  # off the lexicon
    del by_cell[("m1", ANGRY)]["synthesis"]
    paths.cache_path().write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    assert _plan(_rows()).cached == 1

    synth = FakeSynth()
    result = run(_rows(), synth)
    assert len(synth.prompts) == 2 and (result.calls_made, result.calls_cached) == (2, 1)
    for line in _bundle()["m1.jsonl"].splitlines():
        assert set(json.loads(line)["synthesis"]["primary_affect"]) <= set(CIRCUMPLEX_ANCHORS)
    # The repair is cached: the newest row for a key wins.
    assert all(validate_synthesis(r["synthesis"]) for r in load_cache(paths.cache_path()).values())
    again = FakeSynth()
    run(_rows(), again)
    assert again.prompts == []


def test_cache_misses_when_backend_endpoint_or_model_changes(run: Callable[..., Any]) -> None:
    run([_row(SMILE)])
    for attrs in ({"backend": "openai"}, {"base_url": "http://elsewhere/v1"}, {"model_id": "other"}):
        synth = FakeSynth()
        for name, value in attrs.items():
            setattr(synth, name, value)
        run([_row(SMILE)], synth)
        assert len(synth.prompts) == 1, attrs


# --- the bundle directory ---------------------------------------------------

def test_slug_collision_is_refused_before_anything_is_called(
    run: Callable[..., Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    run([_row(SMILE, model="old")])
    before = _bundle()
    calls: list[str] = []
    monkeypatch.setattr(Synthesizer, "call_structured", lambda self, p, **kw: calls.append(p))
    rows = [_row(SMILE, model="Some-Model:Tag"), _row(SMILE, model="some-model-tag")]
    with pytest.raises(ValueError, match="collide on bundle filenames"):
        run(rows)
    assert _bundle() == before and calls == []


def test_cli_dry_run_calls_nothing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from llmoji import cli
    from llmoji.providers import get_provider

    journal = get_provider("codex").journal_path
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({"ts": "t", "model": "gpt-x", "cwd": "", "kaomoji": SMILE,
                                   "user_text": "u", "assistant_text": "hi"}) + "\n")
    monkeypatch.setattr(analyze, "make_synthesizer", lambda *a, **k: pytest.fail("synthesizer built"))
    assert cli.main(["analyze", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "1 call(s) to anthropic/claude-haiku-5-5" in out and "Nothing was called" in out


def test_bundle_dir_is_cleared_and_symlinks_are_unlinked_not_followed(
    run: Callable[..., Any], tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "canary.txt"
    canary.write_text("keep me")
    bundle = paths.bundle_dir()
    (bundle / "subdir").mkdir(parents=True)
    (bundle / "subdir" / "inner.jsonl").write_text("{}\n")
    (bundle / "stray.txt").write_text("would leak")
    (bundle / "stale.jsonl").write_text("{}\n")
    (bundle / "linked-dir").symlink_to(outside, target_is_directory=True)
    (bundle / "linked-file.jsonl").symlink_to(canary)
    (bundle / "dangling.jsonl").symlink_to(tmp_path / "missing")

    run([_row(SMILE)])

    assert sorted(_bundle()) == ["m1.jsonl", "manifest.json"]
    assert not any(p.is_symlink() for p in bundle.iterdir())
    assert canary.read_text() == "keep me"  # the link's target was not emptied


# --- sampling ---------------------------------------------------------------

def _crowd(n: int, model: str = "m1") -> list[ScrapeRow]:
    return [_row(SMILE, f"body{i}", f"user{i}", model=model) for i in range(n)]


def _prompts(rows: list[ScrapeRow], **kw: str) -> list[str]:
    buckets, _, _ = _bucket(rows)
    return [c.prompt for c in _cells(buckets, model_id=MODEL, backend=BACKEND, base_url="", **kw)]


def test_a_crowded_cell_is_sampled_down_to_the_cap(run: Callable[..., Any]) -> None:
    synth = FakeSynth()
    run(_crowd(INSTANCE_SAMPLE_CAP + 6), synth)
    assert len(synth.prompts) == 1
    assert synth.prompts[0].count("[Sample ") == INSTANCE_SAMPLE_CAP
    entry = json.loads(_bundle()["m1.jsonl"])
    assert entry["count"] == INSTANCE_SAMPLE_CAP + 6  # the count is the cell, not the sample


def test_sampling_is_deterministic_and_seeded(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _crowd(30)
    assert _prompts(rows) == _prompts(list(rows))
    seed0 = _prompts(rows)
    monkeypatch.setattr(analyze, "INSTANCE_SAMPLE_SEED", 1)
    assert _prompts(rows) != seed0


def test_prompt_masks_the_face_and_drops_an_empty_user_line() -> None:
    [prompt] = _prompts([_row(SMILE, "  hello", " asked ")])
    assert "[Sample 1]\nUser: asked\nAssistant: [FACE] hello" in prompt
    [bare] = _prompts([_row(SMILE, "hello", "  ")])
    assert "[Sample 1]\nAssistant: [FACE] hello" in bare and "User:" not in bare
    assert SMILE not in prompt


# --- plan_analyze -----------------------------------------------------------

def test_plan_counts_a_cold_run() -> None:
    plan = _plan(_rows())
    assert (plan.total_rows, plan.canonical_unique, plan.cell_count, plan.cached) == (5, 2, 3, 0)
    assert plan.providers_seen == ["test"] and plan.model_counts == {"m1": 4, "m2": 1}
    assert (plan.backend, plan.model_id) == (BACKEND, MODEL)
    assert plan.estimated_input_tokens > 0 and plan.estimated_output_tokens > 0


def test_plan_after_a_run_is_warm_and_costs_nothing(run: Callable[..., Any]) -> None:
    run(_rows())
    warm = _plan(_rows())
    assert (warm.cell_count, warm.cached) == (3, 3)
    assert warm.estimated_input_tokens == warm.estimated_output_tokens == 0

    extended = _plan([*_rows(), _row("(T_T)", "sob", "u9")])
    cold = _plan([_row("(T_T)", "sob", "u9")])
    assert (extended.cell_count, extended.cached) == (4, 3)
    assert extended.estimated_input_tokens == cold.estimated_input_tokens > 0
    assert extended.estimated_output_tokens == cold.estimated_output_tokens > 0


@pytest.mark.parametrize("backend, model", [
    ("anthropic", DEFAULT_ANTHROPIC_MODEL_ID), ("openai", DEFAULT_OPENAI_MODEL_ID),
])
def test_plan_names_the_pinned_model_for_hosted_backends(backend: str, model: str) -> None:
    plan = plan_analyze(_rows(), backend=backend)
    assert (plan.backend, plan.model_id) == (backend, model)


def test_plan_calls_no_backend_and_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("plan_analyze must not build a synthesizer")

    monkeypatch.setattr(analyze, "make_synthesizer", boom)
    assert plan_analyze(_rows()).cell_count == 3
    assert not paths.llmoji_home().exists()


# --- small things -----------------------------------------------------------

@pytest.mark.parametrize("explicit, env, expected", [
    (None, None, 1), (None, "3", 3), (None, "junk", 1), (None, "0", 1), (2, "5", 2), (0, None, 1),
])
def test_worker_count(monkeypatch: pytest.MonkeyPatch, explicit: int | None,
                      env: str | None, expected: int) -> None:
    if env is None:
        monkeypatch.delenv("LLMOJI_CONCURRENCY", raising=False)
    else:
        monkeypatch.setenv("LLMOJI_CONCURRENCY", env)
    assert analyze._workers(explicit) == expected


def test_progress_output_previews_the_bundle(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(analyze, "make_synthesizer", lambda backend, **k: FakeSynth())
    run_analyze(_rows(), print_progress=True)
    out = capsys.readouterr().out
    assert "3 cells across 2 source model(s); 0 cached, 3 to synthesize" in out
    assert "[m1] -> m1.jsonl (2 faces)" in out and "read each <model>.jsonl" in out
