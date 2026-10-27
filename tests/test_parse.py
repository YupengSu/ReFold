"""Command and reply parsing, and the test-output filters."""

from __future__ import annotations

import pytest

from refold.core import filters, parse
from refold.core.engine import PROMPTS
from refold.core.parse import EOF


@pytest.mark.parametrize("cmd,path,start,end,tail", [
    ("cat a.py", "a.py", 1, EOF, None),
    ("cat -n src/a.py", "src/a.py", 1, EOF, None),
    ("head -n 40 a.py", "a.py", 1, 40, None),
    ("head -40 a.py", "a.py", 1, 40, None),
    ("tail -n 5 a.py", "a.py", 1, EOF, 5),
    ("sed -n '10,20p' a.py", "a.py", 10, 20, None),
    ("awk 'NR>=3 && NR<=9' a.py", "a.py", 3, 9, None),
    ("nl -ba a.py | sed -n '5,8p'", "a.py", 5, 8, None),
    ("cat a.py | head -n 7", "a.py", 1, 7, None),
    ("cd /repo && cat a.py", "a.py", 1, EOF, None),
])
def test_file_reads_are_recognised(cmd, path, start, end, tail):
    s = parse.parse_read_command(cmd)
    assert (s.path, s.line_start, s.line_end, s.tail_n) == (path, start, end, tail)


@pytest.mark.parametrize("cmd", ["cat a.py b.py", "cat a.py > b", "grep x a.py", "cat a.py | grep x",
                                 "head a.py", "sed -i 's/a/b/' a.py", "cat a.py; ls"])
def test_anything_else_is_not_a_read(cmd):
    assert parse.parse_read_command(cmd) is None


def test_searches_are_canonicalised():
    assert parse.parse_search_command("cd /r && grep -rn 'x y' src/ 2>&1 | head -20") == "grep -rn x y src/"
    assert parse.parse_search_command("find . -name '*.py'") == "find . -name *.py"
    assert parse.parse_search_command("grep -rn x src | tee out") is None
    assert parse.parse_search_command("grep -rn x src > out") is None


@pytest.mark.parametrize("cmd,yes", [
    ("pytest tests/test_a.py", True), ("cd /r && python -m pytest -x", True), ("FOO=1 py.test", True),
    ("pip install pytest", False), ("python run.py", False)])
def test_pytest_runs_are_detected(cmd, yes):
    assert parse.is_pytest_command(cmd) is yes


def test_test_runners_are_detected():
    assert parse.is_testrunner_command("cd x && go test ./...")
    assert parse.is_testrunner_command("npm run test")
    assert not parse.is_testrunner_command("go build ./...")


def test_progress_is_the_rest_of_the_progress_line():
    assert parse.parse_progress("\n\nPROGRESS: reading models.py\nSTALE: 3") == "reading models.py"
    assert parse.parse_progress("PROGRESS - `reading models.py`") == "reading models.py"
    assert parse.parse_progress("PROGRESS: Main branch pushed") == "Main branch pushed"  # no label to strip
    assert parse.parse_progress("THOUGHT: nothing here") is None and parse.parse_progress(None) is None


def test_stale_ids_come_only_from_the_first_lines():
    assert parse.parse_stale("PROGRESS: x\nSTALE: 3, 4") == [3, 4]
    assert parse.parse_stale("PROGRESS: x\nSTALE: -") == []
    assert parse.parse_stale("PROGRESS: x\nSTALE: [1 2]") == [1, 2]
    assert parse.parse_stale("a\nb\nc\nd\nSTALE: 3") == []  # past the metadata lines


def test_restore_is_recognised_only_as_the_whole_command():
    assert parse.parse_restore_command("restore 12") == [12]
    assert parse.parse_restore_command("  restore 3-5, 9 ") == [3, 4, 5, 9]
    assert parse.parse_restore_command("restore step 4") == [4]
    for cmd in ("restore", "restore -rf dump.img", "cat x && restore 3", "restore 3; ls", "echo restore 3"):
        assert parse.parse_restore_command(cmd) is None, cmd


def test_observation_lines_strip_cat_n_numbering():
    obs = "<returncode>0</returncode>\n<output>\n     1\tfoo\n     2\tbar\n</output>"
    assert parse.observation_lines(obs, used_cat_n=True) == ["foo\n", "bar\n"]


def test_pytest_filter_drops_passes_and_keeps_failures():
    lines = ["===== test session starts =====", "platform linux", "rootdir: /r"]
    lines += [f"tests/test_a.py::test_{i} PASSED" for i in range(30)]
    lines += ["tests/test_a.py::test_bad FAILED", "E   assert 1 == 2", "tests/test_a.py:12: AssertionError",
              "===== 1 failed, 30 passed in 0.5s ====="]
    out, kept, total = filters.filter_pytest_output("\n".join(lines))
    assert total == 37 and kept == len(out.splitlines()) < 10
    assert "test_3 PASSED" not in out and "test_bad FAILED" in out and "E   assert 1 == 2" in out


def test_test_runner_filter_folds_only_passing_lines():
    runs = [f"=== RUN   TestAlpha/subtest_{i}\n--- PASS: TestAlpha/subtest_{i} (0.00s)" for i in range(6)]
    obs = "\n".join([*runs, "--- FAIL: TestB (0.01s)", "FAIL"])
    out = filters.filter_test_output(obs, PROMPTS["note_tests_run"])
    assert out.splitlines() == ["[context-manager] 12 passing/progress lines omitted.", "--- FAIL: TestB (0.01s)", "FAIL"]


def test_too_little_noise_is_left_alone():
    assert filters.filter_test_output("=== RUN   A\n--- PASS: A (0.0s)\nok", PROMPTS["note_tests_run"]) is None
    assert filters.filter_pytest_output("one line") is None
