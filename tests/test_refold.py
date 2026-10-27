"""ReFold's mechanisms, each pinned by what the model would actually be sent."""

from __future__ import annotations

import copy
import re

import pytest
from helpers import make_agent

from refold.strategies.refold.agent import PROMPTS, ReFoldAgent


def run(script, cwd, **cfg):
    agent = make_agent(ReFoldAgent, script, cwd, **cfg)
    agent.run("Investigate the modules.")
    return agent


def text(prompt) -> str:
    return "\n".join(m["content"] for m in prompt if isinstance(m.get("content"), str))


def tool_msg(prompt, step: int) -> str | None:
    """The rendered tool message labelled [step N] (fold notes may precede the label),
    or None if the step was folded away."""
    for m in prompt:
        if m.get("role") == "tool" and (m["content"].startswith(f"[step {step}]\n")
                                        or f"\n[step {step}]\n" in m["content"]):
            return m["content"]
    return None


def read(i):
    return (f"PROGRESS: read module {i}\nSTALE: -", f"cat mod{i:02d}.py")


def filler(n):
    return [(f"PROGRESS: filler {j}\nSTALE: -", f"echo filler-{j}") for j in range(n)]


# --------------------------------------------------------------- the boundary b_t
@pytest.mark.parametrize("k,expected", [
    (3, [0, 0, 0, 0, 0, 0, 3, 3, 3, 6, 6, 6, 9]),
    (1, [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
])
def test_boundary_advances_k_turns_every_k_steps(tmp_path, k, expected):
    agent = make_agent(ReFoldAgent, [("x", "true")], tmp_path, chunk_size=k)
    got = []
    for t in range(len(expected)):
        agent.turns = list(range(t))
        got.append(agent.boundary())
    assert got == expected


def test_unchunked_the_boundary_moves_every_step(tmp_path):
    agent = make_agent(ReFoldAgent, [("x", "true")], tmp_path, chunked=False)
    got = []
    for t in range(9):
        agent.turns = list(range(t))
        got.append(agent.boundary())
    assert got == [0, 0, 0, 0, 1, 2, 3, 4, 5]


def test_a_repeat_renders_full_until_the_boundary_passes_it_then_stubs(corpus):
    # turn 0 reads mod00, turn 1 reads it again; b reaches 3 once there are 6 turns
    agent = run([read(0), read(0), *filler(8)], corpus)
    before = [tool_msg(p, 1) for p in agent.model.prompts[2:6]]  # 2..5 turns: b = 0
    after = tool_msg(agent.model.prompts[6], 1)  # 6 turns: b = 3
    assert all("[context-manager]" not in m for m in before)
    assert after.startswith(f"[step 1]\n[context-manager] {corpus}/mod00.py:1-70 omitted: identical to step 0. "
                            "`restore 1` shows it again.\n")
    assert "\n  def mod00_op_0(value):" in after  # >= 20 lines: a signature skeleton, as detail lines
    assert "<returncode>" not in after  # a note is not dressed up as shell output


def test_between_advances_the_prompt_only_grows_at_the_end(corpus):
    """Call t is sent with t turns. Whenever b_t == b_(t-1), the previous prompt must be a
    prefix of this one -- the property the prefix cache depends on."""
    agent = run([read(i % 4) for i in range(14)], corpus)
    prompts, k, checked = agent.model.prompts, 3, 0
    for t in range(1, len(prompts)):
        if max(0, k * ((t - 1 - k) // k)) == max(0, k * ((t - k) // k)):
            assert prompts[t][:len(prompts[t - 1])] == prompts[t - 1], f"prefix broken at call {t}"
            checked += 1
    assert "[context-manager]" in text(prompts[-1]) and checked >= 8


def test_a_repeat_of_a_stubbed_range_renders_full_and_stays_full(corpus):
    agent = run([read(0), read(0), read(0), *filler(9)], corpus)
    last = agent.model.prompts[-1]
    assert "[context-manager]" in tool_msg(last, 1)
    assert "[context-manager]" not in tool_msg(last, 2)  # the re-read after a stub yields


def test_a_repeat_of_a_partly_shown_range_renders_full_and_stays_full(corpus):
    script = [("PROGRESS: head\nSTALE: -", "sed -n '1,30p' mod00.py"), read(0), *filler(5), read(0), *filler(9)]
    agent = run(script, corpus)
    last = agent.model.prompts[-1]
    assert agent.boundary() > 7 and "lines already shown" in tool_msg(last, 1)
    assert "[context-manager]" not in tool_msg(last, 7)  # the re-read after a partial yields too


def test_a_partial_repeat_keeps_new_lines_and_marks_the_rest(corpus):
    script = [("PROGRESS: head\nSTALE: -", "sed -n '1,30p' mod00.py"), read(0), *filler(8)]
    agent = run(script, corpus)
    msg = tool_msg(agent.model.prompts[-1], 1)
    assert msg.startswith("[step 1]\n[context-manager] lines already shown at earlier steps are omitted below. "
                          "`restore 1` shows all of it again.\n<returncode>0</returncode>\n<output>\n")
    assert "[context-manager] lines 1-30 omitted: identical to step 0.\n" in msg and "def mod00_op_20(value)" in msg


# --------------------------------------------------------------- folding (Eqs. 4, 5)
def test_stale_is_accepted_only_within_the_last_k_steps(corpus):
    script = [read(0), read(1), read(2), read(3),
              ("PROGRESS: done with 1 and 2\nSTALE: 1, 2, 0", "echo x"), *filler(6)]
    agent = run(script, corpus)
    assert agent.stale == {1, 2} and agent.labels["stale_out_of_window"] == 1


def test_a_fold_removes_reply_and_output_and_leaves_a_note(corpus):
    script = [read(0), read(1), read(2), ("PROGRESS: done with the reads\nSTALE: 1, 2", "echo x"), *filler(6)]
    agent = run(script, corpus)
    first_fold = next(i for i, p in enumerate(agent.model.prompts) if "steps 1-2 folded" in text(p))
    assert first_fold == 6  # the boundary reaches 3 at 6 turns, not when STALE was written
    p = agent.model.prompts[-1]
    assert tool_msg(p, 1) is None and tool_msg(p, 2) is None
    assert not any("read module 1" in (m.get("content") or "") for m in p if m["role"] == "assistant")
    note = tool_msg(p, 3)
    assert note.startswith("[context-manager] steps 1-2 folded: read module 1 · read module 2. "
                           "`restore 1-2` shows them again.\n  mod01.py: def mod01_op_0(value):")
    assert note.index("steps 1-2 folded") < note.index("[step 3]")


# --------------------------------------------------------------- reversibility
def test_restore_appends_a_folded_turn_exactly_as_first_sent_and_runs_nothing(corpus):
    script = [read(0), read(1), read(2), ("PROGRESS: done with the reads\nSTALE: 1, 2", "echo x"), *filler(3),
              ("PROGRESS: need module 2 again\nSTALE: -", "restore 2"), *filler(6)]
    agent = run(script, corpus)
    before, after = agent.model.prompts[7], agent.model.prompts[8]  # the restore reply is call 7
    assert tool_msg(before, 2) is None and after[:len(before)] == before  # appended: the prefix holds
    original = agent.messages[agent.tool_msg[2]]["content"]
    # z_i = a_i ⊕ o_i: the agent message that issued step 2, its command, its tool output
    expected = ("[step 7]\n[context-manager] step 2 as first shown: its agent message, command and tool "
                "observation. Re-run the command for its current state.\n"
                f"PROGRESS: read module 2\nSTALE: -\n$ cat mod02.py\n{original}")
    assert tool_msg(after, 7) == expected  # the logged bytes, not a shell's `restore: not found`
    last = agent.model.prompts[-1]
    assert agent.boundary() > 7 and tool_msg(last, 7) == expected  # and it stays that way past the boundary
    assert tool_msg(last, 2) is None  # step 2 itself stays folded
    assert agent.restores == {"calls": 1, "folded": 1, "compressed": 0, "shown": 0, "invalid": 0}


def test_restore_notes_a_step_shown_in_full_or_unknown_and_caps_a_range(corpus):
    script = [read(0), ("PROGRESS: x\nSTALE: -", "restore 0"), ("PROGRESS: x\nSTALE: -", "restore 42"),
              ("PROGRESS: x\nSTALE: -", "restore 0-11"), *filler(1)]
    agent = run(script, corpus)
    last = agent.model.prompts[-1]
    assert tool_msg(last, 1) == "[step 1]\n[context-manager] step 0 is shown in full above: nothing to restore."
    assert tool_msg(last, 2) == "[step 2]\n[context-manager] no step 42: steps so far are 0-1."
    assert tool_msg(last, 3).endswith("\n\n[context-manager] at most 10 steps per restore: the others were not restored.")
    assert agent.restores == {"calls": 3, "folded": 0, "compressed": 0, "shown": 4, "invalid": 8}


def test_a_reissued_folded_command_runs_again_and_shows_in_full(corpus):
    script = [read(0), read(1), read(2), ("PROGRESS: done with the reads\nSTALE: 1, 2", "echo x"), *filler(3),
              ("PROGRESS: need module 1 again\nSTALE: -", "cat mod01.py"), *filler(6)]
    agent = run(script, corpus)
    assert tool_msg(agent.model.prompts[7], 1) is None  # the reply at call 7 was written with step 1 folded
    last = agent.model.prompts[-1]
    assert agent.boundary() > 7 and tool_msg(last, 1) is None  # appended, not un-folded
    assert tool_msg(last, 7) == f"[step 7]\n{agent.messages[agent.tool_msg[7]]['content']}"
    assert "def mod01_op_0(value)" in tool_msg(last, 7)  # its only other copy is folded: nothing to stub against
    assert agent.reruns == {"folded": 1, "folded_identical": 1, "compressed": 0, "compressed_identical": 0}


def test_a_reissue_after_an_edit_shows_the_current_output(corpus):
    script = [read(0), read(1), read(2), ("PROGRESS: done\nSTALE: 1", "echo x"),
              ("PROGRESS: edit\nSTALE: -", "echo '# changed' >> mod01.py"), *filler(2),
              ("PROGRESS: check\nSTALE: -", "cat mod01.py"), *filler(3)]
    agent = run(script, corpus)
    assert "# changed" in tool_msg(agent.model.prompts[-1], 7)
    assert agent.reruns["folded"] == 1 and agent.reruns["folded_identical"] == 0


def test_a_reissued_deduplicated_command_is_counted_but_a_plain_repeat_is_not(corpus):
    script = [read(0), read(0), *filler(5), ("PROGRESS: need the copy\nSTALE: -", "cat mod00.py"), *filler(3)]
    agent = run(script, corpus)
    assert "omitted: identical to step 0" in tool_msg(agent.model.prompts[7], 1)
    assert "[context-manager]" not in tool_msg(agent.model.prompts[-1], 7)
    # step 1 repeated step 0 while step 0 was visible: a repeat, not a recovery
    assert agent.reruns == {"folded": 0, "folded_identical": 0, "compressed": 1, "compressed_identical": 1}


def test_irreversible_offers_no_restore_and_no_note_says_how_to_get_back(corpus):
    script = [read(0), read(0), *filler(5), ("PROGRESS: need it\nSTALE: -", "restore 1"), *filler(3)]
    agent = run(script, corpus, reversible=False)
    assert "restore" not in agent.messages[0]["content"] + agent.messages[1]["content"]
    last = agent.model.prompts[-1]
    assert tool_msg(last, 1).split("\n")[1] == (f"[context-manager] {corpus}/mod00.py:1-70 omitted: "
                                                 "identical to step 0.")
    assert "restore" not in "".join(m["content"] for m in last if m["role"] == "tool" and "[context-manager]" in m["content"]
                                    and "restore 1" not in m["content"])
    assert "<returncode>0</returncode>" not in tool_msg(last, 7) and agent.restores["calls"] == 0  # went to the shell


def test_irreversible_compresses_a_repeat_again(corpus):
    agent = run([read(0), read(0), *filler(5), read(0), *filler(9)], corpus, reversible=False)
    last = agent.model.prompts[-1]
    assert agent.boundary() > 7 and "omitted: identical to step 0" in tool_msg(last, 7)  # no yield


def test_irreversible_keeps_folded_steps_as_dedup_sources(corpus):
    script = [read(0), read(1), read(2), ("PROGRESS: done\nSTALE: 1", "echo x"), *filler(2), read(1), *filler(6)]
    agent = run(script, corpus, reversible=False)
    last = agent.model.prompts[-1]
    assert tool_msg(last, 1) is None and "omitted: identical to step 1" in tool_msg(last, 6)  # names a folded step


def test_a_reread_of_folded_content_is_not_stubbed_against_it(corpus):
    script = [read(0), read(1), read(2), ("PROGRESS: done\nSTALE: 1", "echo x"), *filler(2), read(1), *filler(6)]
    agent = run(script, corpus)
    assert tool_msg(agent.model.prompts[-1], 1) is None  # folded
    assert "[context-manager]" not in tool_msg(agent.model.prompts[-1], 6)  # its only copy was folded


def test_fold_off_parses_stale_but_removes_nothing(corpus):
    script = [read(0), read(1), read(2), ("PROGRESS: done\nSTALE: 1, 2", "echo x"), *filler(6)]
    agent = run(script, corpus, fold=False)
    assert agent.stale == {1, 2}
    assert " folded: " not in text(agent.model.prompts[-1])
    assert tool_msg(agent.model.prompts[-1], 1) is not None


def test_dedup_off_stubs_nothing(corpus):
    agent = run([read(0), read(0), *filler(8)], corpus, dedup=False)
    assert "omitted: identical to" not in text(agent.model.prompts[-1])


# --------------------------------------------------------------- the prompt layer
def test_the_protocol_replaces_the_thought_item_and_names_k(corpus):
    agent = run([read(0)], corpus, chunk_size=5)
    instance = agent.messages[1]["content"]
    assert "Include a THOUGHT section" not in instance
    assert "PROGRESS: <one short line on what THIS step does>" in instance
    assert "Only the last 5 steps before this one can still be named" in instance
    assert agent.messages[0]["content"].count("[context-manager]") == 1


def test_every_observation_is_labelled_with_its_step(corpus):
    agent = run([read(0), read(1), read(2)], corpus)
    tools = [m for m in agent.model.prompts[-1] if m["role"] == "tool"]
    assert [m["content"].split("\n", 1)[0] for m in tools] == ["[step 0]", "[step 1]"]


def test_a_skipped_header_gets_a_reminder_that_escalates(corpus):
    script = [read(0), ("no header", "echo a"), ("still none", "echo b"), read(1)]
    agent = run(script, corpus)
    p2, p3 = agent.model.prompts[2], agent.model.prompts[3]
    assert p2[-1]["content"].endswith("\n\n" + PROMPTS["remind_gentle"])
    assert p3[-1]["content"].endswith("\n\n" + PROMPTS["remind_harsh"].format(misses=2))
    assert PROMPTS["remind_gentle"] not in text(agent.model.prompts[1])


def test_the_reminder_asks_for_the_protocol_the_model_was_given():
    for key in ("remind_gentle", "remind_harsh"):
        assert "STALE" in PROMPTS[key] and "main|side" not in PROMPTS[key]


# --------------------------------------------------------------- the log
def test_rendering_never_edits_the_history(corpus):
    agent = make_agent(ReFoldAgent, [read(0), read(0), read(1), ("PROGRESS: x\nSTALE: 1", "echo x"), *filler(6)],
                       corpus)
    snapshots = []
    real_render = agent.render

    def render():
        before = copy.deepcopy(agent.messages)
        out = real_render()
        snapshots.append(before == agent.messages)
        return out

    agent.render = render
    agent.run("go")
    assert snapshots and all(snapshots)


def test_the_trajectory_records_what_the_mechanism_did(corpus):
    agent = run([read(0), read(0), read(1), ("PROGRESS: x\nSTALE: 1", "echo x"), *filler(6)], corpus)
    info = agent.serialize()["info"]
    rec = info["refold"]
    assert rec["stale"] == [1] and rec["folds"][0]["start"] == 1 and rec["steps"] == 10
    assert rec["refetches"] == {"folded": 0, "folded_identical": 0, "compressed": 0, "compressed_identical": 0}
    assert info["config"]["agent"]["chunk_size"] == 3 and info["config"]["agent"]["chunked"] is True
    replies = [m for m in agent.messages if m["role"] == "assistant"]
    assert all(m["extra"]["overhead_s"] >= 0 for m in replies)  # the per-step overhead


# --------------------------------------------------------------- test output and paths
@pytest.fixture
def runners(corpus):
    """Fake `pytest` and `jest` executables printing a long passing run and one failure."""
    lines = ["===== test session starts =====", "platform linux", "rootdir: /r"]
    lines += [f"tests/test_a.py::test_{i} PASSED" for i in range(40)]
    lines += ["tests/test_a.py::test_bad FAILED", "E   assert 1 == 2", "===== 1 failed, 40 passed in 0.5s ====="]
    (corpus / "pytest").write_text("#!/bin/sh\ncat <<'X'\n" + "\n".join(lines) + "\nX\n")
    jest = [f"  ✓ renders case {i} (3 ms)" for i in range(40)] + ["  ✕ breaks (4 ms)", "Tests: 1 failed, 40 passed"]
    (corpus / "jest").write_text("#!/bin/sh\ncat <<'X'\n" + "\n".join(jest) + "\nX\n")
    for f in ("pytest", "jest"):
        (corpus / f).chmod(0o755)
    return corpus


PYTEST = ("PROGRESS: run the tests\nSTALE: -", "PATH=$PWD:$PATH pytest tests/test_a.py")
JEST = ("PROGRESS: run the js tests\nSTALE: -", "PATH=$PWD:$PATH jest")


def test_test_output_is_filtered_only_behind_the_boundary_and_a_rerun_yields(runners):
    agent = run([PYTEST, *filler(5), PYTEST, *filler(6)], runners)
    prompts = agent.model.prompts
    assert all("test_7 PASSED" in tool_msg(p, 0) for p in prompts[1:6])  # the newest turns are raw
    filtered = tool_msg(prompts[6], 0)
    assert re.match(r"\[step 0\]\n\[context-manager\] \d+ of \d+ lines omitted: passing tests, progress dots "
                    r"and warnings\. `restore 0` shows them again\.\n<returncode>0</returncode>", filtered)
    assert "test_7 PASSED" not in filtered and "test_bad FAILED" in filtered
    rerun = tool_msg(prompts[-1], 6)  # past the boundary by now, and a re-run of a filtered command
    assert agent.boundary() > 6 and "test_7 PASSED" in rerun and "[context-manager]" not in rerun


def test_restore_brings_back_what_a_test_filter_dropped(runners):
    agent = run([PYTEST, *filler(5), ("PROGRESS: need the passes\nSTALE: -", "restore 0"), *filler(3)], runners)
    assert "test_7 PASSED" not in tool_msg(agent.model.prompts[6], 0)
    assert tool_msg(agent.model.prompts[-1], 6).endswith(agent.messages[agent.tool_msg[0]]["content"])
    assert agent.restores["compressed"] == 1


def test_filtered_runner_output_keeps_its_own_frame_only(runners):
    agent = run([JEST, *filler(8)], runners)
    msg = tool_msg(agent.model.prompts[-1], 0)
    assert msg.startswith("[step 0]\n[context-manager] passing and progress lines omitted. "
                          "`restore 0` shows them again.\n<returncode>0</returncode>")
    assert msg.count("<returncode>") == 1 and "\n[context-manager] 40 passing/progress lines omitted.\n" in msg
    assert "✕ breaks" in msg


def test_a_cd_in_one_command_does_not_carry_over(corpus):
    (corpus / "sub").mkdir()
    (corpus / "sub" / "mod00.py").write_text((corpus / "mod00.py").read_text())  # same bytes, other file
    script = [("PROGRESS: read sub\nSTALE: -", "cd sub && cat mod00.py"),
              ("PROGRESS: read top\nSTALE: -", "cat mod00.py"), *filler(8)]
    agent = run(script, corpus)
    assert [t.read.path for t in agent.turns[:2]] == [str(corpus / "sub" / "mod00.py"), str(corpus / "mod00.py")]
    assert "[context-manager]" not in tool_msg(agent.model.prompts[-1], 1)  # a different file, not a repeat


# --------------------------------------------------------------- one grammar for every note
def test_every_note_the_harness_writes_follows_one_grammar(runners):
    """Anything the harness puts where output was starts with `[context-manager]`. A note
    standing for a whole step ends with the `restore` that shows it again, and a step it
    names as holding the same content is in the same prompt; the only notes without that
    clause are the markers inside a partly omitted output, under such a note, and the
    answers to `restore` itself."""
    search = ("PROGRESS: search\nSTALE: -", "grep -rn op_3 mod01.py")
    script = [read(0), read(0), ("PROGRESS: head\nSTALE: -", "sed -n '1,30p' mod01.py"), read(1), search, search,
              PYTEST, JEST, read(2), ("PROGRESS: done\nSTALE: 8", "cat mod03.py"), ("no header", "echo a"),
              *filler(2), ("PROGRESS: back\nSTALE: -", "restore 8"), ("PROGRESS: x\nSTALE: -", "restore 9"),
              ("PROGRESS: x\nSTALE: -", "restore 99"), *filler(7)]
    agent = run(script, runners)
    inline = {"read run": r"lines \d+-\d+ omitted: identical to step \d+\.",
              "test run": r"\d+ passing/progress lines? omitted\."}
    answers = {"restored": r"step \d+ as first shown: its agent message, command and tool observation\. "
                           r"Re-run the command for its current state\.",
               "restore shown": r"step \d+ is shown in full above: nothing to restore\.",
               "restore unknown": r"no step \d+: steps so far are 0-\d+\."}
    whole = {"search": "search output omitted", "read": "omitted: identical to step", "partial": "lines already shown",
             "pytest": "lines omitted: passing", "tests": "passing and progress lines omitted", "fold": " folded: ",
             "reminder": "metadata lines"}  # first match wins: search before read
    clause = r"`restore ([\d-]+)` shows (it|them|all of it) again\."
    seen = set()
    for p in agent.model.prompts:
        visible = {int(x) for x in re.findall(r"(?m)^\[step (\d+)\]$", text(p))}
        for m in (m for m in p if m["role"] == "tool"):
            step = None
            for line in m["content"].split("\n"):
                if re.fullmatch(r"\[step \d+\]", line):
                    step = int(line[6:-1])
                    continue
                if not line.startswith("["):
                    continue
                assert line.startswith("[context-manager] "), line
                body = line[len("[context-manager] "):]
                kind = next((k for k, rx in {**inline, **answers}.items() if re.fullmatch(rx, body)), None)
                if kind is None:
                    kind = next(k for k, w in whole.items() if w in body)
                    ids = re.search(clause + "$", body)
                    assert kind == "reminder" or ids, line
                    if kind == "fold":
                        assert ids.group(1) == re.match(r"steps? ([\d-]+) folded", body).group(1), line
                    elif kind != "reminder":
                        assert ids.group(1) == str(step), line
                assert all(int(x) in visible and int(x) < step for x in re.findall(r"identical to step (\d+)", body)), line
                seen.add(kind)
    assert seen == set(inline) | set(answers) | set(whole), seen


# --------------------------------------------------------------- the E4 / E5 knobs
def test_the_stale_window_can_be_set_apart_from_k(corpus):
    script = [read(0), read(1), read(2), read(3), read(4), ("PROGRESS: done\nSTALE: 0, 2, 3, 4", "echo x"), *filler(2)]
    assert run(script, corpus, stale_window=1).stale == {4}  # only the step just before
    assert run(script, corpus, stale_window=0).stale == {0, 2, 3, 4}  # any earlier step
    assert run(script, corpus).stale == {2, 3, 4}  # k = 3, the default
    instance = run([read(0)], corpus, stale_window=0).messages[1]["content"]
    assert "Any earlier step can still be named." in instance and "Only the last" not in instance
    assert "Only the last 6 steps before this one" in run([read(0)], corpus, stale_window=6).messages[1]["content"]


def test_random_stale_folds_as_many_new_steps_as_the_agent_named_and_is_seeded(corpus):
    script = [read(0), read(1), read(2), read(3), ("PROGRESS: done\nSTALE: 3", "echo x"), *filler(3)]
    a = run(script, corpus, stale_source="random", seed=0)
    assert len(a.stale) == 1 and a.stale <= {1, 2, 3} and a.agent_stale == {3}  # one new step, from the window
    assert run(script, corpus, stale_source="random", seed=0).stale == a.stale  # seeded
    assert any(run(script, corpus, stale_source="random", seed=s).stale != a.stale for s in range(1, 8))
    assert a.serialize()["info"]["refold"]["agent_stale"] == [3]


def test_all_stale_names_every_step_in_the_window(corpus):
    script = [read(0), read(1), read(2), read(3), ("PROGRESS: done\nSTALE: -", "echo x"), *filler(2)]
    assert run(script, corpus, stale_source="all").stale == set(range(6))  # all but the last step, never named
    assert "agent_stale" not in run(script, corpus).serialize()["info"]["refold"]
