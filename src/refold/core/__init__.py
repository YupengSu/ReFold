"""ReFold itself, with no agent framework underneath.

    engine.py   OBSERVE / RENDER / restore over a chat-message history (ReFoldEngine)
    dedup.py    deduplication: repeated reads, searches, outputs, passing test lines
    fold.py     folding: runs of STALE steps behind the boundary, each to one note
    parse.py    commands (reads, searches, tests, restore) and the PROGRESS / STALE lines
    filters.py  test-output filters
    prompts.json  the prompt layer and every note

Adapter: refold.strategies.refold.agent (mini-swe-agent, the harness of the paper's runs).
"""
