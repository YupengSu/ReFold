"""The built-in strategies. Importing this package registers them.

    base     nothing is removed (mini-swe-agent's own append-only loop)
    compact  an extra model call summarises old history at 75% of the budget
    refold   deduplication and folding at the rendering layer

`--compact` runs ReFold beneath compaction (compact + ReFold).
"""

from refold.strategies import base, compact, refold  # noqa: F401
