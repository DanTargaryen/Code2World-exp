"""CoinRun code variants: each modifies one physics param in coinrun.cpp and
adds a [VARIANT] highlight comment so the code encoder notices the change.
"""

# variant_name -> (exact_original_line, replacement_line_with_comment)
# original lines copied verbatim from procgen/src/games/coinrun.cpp
VARIANTS = {
    "base": None,  # no modification
    "fast": (
        "        maxspeed = .5;",
        "        maxspeed = 0.9f;   // [VARIANT] changed from 0.5 (faster horizontal speed)",
    ),
    "slow": (
        "        maxspeed = .5;",
        "        maxspeed = 0.25f;  // [VARIANT] changed from 0.5 (slower horizontal speed)",
    ),
    "lowgrav": (
        "        gravity = 0.2f;",
        "        gravity = 0.1f;   // [VARIANT] changed from 0.2 (lower gravity, floaty)",
    ),
    "highgrav": (
        "        gravity = 0.2f;",
        "        gravity = 0.35f;  // [VARIANT] changed from 0.2 (higher gravity, heavy fall)",
    ),
    "highjump": (
        "        max_jump = 1.5;",
        "        max_jump = 2.5;   // [VARIANT] changed from 1.5 (higher jump)",
    ),
    "lowjump": (
        "        max_jump = 1.5;",
        "        max_jump = 0.9;   // [VARIANT] changed from 1.5 (lower jump)",
    ),
}

VARIANT_NAMES = list(VARIANTS.keys())


def make_variant_source(original_src: str, variant: str) -> str:
    """Return modified source for `variant`. Raises if the target line is missing."""
    spec = VARIANTS[variant]
    if spec is None:
        return original_src
    old, new = spec
    if old not in original_src:
        raise ValueError(f"variant {variant}: target line not found:\n{old}")
    if original_src.count(old) != 1:
        raise ValueError(f"variant {variant}: target line not unique ({original_src.count(old)}x)")
    return original_src.replace(old, new)
