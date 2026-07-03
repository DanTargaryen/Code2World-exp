"""Load a CoinRun game config (YAML) into procgen options.

The YAML is a *readable, declarative* description of CoinRun's tunable mechanics,
grouped by meaning (physics / reward / termination / hazards). It doubles as the
**code condition** for the world model: instead of feeding raw `coinrun.cpp`
(500+ lines, mechanics tangled with rendering/boilerplate), we feed this config,
which exposes exactly the game mechanics.

How it works end-to-end:
  YAML  --load_config-->  coinrun_config dict  --ProcgenEnv(coinrun_config=...)-->
  gym3 encodes each value as float32/bool option  -->  C++ game.cpp parse_options
  -->  options.coinrun_*  read in coinrun.cpp at runtime.

No recompile is needed to change these values (unlike the old variants.py, which
rewrote coinrun.cpp and rebuilt procgen per variant). Compile procgen once with
the options plumbing, then any config is a pure runtime parameter change.

Config layout (grouped; every key optional, omitted => vanilla CoinRun):
  physics:     gravity, max_jump, max_speed, air_control, mixrate   (float)
  reward:      goal_reward                                          (float)
  termination: die_on_enemy, die_on_saw, die_on_lava                (bool)
  hazards:     allow_pit, allow_crate, allow_monsters               (bool)

A flat `mechanics: {...}` block is also accepted (legacy) and merged in.
"""
import os
import yaml

# grouped schema: group -> {mechanic: (coinrun_option, type, vanilla_default, description)}
# description is used to render the full code-condition text fed to the encoder.
SCHEMA = {
    "physics": {
        "gravity":     ("coinrun_gravity",     float, 0.2,  "downward acceleration applied each step; higher = falls faster"),
        "max_jump":    ("coinrun_max_jump",    float, 1.5,  "upward jump impulse and vertical speed cap; higher = jumps higher"),
        "max_speed":   ("coinrun_max_speed",   float, 0.5,  "maximum horizontal running speed"),
        "air_control": ("coinrun_air_control", float, 0.15, "fraction of horizontal control retained while airborne (0=none, 1=full)"),
        "mixrate":     ("coinrun_mixrate",     float, 0.2,  "how quickly horizontal velocity blends toward the target when grounded"),
    },
    "reward": {
        "goal_reward": ("coinrun_goal_reward", float, 10.0, "reward granted when the player reaches the goal coin (ends the level)"),
    },
    "termination": {
        "die_on_enemy": ("coinrun_die_on_enemy", bool, True, "touching a walking monster ends the episode (death)"),
        "die_on_saw":   ("coinrun_die_on_saw",   bool, True, "touching a buzzsaw ends the episode (death)"),
        "die_on_lava":  ("coinrun_die_on_lava",  bool, True, "touching lava ends the episode (death)"),
    },
    "hazards": {
        "allow_pit":      ("coinrun_allow_pit",      bool, True, "level generation may carve pits (gaps with lava/saws/enemies at the bottom)"),
        "allow_crate":    ("coinrun_allow_crate",    bool, True, "level generation may place stackable crates as obstacles/platforms"),
        "allow_monsters": ("coinrun_allow_monsters", bool, True, "level generation may spawn moving monsters on the platforms"),
    },
}

# flat lookups derived from SCHEMA
MECHANIC = {m: spec for grp in SCHEMA.values() for m, spec in grp.items()}
MECHANIC_TO_OPTION = {m: spec[0] for m, spec in MECHANIC.items()}
VANILLA = {m: spec[2] for m, spec in MECHANIC.items()}
DESC = {m: spec[3] for m, spec in MECHANIC.items()}
GROUP_OF = {m: g for g, grp in SCHEMA.items() for m in grp}


class ConfigError(ValueError):
    pass


def _coerce(mech, value):
    """Cast value to the mechanic's declared type; raise ConfigError on mismatch."""
    typ = MECHANIC[mech][1]
    if typ is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"mechanic {mech!r} must be a bool (true/false), got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"mechanic {mech!r} must be a number, got {value!r}")


def load_config(path):
    """Parse a CoinRun YAML config into a normalized dict.

    Returns:
        {
          "name": <str>,
          "game": "coinrun",
          "coinrun_config": {<coinrun_* option>: float|bool, ...},  # only overrides
          "mechanics": {<mechanic>: value, ...},                    # merged w/ vanilla
          "raw": <str, exact YAML text = the code condition>,
        }
    """
    with open(path) as f:
        raw = f.read()
    spec = yaml.safe_load(raw) or {}

    game = spec.get("game", "coinrun")
    if game != "coinrun":
        raise ConfigError(f"only 'coinrun' is supported for now, got game={game!r}")

    # collect mechanics from grouped blocks + a legacy flat `mechanics` block
    incoming = {}
    for grp in SCHEMA:
        block = spec.get(grp, {}) or {}
        if not isinstance(block, dict):
            raise ConfigError(f"group {grp!r} must be a mapping, got {type(block).__name__}")
        for mech, value in block.items():
            if mech not in SCHEMA[grp]:
                raise ConfigError(f"unknown mechanic {mech!r} in group {grp!r}; "
                                  f"allowed: {sorted(SCHEMA[grp])}")
            incoming[mech] = value
    for mech, value in (spec.get("mechanics", {}) or {}).items():   # legacy flat
        if mech not in MECHANIC:
            raise ConfigError(f"unknown mechanic {mech!r}; allowed: {sorted(MECHANIC)}")
        incoming[mech] = value

    coinrun_config = {}
    merged = dict(VANILLA)
    for mech, value in incoming.items():
        v = _coerce(mech, value)
        coinrun_config[MECHANIC_TO_OPTION[mech]] = v
        merged[mech] = v

    name = spec.get("name") or os.path.splitext(os.path.basename(path))[0]
    return {
        "name": name,
        "game": game,
        "coinrun_config": coinrun_config,
        "mechanics": merged,
        "overrides": set(incoming),   # which mechanics the config explicitly set
        "raw": raw,
    }


def render_code_condition(cfg):
    """Render the FULL game rules as the code condition fed to the text encoder.

    A user config may be sparse (omitted mechanic => vanilla default), but the code
    condition must be a COMPLETE rulebook so the encoder actually 'reads' every
    mechanic — otherwise omitted rules carry no signal and code-sensitivity to them
    is impossible. This expands the merged config (all 12 mechanics, effective
    values) into a self-describing text: grouped, with each mechanic's value, its
    meaning, and whether this variant overrides the vanilla default.

    `cfg` is the dict returned by load_config.
    """
    merged = cfg["mechanics"]
    overrides = cfg.get("overrides", set())
    lines = [
        f"# Game: CoinRun  (variant: {cfg['name']})",
        "# A 2D platformer: run/jump rightward across procedurally generated terrain,",
        "# avoid hazards, reach the goal coin. The rules below fully specify this",
        "# variant's mechanics; values differing from vanilla are marked [CHANGED].",
        "",
    ]
    for group, mechs in SCHEMA.items():
        lines.append(f"[{group}]")
        for mech, spec in mechs.items():
            _, typ, default, desc = spec
            val = merged[mech]
            val_str = ("true" if val else "false") if typ is bool else f"{val:g}"
            def_str = ("true" if default else "false") if typ is bool else f"{default:g}"
            mark = f"  [CHANGED from {def_str}]" if mech in overrides else ""
            lines.append(f"  {mech} = {val_str}  # {desc}{mark}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        cfg = load_config(p)
        print(f"[{cfg['name']}] game={cfg['game']}  overrides={sorted(cfg['overrides'])}")
        print("--- code condition (fed to text encoder) ---")
        print(render_code_condition(cfg))

