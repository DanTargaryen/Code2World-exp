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

# grouped schema: group -> {mechanic: (coinrun_option, type, vanilla_default)}
SCHEMA = {
    "physics": {
        "gravity":     ("coinrun_gravity",     float, 0.2),
        "max_jump":    ("coinrun_max_jump",    float, 1.5),
        "max_speed":   ("coinrun_max_speed",   float, 0.5),
        "air_control": ("coinrun_air_control", float, 0.15),
        "mixrate":     ("coinrun_mixrate",     float, 0.2),
    },
    "reward": {
        "goal_reward": ("coinrun_goal_reward", float, 10.0),
    },
    "termination": {
        "die_on_enemy": ("coinrun_die_on_enemy", bool, True),
        "die_on_saw":   ("coinrun_die_on_saw",   bool, True),
        "die_on_lava":  ("coinrun_die_on_lava",  bool, True),
    },
    "hazards": {
        "allow_pit":      ("coinrun_allow_pit",      bool, True),
        "allow_crate":    ("coinrun_allow_crate",    bool, True),
        "allow_monsters": ("coinrun_allow_monsters", bool, True),
    },
}

# flat lookups derived from SCHEMA
MECHANIC = {m: spec for grp in SCHEMA.values() for m, spec in grp.items()}
MECHANIC_TO_OPTION = {m: spec[0] for m, spec in MECHANIC.items()}
VANILLA = {m: spec[2] for m, spec in MECHANIC.items()}
GROUP_OF = {m: g for g, grp in SCHEMA.items() for m in grp}


class ConfigError(ValueError):
    pass


def _coerce(mech, value):
    """Cast value to the mechanic's declared type; raise ConfigError on mismatch."""
    _, typ, _ = MECHANIC[mech]
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
        "raw": raw,
    }


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        cfg = load_config(p)
        print(f"[{cfg['name']}] game={cfg['game']}")
        print(f"  overrides -> {cfg['coinrun_config']}")
        print(f"  merged    -> {cfg['mechanics']}")
