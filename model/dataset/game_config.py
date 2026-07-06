"""Load the CoinRun game spec (YAML) into procgen options.

The YAML is a full, readable rewrite of coinrun.cpp — world / player physics /
level generation / entities / collision / termination — that doubles as BOTH:
  1. the **code condition** fed to the text encoder (the whole file, verbatim), and
  2. the source of the **executable mechanic values** that drive procgen collection.

Mechanic values live at fixed PATHS inside the spec (nested where they belong
semantically, e.g. gravity under player.physics.vertical). extract_mechanics()
reads those paths into a coinrun_config dict; ProcgenEnv(coinrun_config=...) then
applies them at collection time — so changing a value in the spec changes both the
generated data AND the code condition, keeping them in sync.

Only the paths in MECHANICS are executable; everything else in the spec is
descriptive (read by the encoder, not enforced by the extractor — the C++ enforces
the actual dynamics, and the descriptive text must stay faithful to it).
"""
import os
import yaml

# mechanic -> (path in the YAML, procgen option, type, vanilla default, description)
# path is a tuple of nested keys. The value at that path is the effective value.
MECHANICS = {
    "gravity":       (("player", "physics", "vertical", "gravity"),        "coinrun_gravity",       float, 0.2,  "downward acceleration per step"),
    "max_jump":      (("player", "physics", "vertical", "max_jump"),       "coinrun_max_jump",      float, 1.5,  "jump impulse / vertical speed cap"),
    "max_speed":     (("player", "physics", "horizontal", "max_speed"),    "coinrun_max_speed",     float, 0.5,  "max horizontal running speed"),
    "air_control":   (("player", "physics", "horizontal", "air_control"),  "coinrun_air_control",   float, 0.15, "horizontal control retained airborne"),
    "mixrate":       (("player", "physics", "horizontal", "mixrate"),      "coinrun_mixrate",       float, 0.2,  "velocity blend rate when grounded"),
    "goal_reward":   (("termination", "reward", "goal_reward"),            "coinrun_goal_reward",   float, 10.0, "reward for reaching the goal"),
    "die_on_lava":   (("collision_rules", "lethal", "lava"),               "coinrun_die_on_lava",   bool,  True, "lava contact ends the episode"),
    "die_on_saw":    (("collision_rules", "lethal", "saw"),                "coinrun_die_on_saw",    bool,  True, "saw contact ends the episode"),
    "die_on_enemy":  (("collision_rules", "lethal", "monster"),            "coinrun_die_on_enemy",  bool,  True, "monster contact ends the episode"),
    "allow_pit":     (("level_generation", "enabled_hazards", "pit"),      "coinrun_allow_pit",     bool,  True, "generation may carve pits"),
    "allow_crate":   (("level_generation", "enabled_hazards", "crate"),    "coinrun_allow_crate",   bool,  True, "generation may place crates"),
    "allow_monsters":(("level_generation", "enabled_hazards", "monster"),  "coinrun_allow_monsters",bool,  True, "generation may spawn monsters"),
}

VANILLA = {m: spec[3] for m, spec in MECHANICS.items()}

# procgen TOP-LEVEL env options (NOT coinrun_* mechanics): these are passed as
# ProcgenEnv(<key>=...) kwargs, not through coinrun_config. Path in spec -> (kwarg, default).
# Used for global-visual variants (e.g. background on/off) that procgen exposes natively.
ENV_OPTS = {
    "use_backgrounds": (("appearance", "use_backgrounds"), True),
}


class ConfigError(ValueError):
    pass


def _get_path(d, path):
    """Return the value at nested `path` in dict d, or None if any key is missing."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _coerce(mech, value):
    typ = MECHANICS[mech][2]
    if typ is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"mechanic {mech!r} at {MECHANICS[mech][0]} must be bool, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"mechanic {mech!r} at {MECHANICS[mech][0]} must be a number, got {value!r}")


def load_config(path):
    """Parse the CoinRun spec YAML.

    Returns:
        {
          "name": <str>,
          "spec": <full parsed dict>,
          "coinrun_config": {<coinrun_* option>: value, ...},  # ALL mechanics, effective
          "mechanics": {<mechanic>: value, ...},               # ditto, by readable name
          "overrides": {<mechanic>, ...},                      # those differing from vanilla
          "raw": <str, the full YAML text = the code condition>,
        }
    """
    with open(path) as f:
        raw = f.read()
    spec = yaml.safe_load(raw) or {}
    if spec.get("name") != "coinrun" and spec.get("game") != "coinrun":
        # accept either `name: coinrun` (full spec) or `game: coinrun` (older sparse)
        if spec.get("game", "coinrun") != "coinrun":
            raise ConfigError(f"only coinrun supported, got {spec.get('game')!r}")

    coinrun_config, mechanics, overrides = {}, {}, set()
    for mech, (mpath, opt, typ, default, _desc) in MECHANICS.items():
        raw_val = _get_path(spec, mpath)
        if raw_val is None:
            val = default                       # not in spec -> vanilla
        else:
            val = _coerce(mech, raw_val)
            if val != default:
                overrides.add(mech)
        mechanics[mech] = val
        # only forward non-default values as options (omitted => C++ default = vanilla)
        if val != default:
            coinrun_config[opt] = val

    # procgen top-level env options (only forwarded when they differ from default)
    env_opts = {}
    for key, (epath, edefault) in ENV_OPTS.items():
        v = _get_path(spec, epath)
        if v is not None and bool(v) != edefault:
            env_opts[key] = bool(v)
            overrides.add(key)

    name = spec.get("name") or os.path.splitext(os.path.basename(path))[0]
    return {
        "name": name,
        "spec": spec,
        "coinrun_config": coinrun_config,
        "env_opts": env_opts,
        "mechanics": mechanics,
        "overrides": overrides,
        "raw": raw,
    }


def render_code_condition(cfg):
    """The code condition fed to the text encoder = the full spec text, verbatim.

    The whole YAML already describes the complete game (physics, generation,
    collisions, rules) AND carries the effective mechanic values inline, so it needs
    no separate rendering — reading it == reading the game.
    """
    return cfg["raw"]


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        cfg = load_config(p)
        print(f"[{cfg['name']}] overrides={sorted(cfg['overrides'])}")
        print(f"  coinrun_config (drives collection) -> {cfg['coinrun_config']}")
        print(f"  all mechanics (effective)          -> {cfg['mechanics']}")
