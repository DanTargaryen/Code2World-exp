"""Load a CoinRun game config (YAML) into procgen options.

The YAML is a *readable, declarative* description of CoinRun's tunable mechanics.
It doubles as the **code condition** for the world model: instead of feeding raw
`coinrun.cpp` source (523 lines, mechanics tangled with rendering/boilerplate),
we feed this config, which exposes exactly the mechanics that were changed.

How it works end-to-end:
  YAML  --load_config-->  coinrun_config dict  --ProcgenEnv(coinrun_config=...)-->
  gym3 encodes each value as float32 option  -->  C++ game.cpp parse_options
  -->  options.coinrun_*  read in coinrun.cpp at runtime.

No recompile is needed to change these values (unlike the old variants.py, which
rewrote coinrun.cpp and rebuilt procgen per variant). Compile procgen once with
the options plumbing, then any config is a pure runtime parameter change.

Only mechanics listed in MECHANIC_TO_OPTION are settable; anything omitted falls
back to vanilla CoinRun (the procgen C++ defaults, which equal the values below).
"""
import os
import yaml

# readable mechanic name (in YAML) -> procgen float option name (consumed in C++)
MECHANIC_TO_OPTION = {
    "gravity": "coinrun_gravity",
    "max_jump": "coinrun_max_jump",
    "max_speed": "coinrun_max_speed",
    "air_control": "coinrun_air_control",
    "mixrate": "coinrun_mixrate",
    "goal_reward": "coinrun_goal_reward",
}

# vanilla CoinRun values (must mirror the defaults in procgen GameOptions / the
# original hardcoded constants in coinrun.cpp). Used for validation and docs.
VANILLA = {
    "gravity": 0.2,
    "max_jump": 1.5,
    "max_speed": 0.5,
    "air_control": 0.15,
    "mixrate": 0.2,
    "goal_reward": 10.0,
}


class ConfigError(ValueError):
    pass


def load_config(path):
    """Parse a CoinRun YAML config into a normalized dict.

    Returns:
        {
          "name": <str, config name for bookkeeping>,
          "game": "coinrun",
          "coinrun_config": {<coinrun_* option>: float, ...},  # only overrides
          "mechanics": {<mechanic>: float, ...},               # merged w/ vanilla
          "raw": <str, the exact YAML text = the code condition>,
        }
    """
    with open(path) as f:
        raw = f.read()
    spec = yaml.safe_load(raw) or {}

    game = spec.get("game", "coinrun")
    if game != "coinrun":
        raise ConfigError(f"only 'coinrun' is supported for now, got game={game!r}")

    mechanics_in = spec.get("mechanics", {}) or {}
    unknown = set(mechanics_in) - set(MECHANIC_TO_OPTION)
    if unknown:
        raise ConfigError(
            f"unknown mechanic(s) {sorted(unknown)}; "
            f"allowed: {sorted(MECHANIC_TO_OPTION)}"
        )

    coinrun_config = {}
    merged = dict(VANILLA)
    for mech, value in mechanics_in.items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"mechanic {mech!r} must be a number, got {value!r}")
        coinrun_config[MECHANIC_TO_OPTION[mech]] = value
        merged[mech] = value

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
