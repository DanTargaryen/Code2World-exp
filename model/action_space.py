"""Shared action-space definitions for CoinRun act6.

The dataset stores RAW Procgen action ids. Procgen's unified action space is 15;
CoinRun only maps ids 0..8 (9 valid), and the collector further restricts to the
6 meaningful moves (drops the 3 downward actions 0/3/6). So:

  - full Procgen space          : 15
  - model one-hot (num_actions) : 9   (ids 0..8; 0/3/6 stay all-zero in data)
  - actually-used actions       : 6   (raw ids [1,2,4,5,7,8])

"compact" mode maps those 6 raw ids to dense indices [0..5] so the one-hot has no
dead dimensions. RAW ids stay in the dataset untouched; remapping happens only at
the model boundary (prep_batch / rollout), keeping the dataset format stable.
"""
import torch

# raw Procgen ids kept in CoinRun act6 (matches dataset/collect_one.py ACTION_SET)
#   1=left  2=up-left  4=noop  5=up/jump  7=right  8=up-right
ACTION_SET = [1, 2, 4, 5, 7, 8]
NUM_ACTIONS_FULL = 9        # legacy one-hot width (ids 0..8)
NUM_ACTIONS_COMPACT = len(ACTION_SET)  # 6

RAW2IDX = {a: i for i, a in enumerate(ACTION_SET)}   # e.g. 7 -> 4

_LUT = None


def _lut(device):
    global _LUT
    if _LUT is None:
        lut = torch.full((max(ACTION_SET) + 1,), -1, dtype=torch.long)
        for a, i in RAW2IDX.items():
            lut[a] = i
        _LUT = lut
    return _LUT.to(device)


def remap_to_compact(actions):
    """actions: LongTensor of RAW Procgen ids -> dense indices [0..5].
    Raises if any id is outside ACTION_SET (would map to -1)."""
    if not torch.is_tensor(actions):
        actions = torch.as_tensor(actions, dtype=torch.long)
    out = _lut(actions.device)[actions.long()]
    if int(out.min()) < 0:
        bad = sorted(set(int(a) for a in actions.reshape(-1).tolist()) - set(ACTION_SET))
        raise ValueError(f"raw action id(s) {bad} not in ACTION_SET {ACTION_SET}")
    return out
