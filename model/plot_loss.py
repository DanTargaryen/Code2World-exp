"""Plot the flow-matching loss curve from a training log.

Parses `step N | ... | fm X` (train) and `[eval] fm X` lines, plots train + eval
fm loss vs step (log-y). Eval lines have no step, so they are aligned to the
train step that immediately precedes them (eval runs at fixed intervals).

    python plot_loss.py --log ../logs/pipeline_base.log --out ../loss_curve.png
"""
import re, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(log_path):
    train_steps, train_fm, eval_steps, eval_fm = [], [], [], []
    last_step = 0
    step_re = re.compile(r"^step\s+(\d+)\s+\|.*\bfm\s+([\d.]+)")
    eval_re = re.compile(r"\[eval\]\s+fm\s+([\d.]+)")
    with open(log_path) as f:
        for line in f:
            m = step_re.search(line)
            if m:
                last_step = int(m.group(1))
                train_steps.append(last_step); train_fm.append(float(m.group(2)))
                continue
            m = eval_re.search(line)
            if m:
                eval_steps.append(last_step); eval_fm.append(float(m.group(1)))
    return train_steps, train_fm, eval_steps, eval_fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="../logs/pipeline_base.log")
    ap.add_argument("--out", default="../loss_curve.png")
    ap.add_argument("--title", default="CoinRun base single-scene — flow-matching loss")
    args = ap.parse_args()

    ts, tf, es, ef = parse(args.log)
    if not tf:
        raise SystemExit(f"no `step N | fm X` lines found in {args.log}")

    plt.figure(figsize=(9, 5))
    plt.plot(ts, tf, lw=1.2, color="#3a7", label=f"train fm (final {tf[-1]:.4f})")
    if ef:
        plt.plot(es, ef, lw=1.6, color="#d33", marker="o", ms=3,
                 label=f"eval fm (final {ef[-1]:.4f})")
    plt.yscale("log")
    plt.xlabel("step"); plt.ylabel("fm loss (log)")
    plt.title(args.title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"train: {len(tf)} pts  {tf[0]:.3f} -> {tf[-1]:.4f}")
    if ef:
        print(f"eval : {len(ef)} pts  {ef[0]:.3f} -> {ef[-1]:.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
