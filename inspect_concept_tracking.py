# inspect_concept_tracking.py
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to checkpoint .pt (e.g., pcd_layer13.pt)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # ---- basic scalar-ish stuff ----
    print("Keys:", sorted(list(ckpt.keys())))
    for k in ["epoch", "step", "best_val", "curr_bad", "seen_tokens"]:
        if k in ckpt:
            print(f"{k}: {ckpt[k]}")

    seen_tokens = int(ckpt.get("seen_tokens", 0))

    # ---- inactive_concepts_tracker: list[(seen_tokens, avg_inactive)] ----
    tracker = ckpt.get("inactive_concepts_tracker", [])
    print(f"\ninactive_concepts_tracker: {len(tracker)} points")
    if len(tracker) > 0:
        x = np.array([t[0] for t in tracker], dtype=np.int64)
        y = np.array([t[1] for t in tracker], dtype=np.float64)

        print("  last point:", tracker[-1])
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("seen_tokens")
        plt.ylabel("avg inactive concepts (over last window)")
        plt.title("Inactive concepts tracker")
        plt.tight_layout()
        plt.savefig(f"{args.ckpt}_1.png")

    # ---- concepts_last_occ_by_seen_tokens: per-concept last seen token index ----
    last_occ = ckpt.get("concepts_last_occ_by_seen_tokens", None)
    if last_occ is None:
        print("\nNo concepts_last_occ_by_seen_tokens in checkpoint.")
        return

    last_occ = torch.as_tensor(last_occ).cpu().long()
    total = last_occ.numel()
    never = (last_occ < 0)
    n_never = int(never.sum().item())
    print(f"\nconcepts_last_occ_by_seen_tokens: shape={tuple(last_occ.shape)} total={total}")
    print(f"  never seen: {n_never}/{total} = {n_never/total:.3%}")

    if seen_tokens > 0 and n_never < total:
        ages = (seen_tokens - last_occ[~never]).clamp_min(0).numpy()
        print(f"  age stats (tokens since last seen): "
              f"min={ages.min()}, p50={np.percentile(ages,50):.0f}, p90={np.percentile(ages,90):.0f}, max={ages.max()}")

        plt.figure()
        plt.hist(ages, bins=50)
        plt.xlabel("tokens since last seen (age)")
        plt.ylabel("count")
        plt.title("Concept recency histogram")
        plt.tight_layout()
        plt.savefig(f"{args.ckpt}_2.png")
    else:
        print("  (Either seen_tokens==0, or all concepts are never-seen so ages are not meaningful.)")

if __name__ == "__main__":
    main()
