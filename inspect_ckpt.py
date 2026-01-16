# inspect_ckpt.py
import torch, pprint, sys
paths = sys.argv[1:]
for p in paths:
    print("==", p)
    ck = torch.load(p, map_location="cpu")
    print("Type:", type(ck))
    if isinstance(ck, dict):
        print("Top-level keys:", list(ck.keys()))
        # Pretty-print a few likely metadata keys if present
        for k in ("epoch","best_loss","best_val_loss","train_stats","metrics","reward_mean"):
            if k in ck:
                print(f"{k}: {ck[k]}")
    else:
        print("Object repr:", repr(ck)[:400])
    print()