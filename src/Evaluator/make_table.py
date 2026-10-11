"""
Collects every *_results.json below a results folder into one table.

    python make_table.py ./results            -> prints the table and writes results_table.csv
"""
import sys, os, json, glob
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "./results"
rows = []
for path in sorted(glob.glob(os.path.join(root, "**", "*_results.json"), recursive=True)):
    r = json.load(open(path)); s, a = r["summary"], r["settings"]
    f = s["final_round"]; b = s.get("security_buffer", {})
    rows.append({
        "folder": os.path.relpath(os.path.dirname(path), root),
        "buffer": a["buffer"],
        "attack": a["attack"] if a["attack"] == "none" else f'{a["attack"]} x{a["attack_strength"]:g}',
        "runs": a["runs"],
        "final AUC": f'{f["test_auc"]["mean"]:.4f} ± {f["test_auc"]["std"]:.4f}',
        "final F1": f'{f["test_f1"]["mean"]:.4f}',
        "mean client AUC": f'{f["mean_client_auc"]["mean"]:.4f}',
        "final val MSE": f'{f["global_val_mse"]["mean"]:.4g}',
        "worst val MSE": f'{s["worst_val_mse_any_round"]:.4g}',
        "malicious caught": f'{b["malicious_quarantined"]}/{b["malicious_updates"]}' if b else "-",
        "malicious dropped": b.get("malicious_dropped", "-"),
        "honest quarantined": f'{b["honest_quarantined"]}/{b["honest_updates"]}' if b else "-",
        "honest dropped": b.get("honest_dropped", "-"),
    })
table = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
print(table.to_string(index=False))
table.to_csv(os.path.join(root, "results_table.csv"), index=False)
