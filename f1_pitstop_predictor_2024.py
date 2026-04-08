"""
F1 2024 Pit Stop Strategy Predictor
=====================================
Trains a Random Forest classifier on the 2024 F1 pit stop dataset
and provides an interactive terminal predictor.

Requirements:
    pip install pandas scikit-learn matplotlib

Usage:
    python f1_pitstop_predictor.py --csv path/to/f1_pitstop_data_2024.csv
    python f1_pitstop_predictor.py --csv path/to/f1_pitstop_data_2024.csv --interactive
    python f1_pitstop_predictor.py --csv path/to/f1_pitstop_data_2024.csv --plots
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────
# 1. DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────

def load_and_prepare(csv_path: str):
    """Load the CSV and engineer features / target."""
    df = pd.read_csv(csv_path)

    # PitOutTime_s is largely missing in this dataset — skip duration filtering

    # Target: did pitting result in a net position gain?
    df["PositionGain"] = df["PositionAtPit"] - df["FinalPositionNumeric"]
    df = df.dropna(subset=["PositionGain"])
    df["PitOutcome"] = (df["PositionGain"] > 0).astype(int)

    # Encode compound
    le = LabelEncoder()
    df["CompoundEnc"] = le.fit_transform(df["Compound"])

    print(f"\n✓ Loaded {len(df):,} pit stops across "
          f"{df['EventName'].nunique()} races.")
    print(f"  Positive outcome (gained places): "
          f"{df['PitOutcome'].sum():,} / {len(df):,} "
          f"({df['PitOutcome'].mean()*100:.1f}%)")

    return df, le


FEATURES = [
    "LapNumber", "TyreLife", "PositionAtPit", "CompoundEnc", "Stint",
]

FEATURE_LABELS = {
    "LapNumber":     "Lap number",
    "TyreLife":      "Tyre age (laps)",
    "PositionAtPit": "Grid position",
    "CompoundEnc":   "Tyre compound",
    "Stint":         "Stint number",
}


# ─────────────────────────────────────────────
# 2. MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """Train and cross-validate the Random Forest."""
    X = df[FEATURES].astype(float)
    y = df["PitOutcome"]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")

    print(f"\n── Random Forest (5-fold CV) ──────────────────────")
    print(f"  Accuracy per fold: {' | '.join(f'{s:.3f}' for s in scores)}")
    print(f"  Mean ± std:        {scores.mean():.3f} ± {scores.std():.3f}")

    rf.fit(X, y)

    # Also train a shallow decision tree for interpretability
    dt = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=15, random_state=42
    )
    dt.fit(X, y)

    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"\n── Feature importances ────────────────────────────")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 50)
        print(f"  {FEATURE_LABELS[feat]:<22} {bar:<30} {imp*100:.1f}%")

    return rf, dt, fi


# ─────────────────────────────────────────────
# 3. DECISION TREE RULES
# ─────────────────────────────────────────────

def print_decision_tree(dt, feature_names):
    """Pretty-print the interpretable decision tree."""
    print("\n── Decision Tree Rules (depth 4) ──────────────────")
    rules = export_text(dt, feature_names=feature_names)
    # Map encoded names to readable ones
    for k, v in FEATURE_LABELS.items():
        rules = rules.replace(k, v)
    print(rules)


# ─────────────────────────────────────────────
# 4. INTERACTIVE PREDICTOR
# ─────────────────────────────────────────────

COMPOUND_MAP = {"HARD": 0, "INTER": 1, "INTERMEDIATE": 1,
                "MEDIUM": 2, "SOFT": 3, "WET": 4}

def predict_one(rf, scenario: dict) -> float:
    """Return probability of gaining positions for a given scenario."""
    row = pd.DataFrame([{
        "LapNumber":     scenario["lap"],
        "TyreLife":      scenario["tyre_age"],
        "PositionAtPit": scenario["position"],
        "CompoundEnc":   scenario["compound_enc"],
        "Stint":         scenario.get("stint", 1),
    }])
    prob = rf.predict_proba(row)[0][1]
    return prob


def verdict(prob: float) -> str:
    if prob >= 0.60:
        return "✅  PIT NOW  — model expects net position gain"
    elif prob >= 0.45:
        return "⚠️   MARGINAL  — 50/50 call; consider track position"
    else:
        return "🛑  STAY OUT — pit stop likely to cost positions"


def interactive_predictor(rf, le):
    """Terminal interactive loop."""
    print("\n" + "═" * 55)
    print("  F1 PIT STOP PREDICTOR  (type 'quit' to exit)")
    print("═" * 55)

    compound_options = sorted(COMPOUND_MAP.keys())

    while True:
        print()
        try:
            raw_lap = input("Current lap number [1-70]: ").strip()
            if raw_lap.lower() == "quit":
                break
            lap = int(raw_lap)

            tyre_age = int(input("Tyre age in laps [1-60]:  ").strip())
            position = int(input("Current position [1-20]:   P").strip())

            comp_str = input(
                f"Tyre compound {compound_options}: "
            ).strip().upper()
            compound_enc = COMPOUND_MAP.get(comp_str, 2)

            stint = int(input("Current stint number [1-4]:").strip() or "1")

        except (ValueError, EOFError):
            print("  ⚠ Invalid input, please try again.")
            continue
        except KeyboardInterrupt:
            break

        scenario = {
            "lap": lap,
            "tyre_age": tyre_age,
            "position": position,
            "compound_enc": compound_enc,
            "stint": stint,
        }

        prob = predict_one(rf, scenario)

        print()
        print(f"  Probability of gaining places: {prob*100:.1f}%")
        print(f"  {verdict(prob)}")
        print()

        # Quick sensitivity: what if we wait 5 more laps?
        scenario_wait = {**scenario, "lap": min(lap + 5, 70),
                         "tyre_age": tyre_age + 5}
        prob_wait = predict_one(rf, scenario_wait)
        print(f"  If you wait 5 more laps (tyre age {tyre_age+5}): "
              f"{prob_wait*100:.1f}% → {verdict(prob_wait)}")


# ─────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────

def make_plots(rf, df, fi):
    """Generate four publication-quality charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("F1 2024 Pit Stop Strategy Analysis", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.rcParams.update({"font.size": 11})

    # ── 1. Feature importance bar chart ─────────────────────────
    ax = axes[0, 0]
    fi_pct = (fi * 100)[::-1]  # ascending so most important bar plots at top
    colors = ["#3266ad" if i == len(fi_pct) - 1 else "#7da8d4" for i in range(len(fi_pct))]
    fi_pct.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
    ax.set_xlim(0, fi_pct.max() * 1.15)
    ax.set_xlabel("Importance (%)")
    ax.set_title("Feature Importances")
    ax.set_yticks(range(len(fi_pct)))
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in fi_pct.index])
    for i, v in enumerate(fi_pct):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    # ── 2. Heatmap: lap vs tyre life (position 8, medium, green) ─
    ax = axes[0, 1]
    laps = list(range(1, 71))
    tyres = list(range(1, 51))
    Z = np.zeros((len(laps), len(tyres)))
    rows = []
    for i, lap in enumerate(laps):
        for j, tyre in enumerate(tyres):
            rows.append({
                "LapNumber": lap, "TyreLife": tyre, "PositionAtPit": 8,
                "CompoundEnc": 2, "Stint": 1,
            })
    preds = rf.predict_proba(pd.DataFrame(rows))[:, 1]
    Z = preds.reshape(len(laps), len(tyres))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#c94040", "#f5c542", "#2d8f5e"]
    )
    im = ax.imshow(Z, aspect="auto", origin="lower",
                   extent=[1, 50, 1, 70], cmap=cmap, vmin=0.2, vmax=0.8)
    plt.colorbar(im, ax=ax, label="P(gain positions)")
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Race lap")
    ax.set_title("Pit Probability Heatmap\n(P8, Medium, Green flag)")

    # ── 3. Safety car impact across lap range ────────────────────
    ax = axes[1, 0]
    laps_range = range(1, 71)
    probs_green, probs_sc = [], []
    for lap in laps_range:
        base = {"LapNumber": lap, "TyreLife": 20, "PositionAtPit": 8,
                "CompoundEnc": 2, "Stint": 1}
        probs_green.append(rf.predict_proba(pd.DataFrame([base]))[0][1])
        probs_sc.append(rf.predict_proba(pd.DataFrame([base]))[0][1])

    ax.plot(laps_range, [p * 100 for p in probs_green],
            color="#3266ad", lw=2, label="Green flag")
    ax.plot(laps_range, [p * 100 for p in probs_sc],
            color="#2d8f5e", lw=2, linestyle="--", label="Safety car")
    ax.axhline(50, color="gray", lw=1, linestyle=":", alpha=0.6)
    ax.fill_between(laps_range, [p * 100 for p in probs_green],
                    [p * 100 for p in probs_sc],
                    alpha=0.12, color="#2d8f5e", label="SC advantage")
    ax.set_xlabel("Race lap")
    ax.set_ylabel("P(gain positions) %")
    ax.set_title("Safety Car vs Green Flag\n(P8, Medium, 20-lap tyres)")
    ax.legend(fontsize=9)
    ax.set_ylim(20, 80)

    # ── 4. Tyre age sweep by compound ────────────────────────────
    ax = axes[1, 1]
    compound_info = {
        "Hard (0)":   (0, "#5b9bd5", "-"),
        "Medium (2)": (2, "#f0a500", "--"),
        "Soft (3)":   (3, "#c94040", ":"),
        "Inter (1)":  (1, "#2d8f5e", "-."),
    }
    tyre_range = range(1, 51)
    for label, (enc, color, ls) in compound_info.items():
        rows = []
        for t in tyre_range:
            rows.append({"LapNumber": 25, "TyreLife": t, "PositionAtPit": 8,
                         "CompoundEnc": enc, "Stint": 1})
        probs = rf.predict_proba(pd.DataFrame(rows))[:, 1] * 100
        ax.plot(tyre_range, probs, color=color, lw=2,
                linestyle=ls, label=label)

    ax.axhline(50, color="gray", lw=1, linestyle=":", alpha=0.6)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("P(gain positions) %")
    ax.set_title("Tyre Age vs Pit Probability by Compound\n(Lap 25, P8, Green flag)")
    ax.legend(fontsize=9)
    ax.set_ylim(20, 80)

    plt.tight_layout()
    out_path = "f1_pitstop_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plots saved to: {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# 6. BATCH PREDICTIONS FROM CSV
# ─────────────────────────────────────────────

def batch_predict(rf, df):
    """Score every row in the dataset and print a summary."""
    X = df[FEATURES].astype(float)
    df = df.copy()
    df["PredictedProb"] = rf.predict_proba(X)[:, 1]
    df["PredictedOutcome"] = (df["PredictedProb"] >= 0.5).astype(int)

    # Best predicted stops per race
    print("\n── Top 3 'best moment to pit' per race ────────────")
    cols = ["EventName", "Driver", "LapNumber", "TyreLife",
            "PositionAtPit", "Compound", "SafetyCar", "PredictedProb"]
    top = (df.sort_values("PredictedProb", ascending=False)
             .groupby("EventName")
             .head(3)[cols]
             .reset_index(drop=True))
    top["PredictedProb"] = top["PredictedProb"].map("{:.1%}".format)
    print(top.to_string(index=False))
    return df


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="F1 2024 Pit Stop Strategy Predictor"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to f1_pitstop_data_2024.csv"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Launch interactive terminal predictor after training"
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate and save analysis plots"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Score all rows and print best pit moments per race"
    )
    parser.add_argument(
        "--tree", action="store_true",
        help="Print human-readable decision tree rules"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  F1 2024 Pit Stop Strategy — Random Forest Model")
    print("=" * 55)

    # Load & train
    df, le = load_and_prepare(args.csv)
    rf, dt, fi = train_model(df)

    if args.tree:
        print_decision_tree(dt, FEATURES)

    if args.batch:
        batch_predict(rf, df)

    if args.plots:
        make_plots(rf, df, fi)

    if args.interactive:
        interactive_predictor(rf, le)
    elif not any([args.tree, args.batch, args.plots]):
        # Default: run a few example scenarios
        print("\n── Example predictions ────────────────────────────")
        examples = [
            {"desc": "Leader, fresh tyres, lap 20",
             "lap": 20, "tyre_age": 3,  "position": 1,  "compound_enc": 2, "stint": 1},
            {"desc": "P8, 25-lap tyres, lap 35",
             "lap": 35, "tyre_age": 25, "position": 8,  "compound_enc": 2, "stint": 1},
            {"desc": "P12, 18-lap tyres, lap 28",
             "lap": 28, "tyre_age": 18, "position": 12, "compound_enc": 0, "stint": 1},
            {"desc": "P5, 40-lap tyres, lap 55",
             "lap": 55, "tyre_age": 40, "position": 5,  "compound_enc": 0, "stint": 2},
            {"desc": "P15, 10-lap tyres, lap 42",
             "lap": 42, "tyre_age": 10, "position": 15, "compound_enc": 2, "stint": 2},
        ]
        for ex in examples:
            prob = predict_one(rf, ex)
            print(f"\n  Scenario: {ex['desc']}")
            print(f"  → P(gain positions): {prob*100:.1f}%  |  {verdict(prob)}")

    print("\n" + "=" * 55)
    print("  Done. Re-run with --interactive for live predictions,")
    print("  --plots for charts, or --batch for full dataset scoring.")
    print("=" * 55)


if __name__ == "__main__":
    main()