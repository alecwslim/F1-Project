"""
F1 Pit Stop Advantage Predictor
===============================
Train a Random Forest to predict whether a pit stop will result in a net
position gain by the end of the race.

Target:
    PositionGain = PositionAtPit - FinalPositionNumeric
    PitAdvantage = 1 if PositionGain > 0 else 0

Notes:
    - Uses raw backup data because it still contains final-race outcome fields.
    - Trains only on actual pit-in laps.
    - Uses the previous lap position as a proxy for "PositionAtPit" when
      available, which is usually closer to the driver's running position
      before entering the pits.
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text


TARGET_COL = "PitAdvantage"
RAW_CSV = "f1_lap_data_raw_backup.csv"
PREDICTION_THRESHOLD = 0.35
GRAPHS_DIR = Path("graphs")
MODEL_PATH = Path("f1_pitstop_advantage_predictor_model.joblib")

KEY_COLS = ["Year", "RoundNumber", "EventName", "Driver", "LapNumber"]
GROUP_COLS = ["Year", "RoundNumber", "EventName", "Driver"]

FEATURES = [
    "LapNumber",
    "LapsRemaining",
    "Stint",
    "TyreLife",
    "TyreLifeRatio",
    "LapTime_s",
    "PitCountSoFar",
    "PositionAtPit",
    "HasPittedBefore",
    "IsFirstStint",
    "AirTemp_C",
    "TrackTemp_C",
    "TrackTemp_AirTemp_Delta",
    "Humidity_pct",
    "WindSpeed_ms",
    "Rainfall",
    "YellowFlag",
    "SafetyCar",
    "VirtualSafetyCar",
    "RecentLapTimeMean_3",
    "RecentLapTimeStd_3",
    "LapTimeDeltaToRecent_3",
    "PositionChangeFromPrevLap",
    "RecentPositionMean_3",
    "PositionVsRecentMean_3",
]

FEATURE_LABELS = {
    "LapNumber": "Lap number",
    "LapsRemaining": "Laps remaining",
    "Stint": "Stint number",
    "TyreLife": "Tyre age",
    "TyreLifeRatio": "Tyre age / lap",
    "LapTime_s": "Lap time (s)",
    "PitCountSoFar": "Pit count so far",
    "PositionAtPit": "Position at pit",
    "HasPittedBefore": "Has pitted before",
    "IsFirstStint": "First stint",
    "AirTemp_C": "Air temp (C)",
    "TrackTemp_C": "Track temp (C)",
    "TrackTemp_AirTemp_Delta": "Track-air delta",
    "Humidity_pct": "Humidity %",
    "WindSpeed_ms": "Wind speed",
    "Rainfall": "Rainfall",
    "YellowFlag": "Yellow flag",
    "SafetyCar": "Safety car",
    "VirtualSafetyCar": "VSC",
    "RecentLapTimeMean_3": "Recent lap mean",
    "RecentLapTimeStd_3": "Recent lap std",
    "LapTimeDeltaToRecent_3": "Lap time delta to recent",
    "PositionChangeFromPrevLap": "Position change",
    "RecentPositionMean_3": "Recent position mean",
    "PositionVsRecentMean_3": "Position vs recent",
}


def normalize_boolean(series: pd.Series) -> pd.Series:
    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        True: 1,
        False: 0,
    }
    return series.astype("string").str.strip().str.lower().map(mapping)


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.sort_values(KEY_COLS).reset_index(drop=True)

    for col in ["Rainfall", "GreenFlag", "YellowFlag", "SafetyCar", "RedFlag", "VirtualSafetyCar"]:
        if col in df.columns:
            df[col] = normalize_boolean(df[col]).fillna(0).astype("int8")

    numeric_cols = [
        "Year", "RoundNumber", "LapNumber", "Stint", "TyreLife", "LapTime_s",
        "PitInTime_s", "PitOutTime_s", "PitCountSoFar", "PositionAtLapEnd",
        "FinalPositionNumeric", "AirTemp_C", "TrackTemp_C", "Humidity_pct",
        "Pressure_mbar", "WindSpeed_ms", "WindDirection",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    driver_group = df.groupby(GROUP_COLS, dropna=False)
    df["PrevLapPosition"] = driver_group["PositionAtLapEnd"].shift(1)
    df["PositionAtPit"] = df["PrevLapPosition"].fillna(df["PositionAtLapEnd"])

    race_group_cols = ["Year", "RoundNumber", "EventName"]
    max_laps = df.groupby(race_group_cols, dropna=False)["LapNumber"].transform("max")
    df["LapsRemaining"] = (max_laps - df["LapNumber"]).clip(lower=0)
    df["TyreLifeRatio"] = (df["TyreLife"] / df["LapNumber"].replace(0, pd.NA)).fillna(0)
    df["HasPittedBefore"] = (df["PitCountSoFar"] > 0).astype("int8")
    df["IsFirstStint"] = (df["Stint"] <= 1).astype("int8")
    df["TrackTemp_AirTemp_Delta"] = df["TrackTemp_C"] - df["AirTemp_C"]
    df["RecentLapTimeMean_3"] = driver_group["LapTime_s"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["RecentLapTimeStd_3"] = driver_group["LapTime_s"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    df["LapTimeDeltaToRecent_3"] = df["LapTime_s"] - df["RecentLapTimeMean_3"]
    df["PositionChangeFromPrevLap"] = df["PrevLapPosition"] - df["PositionAtLapEnd"]
    df["RecentPositionMean_3"] = driver_group["PositionAtLapEnd"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["PositionVsRecentMean_3"] = df["RecentPositionMean_3"] - df["PositionAtLapEnd"]

    # Use actual pit-in laps only for the advantage target.
    pit_df = df[df["PitInTime_s"].notna()].copy()
    pit_df = pit_df.dropna(subset=["FinalPositionNumeric", "PositionAtPit"]).copy()
    pit_df["PositionGain"] = pit_df["PositionAtPit"] - pit_df["FinalPositionNumeric"]
    pit_df[TARGET_COL] = (pit_df["PositionGain"] > 0).astype(int)
    pit_df["RaceGroup"] = pit_df["Year"].astype(str) + "-R" + pit_df["RoundNumber"].astype(int).astype(str)

    for col in FEATURES:
        pit_df[col] = pd.to_numeric(pit_df[col], errors="coerce")
        pit_df[col] = pit_df[col].fillna(pit_df[col].median())

    print(f"\n✓ Loaded {len(pit_df):,} pit-in laps across {pit_df['RaceGroup'].nunique()} races.")
    print(
        f"  Positive outcome ({TARGET_COL}=1): "
        f"{pit_df[TARGET_COL].sum():,} / {len(pit_df):,} "
        f"({pit_df[TARGET_COL].mean() * 100:.1f}%)"
    )
    return pit_df


def train_model(df: pd.DataFrame):
    X = df[FEATURES].astype(float)
    y = df[TARGET_COL]
    groups = df["RaceGroup"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        min_samples_split=8,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = GroupKFold(n_splits=5)
    fold_scores = []
    for fold_train, fold_valid in cv.split(X_train, y_train, groups_train):
        fold_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=4,
            min_samples_split=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        fold_model.fit(X_train.iloc[fold_train], y_train.iloc[fold_train])
        fold_pred = fold_model.predict(X_train.iloc[fold_valid])
        fold_scores.append(f1_score(y_train.iloc[fold_valid], fold_pred, zero_division=0))

    print("\n── Random Forest (5-fold GroupKFold by race) ─────")
    print(f"  F1 per fold:       {' | '.join(f'{s:.3f}' for s in fold_scores)}")
    print(f"  Mean ± std:        {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

    history_steps = [25, 50, 75, 100, 150, 200, 250, 300]
    history_rows = []
    history_model = RandomForestClassifier(
        n_estimators=0,
        max_depth=12,
        min_samples_leaf=4,
        min_samples_split=8,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        warm_start=True,
    )
    for n_trees in history_steps:
        history_model.set_params(n_estimators=n_trees)
        history_model.fit(X_train, y_train)
        train_pred = (history_model.predict_proba(X_train)[:, 1] >= PREDICTION_THRESHOLD).astype(int)
        valid_pred = (history_model.predict_proba(X_test)[:, 1] >= PREDICTION_THRESHOLD).astype(int)
        history_rows.append(
            {
                "n_trees": n_trees,
                "train_accuracy": accuracy_score(y_train, train_pred),
                "valid_accuracy": accuracy_score(y_test, valid_pred),
                "train_f1": f1_score(y_train, train_pred, zero_division=0),
                "valid_f1": f1_score(y_test, valid_pred, zero_division=0),
            }
        )

    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)

    print("\n── Held-out grouped test split ───────────────────")
    print(f"  Accuracy:          {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision:         {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  Recall:            {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  F1:                {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  ROC AUC:           {roc_auc_score(y_test, y_prob):.3f}")
    print(f"  Threshold:         {PREDICTION_THRESHOLD:.2f}")

    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42)
    dt.fit(X_train, y_train)

    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n── Feature importances ────────────────────────────")
    for feat, imp in fi.head(12).items():
        bar = "█" * int(imp * 60)
        print(f"  {FEATURE_LABELS.get(feat, feat):<24} {bar:<35} {imp * 100:.1f}%")

    defaults = df[FEATURES].median().to_dict()
    history = pd.DataFrame(history_rows)
    return rf, dt, fi, defaults, history


def print_decision_tree(dt, feature_names):
    print("\n── Decision Tree Rules (depth 4) ──────────────────")
    rules = export_text(dt, feature_names=feature_names)
    for key, value in FEATURE_LABELS.items():
        rules = rules.replace(key, value)
    print(rules)


def build_row_from_scenario(scenario: dict, defaults: dict) -> pd.DataFrame:
    row = defaults.copy()
    row.update(scenario)
    return pd.DataFrame([row], columns=FEATURES).astype(float)


def predict_one(rf, scenario: dict, defaults: dict) -> float:
    row = build_row_from_scenario(scenario, defaults)
    return rf.predict_proba(row)[0][1]


def verdict(prob: float) -> str:
    if prob >= 0.60:
        return "LIKELY ADVANTAGEOUS PIT"
    if prob >= PREDICTION_THRESHOLD:
        return "BORDERLINE PIT ADVANTAGE"
    return "LIKELY NO NET ADVANTAGE"


def make_plots(rf, df, fi, defaults: dict, history: pd.DataFrame):
    GRAPHS_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11})

    fig, ax = plt.subplots(figsize=(7, 5))
    fi_pct = (fi.head(10) * 100)[::-1]
    colors = ["#3266ad" if i == len(fi_pct) - 1 else "#7da8d4" for i in range(len(fi_pct))]
    fi_pct.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
    ax.set_xlabel("Importance (%)")
    ax.set_title("Top Feature Importances")
    ax.set_yticks(range(len(fi_pct)))
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in fi_pct.index])
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "f1_pitstop_advantage_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    laps = list(range(1, 71))
    positions = list(range(1, 21))
    rows = []
    for lap in laps:
        for pos in positions:
            rows.append(
                build_row_from_scenario(
                    {
                        "LapNumber": lap,
                        "LapsRemaining": max(0, 70 - lap),
                        "PositionAtPit": pos,
                    },
                    defaults,
                ).iloc[0].to_dict()
            )
    preds = rf.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]
    z = preds.reshape(len(laps), len(positions))
    cmap = mcolors.LinearSegmentedColormap.from_list("adv", ["#c94040", "#f5c542", "#2d8f5e"])
    im = ax.imshow(z, aspect="auto", origin="lower", extent=[1, 20, 1, 70], cmap=cmap, vmin=0.1, vmax=0.9)
    plt.colorbar(im, ax=ax, label="P(net position gain)")
    ax.set_xlabel("Position at pit")
    ax.set_ylabel("Current lap")
    ax.set_title("Pit Advantage Heatmap")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "f1_pitstop_advantage_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    tyre_range = list(range(0, 41))
    rows = []
    for tyre in tyre_range:
        rows.append(
            build_row_from_scenario(
                {
                    "TyreLife": tyre,
                    "TyreLifeRatio": tyre / 30 if 30 else 0,
                    "LapNumber": 30,
                    "LapsRemaining": 40,
                    "PositionAtPit": 10,
                },
                defaults,
            ).iloc[0].to_dict()
        )
    probs = rf.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1] * 100
    ax.plot(tyre_range, probs, color="#c94040", lw=2)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Advantage probability %")
    ax.set_title("Advantage vs Tyre Age")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "f1_pitstop_advantage_tyre_age.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    base_rows = []
    repeat_rows = []
    for pos in positions:
        base_rows.append(build_row_from_scenario({"PositionAtPit": pos, "PitCountSoFar": 0, "HasPittedBefore": 0}, defaults).iloc[0].to_dict())
        repeat_rows.append(build_row_from_scenario({"PositionAtPit": pos, "PitCountSoFar": 1, "HasPittedBefore": 1}, defaults).iloc[0].to_dict())
    prob_base = rf.predict_proba(pd.DataFrame(base_rows)[FEATURES])[:, 1] * 100
    prob_repeat = rf.predict_proba(pd.DataFrame(repeat_rows)[FEATURES])[:, 1] * 100
    ax.plot(positions, prob_base, color="#3266ad", lw=2, label="First stop")
    ax.plot(positions, prob_repeat, color="#2d8f5e", lw=2, linestyle="--", label="Already pitted")
    ax.set_xlabel("Position at pit")
    ax.set_ylabel("Advantage probability %")
    ax.set_title("Effect of Prior Pit History")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "f1_pitstop_advantage_prior_pit_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["n_trees"], history["train_accuracy"], marker="o", color="#3266ad", label="Training")
    ax.plot(history["n_trees"], history["valid_accuracy"], marker="o", color="#c94040", label="Validation")
    ax.set_title("Accuracy Over Training")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "f1_pitstop_advantage_accuracy_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["n_trees"], history["train_f1"], marker="o", color="#3266ad", label="Training")
    ax.plot(history["n_trees"], history["valid_f1"], marker="o", color="#2d8f5e", label="Validation")
    ax.set_title("F1 Over Training")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("F1 score")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "f1_pitstop_advantage_f1_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n✓ Plots saved to individual files:")
    for path in [
        GRAPHS_DIR / "f1_pitstop_advantage_feature_importance.png",
        GRAPHS_DIR / "f1_pitstop_advantage_heatmap.png",
        GRAPHS_DIR / "f1_pitstop_advantage_tyre_age.png",
        GRAPHS_DIR / "f1_pitstop_advantage_prior_pit_history.png",
        GRAPHS_DIR / "f1_pitstop_advantage_accuracy_curve.png",
        GRAPHS_DIR / "f1_pitstop_advantage_f1_curve.png",
    ]:
        print(f"  {path}")


def interactive_predictor(rf, defaults: dict):
    print("\n" + "=" * 60)
    print("  F1 PIT STOP ADVANTAGE PREDICTOR  (type 'quit' to exit)")
    print("=" * 60)
    while True:
        print()
        try:
            raw_lap = input("Current lap number [1-70]: ").strip()
            if raw_lap.lower() == "quit":
                break
            lap = int(raw_lap)
            tyre_age = int(input("Tyre age in laps [0-60]:  ").strip())
            position = int(input("Position before pit [1-20]: P").strip())
            stint = int(input("Current stint number [1-4]: ").strip() or "1")
            pits_so_far = int(input("Pit stops so far [0-3]:    ").strip() or "0")
        except (ValueError, EOFError):
            print("  Invalid input, please try again.")
            continue
        except KeyboardInterrupt:
            break

        scenario = {
            "LapNumber": lap,
            "LapsRemaining": max(0, 70 - lap),
            "Stint": stint,
            "TyreLife": tyre_age,
            "TyreLifeRatio": tyre_age / lap if lap else 0,
            "PositionAtPit": position,
            "PitCountSoFar": pits_so_far,
            "HasPittedBefore": int(pits_so_far > 0),
            "IsFirstStint": int(stint == 1),
            "PositionChangeFromPrevLap": 0,
        }

        prob = predict_one(rf, scenario, defaults)
        print()
        print(f"  Probability this pit gains places by finish: {prob * 100:.1f}%")
        print(f"  {verdict(prob)}")


def main():
    parser = argparse.ArgumentParser(description="F1 Pit Stop Advantage Predictor")
    parser.add_argument("--csv", default=RAW_CSV, help="Path to raw backup CSV with final position fields")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive predictor after training")
    parser.add_argument("--plots", action="store_true", help="Generate and save analysis plots")
    parser.add_argument("--tree", action="store_true", help="Print human-readable decision tree rules")
    args = parser.parse_args()

    print("=" * 60)
    print("  F1 Pit Stop Advantage Predictor")
    print("=" * 60)

    df = load_and_prepare(args.csv)
    rf, dt, fi, defaults, history = train_model(df)
    joblib.dump(
        {
            "model": rf,
            "features": FEATURES,
            "threshold": PREDICTION_THRESHOLD,
            "defaults": defaults,
            "target": TARGET_COL,
        },
        MODEL_PATH,
    )
    print(f"\n✓ Model saved to: {MODEL_PATH}")

    if args.tree:
        print_decision_tree(dt, FEATURES)
    if args.plots:
        make_plots(rf, df, fi, defaults, history)
    if args.interactive:
        interactive_predictor(rf, defaults)
    else:
        print("\n── Example predictions ────────────────────────────")
        examples = [
            {"desc": "P7 stopping lap 18", "LapNumber": 18, "LapsRemaining": 52, "TyreLife": 12, "TyreLifeRatio": 12 / 18, "PositionAtPit": 7},
            {"desc": "P12 long first stint lap 30", "LapNumber": 30, "LapsRemaining": 40, "TyreLife": 24, "TyreLifeRatio": 24 / 30, "PositionAtPit": 12, "Stint": 1},
            {"desc": "Late stop from P5 lap 48", "LapNumber": 48, "LapsRemaining": 22, "TyreLife": 30, "TyreLifeRatio": 30 / 48, "PositionAtPit": 5, "Stint": 2},
        ]
        for ex in examples:
            desc = ex.pop("desc")
            prob = predict_one(rf, ex, defaults)
            print(f"\n  Scenario: {desc}")
            print(f"  -> P(net position gain): {prob * 100:.1f}%  |  {verdict(prob)}")


if __name__ == "__main__":
    main()
