"""
F1 Lap Data Collector (2020–2024)
Extracts one record per lap including position, pit-stop fields, pit counts,
gaps to cars ahead/behind, weather, and track status.

Install: pip install fastf1 pandas
Usage:   python f1_lap_collector.py
Output:  f1_lap_data_2020_2024.csv
"""

import os
import fastf1
import pandas as pd

CACHE_DIR = "f1_cache"
OUTPUT_FILE = "f1_lap_data2.csv"
YEARS = [2022,2023,2024]

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def get_final_positions(session) -> dict[str, int | str]:
    """Return {driver_abbreviation: classified_position} from race results."""
    results = session.results[["Abbreviation", "ClassifiedPosition", "Position"]].copy()
    pos_map = {}
    for _, row in results.iterrows():
        abbr = row["Abbreviation"]
        classified = row["ClassifiedPosition"]
        pos_map[abbr] = classified if pd.notna(classified) else row["Position"]
    return pos_map


def get_weather_lookup(session) -> pd.DataFrame:
    """Return weather data sorted by Time for nearest-time lookup."""
    weather = session.weather_data.copy()
    weather = weather.sort_values("Time").reset_index(drop=True)
    return weather


def parse_gap_value(value):
    """
    Convert FastF1 gap/interval values into seconds when possible.
    Works with Timedelta, strings like '+1.234', and numeric values.
    """
    if value is None or pd.isna(value):
        return None

    if hasattr(value, "total_seconds"):
        return round(float(value.total_seconds()), 3)

    s = str(value).strip()
    if not s:
        return None

    s = s.replace("+", "")

    # Timedelta-like strings
    if ":" in s or "day" in s.lower() or "days" in s.lower():
        try:
            return round(pd.to_timedelta(s).total_seconds(), 3)
        except Exception:
            pass

    # Plain numeric strings
    try:
        return round(float(s), 3)
    except Exception:
        return None


def nearest_weather(weather_df: pd.DataFrame, target_time) -> dict:
    """Return the weather row closest in time to target_time."""
    if weather_df.empty or target_time is None or pd.isna(target_time):
        return {}

    idx = (weather_df["Time"] - target_time).abs().idxmin()
    row = weather_df.loc[idx]

    return {
        "AirTemp_C": round(float(row["AirTemp"]), 1) if pd.notna(row["AirTemp"]) else None,
        "TrackTemp_C": round(float(row["TrackTemp"]), 1) if pd.notna(row["TrackTemp"]) else None,
        "Humidity_pct": round(float(row["Humidity"]), 1) if pd.notna(row["Humidity"]) else None,
        "Pressure_mbar": round(float(row["Pressure"]), 1) if pd.notna(row["Pressure"]) else None,
        "WindSpeed_ms": round(float(row["WindSpeed"]), 1) if pd.notna(row["WindSpeed"]) else None,
        "WindDirection": int(row["WindDirection"]) if pd.notna(row["WindDirection"]) else None,
        "Rainfall": bool(row["Rainfall"]) if pd.notna(row["Rainfall"]) else False,
    }


# '1'=Green  '2'=Yellow  '4'=SafetyCar  '5'=RedFlag  '6'=VSC  '7'=VSCEnding
def parse_track_status(status_str) -> dict:
    """Return boolean flag columns for each track status code."""
    if pd.isna(status_str) or str(status_str).strip() == "":
        s = ""
    else:
        s = str(status_str)

    return {
        "TrackStatus": s if s else "Unknown",
        "GreenFlag": "1" in s,
        "YellowFlag": "2" in s,
        "SafetyCar": "4" in s,
        "RedFlag": "5" in s,
        "VirtualSafetyCar": "6" in s or "7" in s,
    }


def get_track_condition(rainfall: bool, compound: str | None) -> str:
    """Derive surface condition from rainfall and compound."""
    c = (compound or "").upper()
    if c == "WET":
        return "Wet"
    if c == "INTERMEDIATE":
        return "Intermediate"
    if rainfall:
        return "Damp"
    return "Dry"


def build_position_context(laps: pd.DataFrame) -> dict[tuple[int, str], dict]:
    """
    Build a lookup keyed by (lap_number, driver) with:
      - DriverAhead
      - DriverBehind
      - GapToDriverAhead_s
      - GapToDriverBehind_s

    Gaps are derived from GapToLeader when available.
    """
    context = {}

    for lap_num, group in laps.groupby("LapNumber", dropna=True):
        g = group.copy()
        g["PositionAtLapEnd"] = pd.to_numeric(g["Position"], errors="coerce")
        g["GapToLeader_s"] = g["GapToLeader"].apply(parse_gap_value) if "GapToLeader" in g.columns else None

        g = g.dropna(subset=["PositionAtLapEnd"]).sort_values(["PositionAtLapEnd", "Driver"])

        rows = g[["Driver", "GapToLeader_s"]].to_dict("records")
        for i, row in enumerate(rows):
            ahead = rows[i - 1] if i > 0 else None
            behind = rows[i + 1] if i < len(rows) - 1 else None

            gap_ahead = None
            if ahead and row["GapToLeader_s"] is not None and ahead["GapToLeader_s"] is not None:
                gap_ahead = round(row["GapToLeader_s"] - ahead["GapToLeader_s"], 3)

            gap_behind = None
            if behind and row["GapToLeader_s"] is not None and behind["GapToLeader_s"] is not None:
                gap_behind = round(behind["GapToLeader_s"] - row["GapToLeader_s"], 3)

            context[(int(lap_num), row["Driver"])] = {
                "DriverAhead": ahead["Driver"] if ahead else None,
                "DriverBehind": behind["Driver"] if behind else None,
                "GapToDriverAhead_s": gap_ahead,
                "GapToDriverBehind_s": gap_behind,
            }

    return context


def extract_laps(session, year: int, event_name: str, round_number: int) -> list[dict]:
    """
    Return one record per lap.
    """
    laps = session.laps.copy().sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    final_positions = get_final_positions(session)
    weather_df = get_weather_lookup(session)

    # Pit count by driver
    laps["PitCountSoFar"] = laps.groupby("Driver")["PitInTime"].transform(lambda s: s.notna().cumsum())

    # Position-based lookup for ahead/behind names and gaps
    position_context = build_position_context(laps)

    # PitOutTime is stored on the out lap, so look it up by driver + lap number.
    out_time_lookup = (
        laps[laps["PitOutTime"].notna()]
        .set_index(["Driver", "LapNumber"])["PitOutTime"]
        .to_dict()
    )

    records = []

    for _, lap in laps.iterrows():
        driver = lap["Driver"]
        lap_number = int(lap["LapNumber"]) if pd.notna(lap["LapNumber"]) else None

        position_at_lap_end = lap.get("Position")
        position_at_lap_end = int(position_at_lap_end) if pd.notna(position_at_lap_end) else None

        final_pos = final_positions.get(driver)

        pit_in = lap.get("PitInTime")
        pit_out = out_time_lookup.get((driver, lap_number + 1)) if lap_number is not None else None

        pit_in_s = round(pit_in.total_seconds(), 3) if pd.notna(pit_in) else None
        pit_out_s = round(pit_out.total_seconds(), 3) if pit_out is not None else None

        weather_time = pit_in if pd.notna(pit_in) else lap.get("Time")
        weather = nearest_weather(weather_df, weather_time)

        track_status = parse_track_status(lap.get("TrackStatus"))
        compound = lap.get("Compound", None)
        track_cond = get_track_condition(weather.get("Rainfall", False), compound)

        ctx = position_context.get((lap_number, driver), {})
        gap_ahead = parse_gap_value(lap.get("IntervalToPositionAhead"))
        gap_behind = parse_gap_value(lap.get("IntervalToPositionBehind"))

        # Fall back to computed values from position order if interval columns are missing
        if gap_ahead is None:
            gap_ahead = ctx.get("GapToDriverAhead_s")
        if gap_behind is None:
            gap_behind = ctx.get("GapToDriverBehind_s")

        records.append({
            "Year": year,
            "RoundNumber": round_number,
            "EventName": event_name,
            "Driver": driver,
            "Team": lap.get("Team", None),
            "LapNumber": lap_number,
            "Stint": int(lap["Stint"]) if pd.notna(lap["Stint"]) else None,
            "Compound": compound,
            "TyreLife": int(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else None,
            "LapTime_s": round(lap["LapTime"].total_seconds(), 3) if pd.notna(lap.get("LapTime")) else None,
            "PitInTime_s": pit_in_s,
            "PitOutTime_s": pit_out_s,
            "PitCountSoFar": int(lap["PitCountSoFar"]) if pd.notna(lap["PitCountSoFar"]) else 0,
            "PositionAtLapEnd": position_at_lap_end,
            "DriverAhead": ctx.get("DriverAhead"),
            "GapToDriverAhead_s": gap_ahead,
            "DriverBehind": ctx.get("DriverBehind"),
            "GapToDriverBehind_s": gap_behind,
            "FinalPosition": final_pos,
            **weather,
            **track_status,
            "TrackCondition": track_cond,
        })

    return records


def main():
    all_records = []

    for year in YEARS:
        print(f"\n{'='*60}")
        print(f"  Year: {year}")
        print(f"{'='*60}")

        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  Could not fetch schedule for {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            round_num = int(event["RoundNumber"])
            event_name = event["EventName"]

            print(f"  [{round_num:02d}] {event_name} ...", end=" ", flush=True)
            try:
                session = fastf1.get_session(year, round_num, "R")
                session.load(telemetry=False, weather=True, messages=False)

                records = extract_laps(session, year, event_name, round_num)
                all_records.extend(records)
                print(f"{len(records)} laps collected")

            except Exception as e:
                print(f"SKIPPED ({e})")
                continue

    if not all_records:
        print("\nNo records collected. Check your internet connection or fastf1 version.")
        return

    df = pd.DataFrame(all_records)

    # Coerce FinalPosition to numeric where possible (DNF/DSQ stay as-is)
    df["FinalPositionNumeric"] = pd.to_numeric(df["FinalPosition"], errors="coerce")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. {len(df)} total lap records saved to '{OUTPUT_FILE}'")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()