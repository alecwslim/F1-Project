"""
F1 Pit Stop Data Collector (2020–2024)
Extracts pit stop records including driver position at pit and final race finish.

Install: pip install fastf1 pandas
Usage:   python f1_pitstop_collector.py
Output:  f1_pitstop_data_2020_2024.csv
"""

import os
import fastf1
import pandas as pd

CACHE_DIR   = "f1_cache"
OUTPUT_FILE = "f1_pitstop_data_2024.csv"
YEARS       = [2024]

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def get_final_positions(session) -> dict[str, int | str]:
    """Return {driver_abbreviation: classified_position} from race results."""
    results = session.results[["Abbreviation", "ClassifiedPosition", "Position"]].copy()
    pos_map = {}
    for _, row in results.iterrows():
        abbr = row["Abbreviation"]
        # ClassifiedPosition is the official result (includes DNF/DSQ as strings)
        # Fall back to Position (numeric finishing order) when available
        classified = row["ClassifiedPosition"]
        pos_map[abbr] = classified if pd.notna(classified) else row["Position"]
    return pos_map


def extract_pitstops(session, year: int, event_name: str, round_number: int) -> list[dict]:
    """
    Return one record per pit stop, with:
      - driver position at the pit-in lap
      - driver's final classified race position
    """
    laps = session.laps
    final_positions = get_final_positions(session)

    # A pit stop lap is one where PitInTime is recorded (car entered pit lane).
    # PitOutTime is stored on the *following* lap (the out lap), so we index
    # all laps by driver+lap number to look it up.
    pit_laps = laps[laps["PitInTime"].notna()].copy()

    # Build a lookup: (driver, lap_number) -> PitOutTime
    out_time_lookup = (
        laps[laps["PitOutTime"].notna()]
        .set_index(["Driver", "LapNumber"])["PitOutTime"]
        .to_dict()
    )

    records = []
    for _, lap in pit_laps.iterrows():
        driver     = lap["Driver"]
        lap_number = int(lap["LapNumber"])

        # Position column = car's position at the end of this lap (last known
        # before/at pit entry — NaN for some early laps, stored as float by pandas)
        position_at_pit = lap["Position"]
        if pd.notna(position_at_pit):
            position_at_pit = int(position_at_pit)
        else:
            position_at_pit = None

        final_pos = final_positions.get(driver)

        # PitOutTime lives on the next lap (out lap)
        pit_out = out_time_lookup.get((driver, lap_number + 1))
        pit_out_s = round(pit_out.total_seconds(), 3) if pit_out is not None else None

        records.append({
            "Year":              year,
            "RoundNumber":       round_number,
            "EventName":         event_name,
            "Driver":            driver,
            "Team":              lap.get("Team", None),
            "LapNumber":         lap_number,
            "Stint":             int(lap["Stint"]),
            "Compound":          lap.get("Compound", None),
            "TyreLife":          int(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else None,
            "PitInTime_s":       round(lap["PitInTime"].total_seconds(), 3),
            "PitOutTime_s":      pit_out_s,
            "PositionAtPit":     position_at_pit,
            "FinalPosition":     final_pos,
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
            round_num  = int(event["RoundNumber"])
            event_name = event["EventName"]

            print(f"  [{round_num:02d}] {event_name} ...", end=" ", flush=True)
            try:
                session = fastf1.get_session(year, round_num, "R")
                # Load only what we need — skip telemetry/weather/messages
                session.load(telemetry=False, weather=False, messages=False)

                records = extract_pitstops(session, year, event_name, round_num)
                all_records.extend(records)
                print(f"{len(records)} pit stops collected")

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
    print(f"\nDone. {len(df)} total pit stop records saved to '{OUTPUT_FILE}'")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
