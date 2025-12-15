import json
import math
from pathlib import Path

import pandas as pd


# Adjust if your Excel is elsewhere
BASE_DATA_PATH = Path("data/metadata/germany/Base_Data_exc_Dulan.xlsx")
OUTPUT_DIR = Path("data/metadata/germany")

# Mapping from provider internal ID to your anonymized plant IDs
ID_TO_PLANT = {
    "1_AMS": "plant_01",
    "2_FAB": "plant_02",
    "3_GRB": "plant_03",
    "4_SAD": "plant_04",
    "5_SCW": "plant_05",
    "6_TER1": "plant_06",
    "6_TER2": "plant_06",  # merge 6_TER1 and 6_TER2
}


def to_float(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_tilt(beta_val):
    """Neigungswinkel (β), '20°' -> 20.0"""
    if pd.isna(beta_val):
        return None
    s = str(beta_val)
    s = s.replace("°", "").strip()
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_sy_azimuth(alpha_val):
    """
    Ausrichtungs-winkel (α), S = 0, clockwise.
    Examples:
      '0°'
      '17°'
      '-1° und 18°'
      '0° und 45°'
    We return the mean in the provider convention.
    """
    if pd.isna(alpha_val):
        return None
    s = str(alpha_val)
    s = s.replace("°", "")
    s = s.replace("und", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]

    vals = []
    for p in parts:
        p = p.replace(",", ".")
        p = p.replace("−", "-")  # just in case
        try:
            vals.append(float(p))
        except ValueError:
            continue

    if not vals:
        return None
    return sum(vals) / len(vals)


def sy_to_pvlib_azimuth(alpha_sy):
    """
    Convert Syneco azimuth (S=0, clockwise)
    to PVLib convention (N=0, clockwise).
    Formula: alpha_pvlib = (alpha_sy + 180) mod 360
    """
    if alpha_sy is None:
        return None
    return (alpha_sy + 180.0) % 360.0


def parse_tracker(text):
    if pd.isna(text):
        return False
    s = str(text).strip().lower()
    return "ja" in s


def parse_mount_type(text):
    if pd.isna(text):
        return "unknown"
    s = str(text).strip().lower()
    if "frei" in s:
        return "ground"
    if "dach" in s:
        return "roof"
    return "unknown"


def to_str_or_none(x):
    return None if pd.isna(x) else str(x)


def main():
    if not BASE_DATA_PATH.exists():
        raise FileNotFoundError(f"Base data Excel not found at {BASE_DATA_PATH}")

    df = pd.read_excel(BASE_DATA_PATH, sheet_name="Basedata")

    # Keep only rows with INT_ID
    df = df[df["INT_ID [Dulan]"].notna()].copy()

    # Attach your plant_id mapping
    df["provider_internal_id"] = df["INT_ID [Dulan]"].astype(str)
    df["plant_id"] = df["provider_internal_id"].map(ID_TO_PLANT)

    # Drop anything unmapped, just in case
    df = df[df["plant_id"].notna()].copy()

    # Fix decimal commas in coordinates and capacity
    df["installed_capacity_kw"] = df["Installierte Leistung"].map(to_float)
    df["lat"] = df["Geographische Lage"].map(to_float)
    df["lon"] = df["Unnamed: 5"].map(to_float)
    df["tilt_deg"] = df["Neigungswinkel (β)"].map(parse_tilt)
    df["azimuth_deg_sy"] = df["Ausrichtungs-winkel (α)"].map(parse_sy_azimuth)

    # Group by your plant_id to merge 6_TER1 and 6_TER2 into plant_06
    groups = df.groupby("plant_id")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for plant_id, g in groups:
        # Capacity for weighting
        caps = g["installed_capacity_kw"].values
        cap_total = caps.sum()

        def cap_weighted(col):
            vals = g[col].values
            num = 0.0
            den = 0.0
            for v, c in zip(vals, caps):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                num += float(v) * c
                den += c
            if den == 0:
                return None
            return num / den

        installed_capacity_kw = round(cap_total, 1) if cap_total is not None else None
        latitude = round(cap_weighted("lat"), 6) if cap_weighted("lat") is not None else None
        longitude = round(cap_weighted("lon"), 6) if cap_weighted("lon") is not None else None
        tilt_deg = round(cap_weighted("tilt_deg"), 2) if cap_weighted("tilt_deg") is not None else None
        azimuth_deg_sy = round(cap_weighted("azimuth_deg_sy"), 2) if cap_weighted("azimuth_deg_sy") is not None else None
        azimuth_deg = round(sy_to_pvlib_azimuth(azimuth_deg_sy), 2) if azimuth_deg_sy is not None else None

        # Assume timezone and country for all
        country = "DE"
        timezone = "Europe/Berlin"

        # Use first row's mount and tracker and raw metadata
        first = g.iloc[0]

        mount_type = parse_mount_type(first["Freifläche / Aufdach"])
        tracker = parse_tracker(first["Solartracker "])

        raw_block = {
            "technologie": to_str_or_none(first["Technologie"]),
            "freiflaeche_aufdach": to_str_or_none(first["Freifläche / Aufdach"]),
            "solartracker_text": to_str_or_none(first["Solartracker "]),
            "anzahl_module": to_str_or_none(first["Anzahl Module"]),
            "modultyp": to_str_or_none(first["Modultyp"]),
            "anzahl_wechselrichter": to_str_or_none(first["Anzahl Wechselrichter"]),
            "wechselrichtertyp": to_str_or_none(first["Wechsel-richtertyp"]),
            "wechselrichter_leistung": to_str_or_none(first["Wechselrichter-leistung"]),
            "netzbetreiber": to_str_or_none(first["Übertragungs-netzbetreiber"]),
        }

        # If multiple provider IDs (TER1 + TER2), store them all
        provider_ids = sorted(set(g["provider_internal_id"].astype(str).tolist()))
        if len(provider_ids) == 1:
            provider_internal_id = provider_ids[0]
        else:
            provider_internal_id = "+".join(provider_ids)

        meta = {
            "plant_id": plant_id,
            "provider_internal_id": provider_internal_id,
            "country": country,
            "timezone": timezone,
            "installed_capacity_kw": installed_capacity_kw,
            "latitude": latitude,
            "longitude": longitude,
            "tilt_deg": tilt_deg,
            "azimuth_deg_sy": azimuth_deg_sy,
            "azimuth_deg": azimuth_deg,
            "mount_type": mount_type,
            "tracker": tracker,
            "raw": raw_block,
            "source": "syneco_base_data_v1",
        }

        out_path = OUTPUT_DIR / f"{plant_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
