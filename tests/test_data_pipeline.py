"""
test_data_pipeline.py - unit tests for the data/preprocessing layer.

All inputs are small synthetic frames; no real (NDA) data, no network. These tests import the real
repository functions, so they fail if the pipeline logic regresses (DST handling, scale-factor
provenance, split purging, gap-aware windows, plausibility checks).
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_alignment_script():
    spec = importlib.util.spec_from_file_location(
        "check_weather_alignment", REPO_ROOT / "scripts" / "check_weather_alignment.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Time handling (DST)
# ---------------------------------------------------------------------------
class TestTimeUtils:
    def test_utc_index_is_uniform_and_tz_aware(self):
        from src.data.time_utils import utc_index_from_epoch

        idx = utc_index_from_epoch(1_672_527_600, 3600, 48)
        assert str(idx.tz) == "UTC"
        assert (idx.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()

    def test_repair_legacy_labels_matches_true_utc_across_dst(self):
        from src.data.time_utils import repair_legacy_naive_local

        truth = pd.date_range("2023-01-01 00:00", periods=24 * 365, freq="1h", tz="Europe/Berlin").tz_convert("UTC")
        # legacy labels: uniform naive steps from the first local wall clock
        first_local = truth[0].tz_convert("Europe/Berlin").tz_localize(None)
        # generate the way the old fetcher did: uniform in UTC, labelled by the *first* local time
        utc = pd.date_range(truth[0], periods=len(truth), freq="1h")
        labels = pd.Series(pd.date_range(first_local, periods=len(utc), freq="1h"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            repaired = repair_legacy_naive_local(labels)
        assert (repaired == utc).all()

    def test_repair_rejects_non_uniform_labels(self):
        from src.data.time_utils import repair_legacy_naive_local

        labels = pd.Series(pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 03:00"]))
        with pytest.raises(ValueError):
            repair_legacy_naive_local(labels)

    @pytest.mark.parametrize("legacy", [True, False])
    def test_weather_preprocessing_has_no_summer_offset(self, legacy):
        """The pipeline must keep daily GHI peaks on the clear-sky peak in every month."""
        chk = _load_alignment_script()
        lat, lon = 48.14, 11.58  # generic Bavaria-area point, not a real plant
        raw = chk.synthetic_hourly(lat, lon, legacy=legacy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = chk.run_pipeline_on_hourly(raw)
        offsets = chk.monthly_peak_offset(w["shortwave_radiation_instant"], lat, lon)
        assert offsets.abs().max() <= 0.5, offsets.to_dict()


# ---------------------------------------------------------------------------
# validate_pv_frame
# ---------------------------------------------------------------------------
def _pv(n=500):
    t = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"timestamp_utc": t, "power_norm": np.abs(np.sin(np.arange(n) / 20)) * 0.8})


class TestValidatePvFrame:
    def test_clean_frame_has_no_issues(self):
        from src.data.schema import validate_pv_frame

        assert validate_pv_frame(_pv()) == []

    def test_flags_non_utc_timezone(self):
        from src.data.schema import validate_pv_frame

        df = _pv()
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
        assert any("not UTC" in i for i in validate_pv_frame(df))

    def test_flags_range_stuck_gap_and_duplicates(self):
        from src.data.schema import validate_pv_frame

        df = _pv()
        df.loc[10, "power_norm"] = 3.0
        df.loc[100:140, "power_norm"] = 0.5
        dup = pd.concat([df, df.iloc[[5]]], ignore_index=True)
        issues = " | ".join(validate_pv_frame(dup))
        assert "above" in issues and "stuck value" in issues and "duplicate" in issues
        gap = df.drop(index=range(200, 400))
        assert any("largest gap" in i for i in validate_pv_frame(gap, max_gap_steps=100))

    def test_raise_on_error(self):
        from src.data.schema import validate_pv_frame

        df = _pv()
        df["power_norm"] = -0.1
        with pytest.raises(ValueError):
            validate_pv_frame(df, raise_on_error=True)

    def test_germany_processed_path_exists(self):
        from src.data.schema import DataPaths

        assert DataPaths(REPO_ROOT).germany_processed.name == "germany"


# ---------------------------------------------------------------------------
# Windows must not span gaps (opt-in), and chronological purged split
# ---------------------------------------------------------------------------
class TestWindows:
    def _frame(self):
        t = pd.date_range("2023-01-01", periods=60, freq="15min", tz="UTC")
        df = pd.DataFrame({"t": t, "x": np.arange(60.0), "y": np.arange(60.0)})
        return df.drop(index=range(20, 30)).reset_index(drop=True)  # 10-row gap

    def test_default_keeps_positional_windows(self):
        torch = pytest.importorskip("torch")  # noqa: F841
        from src.features.sequence_generator import SimpleWindowDataset

        d = SimpleWindowDataset(self._frame(), "t", None, ["x"], "y", input_window=8, forecast_horizon=2)
        assert len(d) == 41 and d.n_dropped_gap_windows == 0

    def test_require_contiguous_drops_gap_windows(self):
        pytest.importorskip("torch")
        from src.features.sequence_generator import SimpleWindowDataset

        d = SimpleWindowDataset(
            self._frame(), "t", None, ["x"], "y", input_window=8, forecast_horizon=2, require_contiguous=True
        )
        assert d.n_dropped_gap_windows == 9 and len(d) == 32
        # every kept window spans exactly (input + horizon - 1) steps of 15 min: no jump across the gap
        t = d.df["t"]
        span = pd.Timedelta(minutes=15) * (8 + 2 - 1)
        assert all(t.loc[out_end] - t.loc[in_start] == span for in_start, _, _, out_end in d.indices)

    def test_chronological_split_is_disjoint_ordered_and_purged(self):
        from src.preprocessing.germany_pretrain_normalize_split import chronological_split_with_purge

        tr, va, te = chronological_split_with_purge(1000, 0.7, 0.15, purge=10)
        assert tr[-1] < va[0] < va[-1] < te[0]
        assert va[0] - tr[-1] > 10 and te[0] - va[-1] > 10
        assert len(set(tr) & set(va)) == 0 and len(set(va) & set(te)) == 0


# ---------------------------------------------------------------------------
# Scale-factor script: provenance + idempotence
# ---------------------------------------------------------------------------
class TestScalingProvenance:
    def _setup(self, tmp_path, monkeypatch, scale_down):
        import src.data.fix_germany_pv_scaling as fx

        pv_dir, meta_dir = tmp_path / "pv", tmp_path / "meta"
        pv_dir.mkdir(), meta_dir.mkdir()
        monkeypatch.setattr(fx, "PV_DIR", pv_dir)
        monkeypatch.setattr(fx, "META_DIR", meta_dir)
        (meta_dir / "plant_x.json").write_text(json.dumps({"installed_capacity_kw": 1000.0}))
        kw = np.abs(np.sin(np.linspace(0, 20, 400))) * 800.0 * scale_down
        df = pd.DataFrame({
            "timestamp_utc": pd.date_range("2023-01-01", periods=400, freq="15min", tz="UTC"),
            "power_kw": kw, "power_w": kw * 1000.0, "power_norm": kw / 1000.0,
        })
        df.to_parquet(pv_dir / "plant_x_pv_15min.parquet", index=False)
        return fx, pv_dir

    def test_broken_plant_is_scaled_to_capacity_and_logged(self, tmp_path, monkeypatch):
        fx, pv_dir = self._setup(tmp_path, monkeypatch, scale_down=1e-6)
        fx.fix_plant_if_needed("plant_x")
        df = pd.read_parquet(pv_dir / "plant_x_pv_15min.parquet")
        assert df["power_kw"].max() == pytest.approx(1000.0)
        assert df["power_norm"].max() == pytest.approx(1.0)
        rec = json.loads((pv_dir / "plant_x_scaling.json").read_text())
        assert rec["applied"] is True and rec["scale"] > 1e5 and rec["installed_capacity_kw"] == 1000.0

    def test_second_run_is_idempotent_and_keeps_record(self, tmp_path, monkeypatch):
        fx, pv_dir = self._setup(tmp_path, monkeypatch, scale_down=1e-6)
        fx.fix_plant_if_needed("plant_x")
        first = (pv_dir / "plant_x_pv_15min.parquet").read_bytes()
        fx.fix_plant_if_needed("plant_x")
        assert (pv_dir / "plant_x_pv_15min.parquet").read_bytes() == first
        assert json.loads((pv_dir / "plant_x_scaling.json").read_text())["applied"] is True

    def test_healthy_plant_untouched(self, tmp_path, monkeypatch):
        fx, pv_dir = self._setup(tmp_path, monkeypatch, scale_down=1.0)
        before = (pv_dir / "plant_x_pv_15min.parquet").read_bytes()
        fx.fix_plant_if_needed("plant_x")
        assert (pv_dir / "plant_x_pv_15min.parquet").read_bytes() == before
        assert json.loads((pv_dir / "plant_x_scaling.json").read_text())["applied"] is False

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        fx, pv_dir = self._setup(tmp_path, monkeypatch, scale_down=1e-6)
        before = (pv_dir / "plant_x_pv_15min.parquet").read_bytes()
        fx.fix_plant_if_needed("plant_x", dry_run=True)
        assert (pv_dir / "plant_x_pv_15min.parquet").read_bytes() == before
        assert not (pv_dir / "plant_x_scaling.json").exists()


# ---------------------------------------------------------------------------
# PVLib feature generator: DNI handling
# ---------------------------------------------------------------------------
class TestPvlibFeatures:
    def _weather(self):
        import pvlib

        t = pd.date_range("2023-06-21", periods=96, freq="15min", tz="UTC")
        cs = pvlib.location.Location(48.7, 12.6).get_clearsky(t)
        w = pd.DataFrame({
            "timestamp_utc": t,
            "shortwave_radiation_instant": cs["ghi"].values,
            "diffuse_radiation_instant": cs["dhi"].values,
            "direct_radiation_instant": (cs["ghi"] - cs["dhi"]).values,  # horizontal beam, NOT dni
        })
        return w, cs

    def test_derived_dni_matches_given_dni(self):
        from src.data.pvlib_feature_generator import generate_pvlib_features

        w, cs = self._weather()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            derived = generate_pvlib_features(w, 48.7, 12.6, tilt=25, azimuth=180)
            w2 = w.assign(direct_normal_irradiance_instant=cs["dni"].values)
            given = generate_pvlib_features(w2, 48.7, 12.6, tilt=25, azimuth=180)
        assert derived["poa_global"].max() == pytest.approx(given["poa_global"].max(), rel=0.05)

    def test_missing_irradiance_falls_back_with_warning(self):
        from src.data.pvlib_feature_generator import generate_pvlib_features

        w, _ = self._weather()
        with pytest.warns(UserWarning, match="clear-sky"):
            out = generate_pvlib_features(w[["timestamp_utc"]], 48.7, 12.6)
        assert out["poa_global"].max() > 0
