"""
Tests for the configuration module.
"""
import pytest
from pathlib import Path
from hydroburn.config import HydroBurnConfig, load_config

def test_config_defaults():
    """Test that the default configuration loads correctly."""
    cfg = HydroBurnConfig()
    assert cfg.fire_date == "2024-06-17"
    assert cfg.discharge_units == "cfs"
    assert cfg.lh_alpha == 0.925

def test_config_override():
    """Test overriding config values."""
    cfg = load_config(fire_date="2022-01-01", lh_alpha=0.95)
    assert cfg.fire_date == "2022-01-01"
    assert cfg.lh_alpha == 0.95

def test_config_from_yaml(tmp_path):
    """Test loading configuration from a YAML file."""
    config_content = """
fire_date: "2023-05-10"
discharge_units: "m3s"
event_threshold_percentile: 85
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)

    cfg = load_config(config_path=str(config_file))
    assert cfg.fire_date == "2023-05-10"
    assert cfg.discharge_units == "m3s"
    assert cfg.event_threshold_percentile == 85

def test_output_paths():
    """Test that output paths are generated correctly."""
    cfg = HydroBurnConfig(output_dir="test_output")
    run_dir = cfg.output_run_dir
    assert run_dir.name.startswith("run_")
    assert cfg.figures_dir == run_dir / "figures"
    assert cfg.tables_dir == run_dir / "tables"
    assert cfg.data_dir == run_dir / "data"
