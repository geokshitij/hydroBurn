"""
Configuration module for HydroBurn.

Defines the HydroBurnConfig dataclass with all analysis parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml
from datetime import datetime


@dataclass
class HydroBurnConfig:
    """Configuration for HydroBurn post-wildfire runoff analysis."""
    
    # Input files
    catchment_file: str = "catchments/USGS_08390500.shp"
    streamflow_file: str = "streamflows/08390500.csv"
    
    # Key dates
    fire_date: str = "2024-06-17"
    fire_name: str = "South Fork / Salt Fires"
    
    # Station metadata
    station_id: str = "08390500"
    station_name: str = "Rio Ruidoso at Hollywood, NM"
    
    # Data specifications
    discharge_column: str = "00060_Mean"
    datetime_column: str = "datetime"
    discharge_units: str = "cfs"  # 'cfs' or 'm3s'
    timezone: str = "America/Denver"
    
    # Baseflow separation (Lyne-Hollick parameters)
    lh_alpha: float = 0.925
    lh_passes: int = 3
    
    # Event extraction
    event_threshold_percentile: float = 90
    min_event_separation_hours: float = 48
    min_peak_m3s: float = 0.1
    
    # Flood frequency analysis
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    return_periods: List[int] = field(default_factory=lambda: [2, 5, 10, 25, 50, 100])
    
    # Quality control
    max_gap_days_interpolate: int = 2
    flag_negative_values: bool = True
    
    # Output settings
    output_dir: str = "output"
    figure_format: str = "png"
    figure_dpi: int = 300
    generate_html_report: bool = True
    
    # Derived attributes (set after loading)
    catchment_area_km2: Optional[float] = None
    run_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Set derived attributes after initialization."""
        if self.run_timestamp is None:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @property
    def fire_datetime(self) -> datetime:
        """Parse fire_date string to datetime."""
        return datetime.strptime(self.fire_date, "%Y-%m-%d")
    
    @property
    def output_run_dir(self) -> Path:
        """Get the timestamped output directory for this run."""
        return Path(self.output_dir) / f"run_{self.run_timestamp}"
    
    @property
    def figures_dir(self) -> Path:
        """Get the figures subdirectory."""
        return self.output_run_dir / "figures"
    
    @property
    def tables_dir(self) -> Path:
        """Get the tables subdirectory."""
        return self.output_run_dir / "tables"
    
    @property
    def data_dir(self) -> Path:
        """Get the data subdirectory."""
        return self.output_run_dir / "data"
    
    def create_output_dirs(self) -> None:
        """Create all output directories."""
        for d in [self.output_run_dir, self.figures_dir, self.tables_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        if path is None:
            path = self.output_run_dir / "config.yaml"
        
        # Convert to dict, excluding None values
        config_dict = {
            k: v for k, v in self.__dict__.items() 
            if v is not None and not k.startswith('_')
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: str) -> "HydroBurnConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested structure if present
        flat_dict = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                flat_dict.update(value)
            else:
                flat_dict[key] = value
        
        # Map YAML keys to dataclass fields
        field_mapping = {
            'date': 'fire_date',
            'name': 'fire_name',
            'method': None,  # Ignored, always Lyne-Hollick
            'alpha': 'lh_alpha',
            'passes': 'lh_passes',
            'threshold_percentile': 'event_threshold_percentile',
            'min_separation_hours': 'min_event_separation_hours',
            'distribution': None,  # Ignored, always Gumbel
            'directory': 'output_dir',
        }
        
        processed = {}
        for key, value in flat_dict.items():
            mapped_key = field_mapping.get(key, key)
            if mapped_key is not None:
                processed[mapped_key] = value
        
        return cls(**processed)


def load_config(config_path: Optional[str] = None, **overrides) -> HydroBurnConfig:
    """
    Load configuration from file and/or keyword arguments.
    
    Parameters
    ----------
    config_path : str, optional
        Path to YAML configuration file
    **overrides : dict
        Keyword arguments to override config values
    
    Returns
    -------
    HydroBurnConfig
        Configuration object
    """
    if config_path:
        config = HydroBurnConfig.from_yaml(config_path)
    else:
        config = HydroBurnConfig()
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    
    return config
