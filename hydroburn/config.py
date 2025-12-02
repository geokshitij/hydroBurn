"""
Configuration module for HydroBurn.

Defines the HydroBurnConfig dataclass with all analysis parameters.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional
import yaml
from datetime import datetime


@dataclass
class HydroBurnConfig:
    """Configuration for a HydroBurn analysis."""
    # Core inputs
    station_id: str
    streamflow_file: str
    catchment_file: str
    precipitation_file: Optional[str] = None  # Path to precipitation CSV
    
    # Optional inputs with defaults
    fire_history_file: Optional[str] = None  # Path to fire history shapefile
    burn_area_threshold_pct: float = 1.0  # Min % of catchment burned to count as a fire year
    burn_area_threshold_km2: Optional[float] = None  # Min area (km2) burned to count as a fire year
    config_file: Optional[str] = None
    output_dir: str = "output"
    discharge_units: str = "cfs"  # 'cfs' or 'm3s'
    discharge_column: str = "discharge"
    timezone: str = "UTC"
    
    # Key dates
    fire_date: Optional[str] = None
    fire_name: str = "Wildfire Event"
    
    # Station metadata
    station_name: str = "Rio Ruidoso at Hollywood, NM"
    
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
    restrict_to_fire_record: bool = True  # Restrict analysis to the period of fire data
    
    # Output settings
    figure_format: str = "png"
    figure_dpi: int = 300
    generate_html_report: bool = True
    
    # Derived attributes (set after loading)
    catchment_area_km2: Optional[float] = None
    run_timestamp: Optional[str] = None
    
    # Analysis window
    analysis_window_years: int = 2
    post_fire_window_months: Optional[int] = None
    
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
    config_data = {}
    if config_path:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

    # Apply overrides from CLI
    config_data.update(overrides)

    # Ensure required arguments are present
    if 'streamflow_file' not in config_data:
        raise ValueError("Missing required argument: streamflow_file")
    if 'catchment_file' not in config_data:
        raise ValueError("Missing required argument: catchment_file")

    # Set station_id from filename if not provided
    if 'station_id' not in config_data:
        config_data['station_id'] = Path(config_data['streamflow_file']).stem

    # Filter out any keys not in the dataclass
    valid_keys = {f.name for f in fields(HydroBurnConfig)}
    filtered_data = {k: v for k, v in config_data.items() if k in valid_keys}

    return HydroBurnConfig(**filtered_data)
