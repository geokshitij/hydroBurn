# HydroBurn

**Autonomous Post-Wildfire Runoff Analysis Framework**

HydroBurn is a Python framework designed to assess the hydrologic impacts of wildfires using streamflow and catchment data. It produces scientifically defensible, reproducible results and publication-quality figures.

## Features

- **Automated Data Processing:** Gap filling, unit conversion, baseflow separation.
- **Event Extraction:** Identifies storm events and computes metrics (peak, volume, timing).
- **Statistical Analysis:** Pre/post change detection, flow duration curves, flood frequency analysis.
- **Visualization:** Generates comprehensive suite of figures (hydrographs, FDCs, boxplots).
- **Reporting:** Auto-generates Markdown and HTML reports.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Basic Run

```bash
python -m hydroburn.cli run \
    --catchment catchments/USGS_08390500.shp \
    --streamflow streamflows/08390500.csv \
    --fire-date 2024-06-17
```

### With Configuration File

```bash
python -m hydroburn.cli run --config config.yaml
```

## Output

The framework creates a timestamped output directory containing:
- `figures/`: PNG plots
- `tables/`: CSV results
- `data/`: Processed time series
- `report.html`: Summary report

## License

MIT
