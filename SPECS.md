# HydroBurn: Post-Wildfire Runoff Change Detection Framework

## Specification Document v1.0

**Project**: Autonomous analysis of wildfire impacts on runoff and flood risk  
**Case Study**: 2024 South Fork / Salt Fires, Ruidoso, NM (USGS 08390500 – Rio Ruidoso at Hollywood, NM)  
**Fire Date**: June 17, 2024  
**Author**: HydroBurn Analysis Agent  
**Date**: November 2025

---

## 1. Scientific Objectives

### Primary Research Questions
1. **Did the 2024 Ruidoso wildfires significantly alter runoff characteristics?**
   - Changes in peak flow magnitudes
   - Changes in runoff volume per event
   - Changes in hydrograph shape (time-to-peak, recession)

2. **How has flood risk changed post-fire?**
   - Quantify changes in flood quantiles (Q₁₀, Q₅₀, Q₁₀₀)
   - Provide confidence intervals on hazard ratios

3. **What hydrologic indicators suggest wildfire-driven change?**
   - Baseflow index reduction
   - Quickflow fraction increase
   - Flow duration curve shifts
   - Event response metrics

---

## 2. Data Inventory

### 2.1 Available Data

| Dataset | File | Format | Description |
|---------|------|--------|-------------|
| Catchment boundary | `catchments/USGS_08390500.shp` | ESRI Shapefile (WGS84) | Watershed polygon for area calculation |
| Streamflow | `streamflows/08390500.csv` | CSV | Daily mean discharge, 1939-10-01 to present |

### 2.2 Streamflow Data Schema

```
Column: datetime          – ISO 8601 with timezone (UTC)
Column: site_no           – USGS station ID (08390500)
Column: 00060_Mean        – Daily mean discharge (ft³/s, cfs)
Column: 00060_Mean_cd     – Data quality code (A=Approved, P=Provisional)
```

### 2.3 Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Fire start date | 2024-06-17 | South Fork Fire ignition |
| Gauge timezone | America/Denver (MST/MDT) | USGS metadata |
| Discharge units | ft³/s (cfs) | USGS NWIS |
| Temporal resolution | Daily | USGS daily values |
| Record length | ~85 years (1939–2025) | USGS NWIS |

---

## 3. Analysis Framework

### 3.1 Module Architecture

```
hydroburn/
├── __init__.py
├── config.py              # Configuration dataclass
├── cli.py                 # Command-line interface
├── io/
│   ├── __init__.py
│   ├── load_catchment.py  # Shapefile/GeoJSON loader
│   └── load_streamflow.py # CSV parser with QC
├── preprocess/
│   ├── __init__.py
│   ├── qc.py              # Gap detection, interpolation
│   ├── units.py           # Unit conversion (cfs → m³/s → mm/day)
│   └── baseflow.py        # Lyne-Hollick filter
├── analysis/
│   ├── __init__.py
│   ├── events.py          # Peak detection, event extraction
│   ├── statistics.py      # Pre/post stats, tests
│   ├── flood_freq.py      # Gumbel/L-moments, bootstrap
│   └── fdc.py             # Flow duration curves
├── visualization/
│   ├── __init__.py
│   ├── hydrograph.py      # Time series with fire date
│   ├── fdc_plot.py        # Flow duration curves
│   ├── boxplots.py        # Event comparisons
│   ├── annual_max.py      # Annual maxima series
│   └── map_catchment.py   # Catchment map with gauge
└── report/
    ├── __init__.py
    └── generate.py        # Markdown/HTML report
```

### 3.2 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT STAGE                                  │
├─────────────────────────────────────────────────────────────────────┤
│  1. Load catchment shapefile → compute area (km²)                   │
│  2. Load streamflow CSV → parse datetime, set timezone              │
│  3. Load configuration (fire_date, units, thresholds)               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING STAGE                             │
├─────────────────────────────────────────────────────────────────────┤
│  4. Quality control: detect gaps, flag issues                       │
│  5. Gap filling: linear interpolation for gaps ≤ 2 days             │
│  6. Unit conversion: cfs → m³/s                                     │
│  7. Depth conversion: m³/s → mm/day using catchment area            │
│  8. Baseflow separation: Lyne-Hollick recursive filter              │
│  9. Compute quickflow = total - baseflow                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ANALYSIS STAGE                                 │
├─────────────────────────────────────────────────────────────────────┤
│  10. Split data at fire_date → pre-fire / post-fire subsets         │
│  11. Extract storm events: peak detection, volume, timing           │
│  12. Compute flow duration curves (pre/post)                        │
│  13. Extract annual maxima series (water years)                     │
│  14. Fit flood frequency distributions (Gumbel + bootstrap CI)      │
│  15. Statistical tests: Mann-Whitney, K-S, Pettitt, Mann-Kendall    │
│  16. Compute diagnostic indicators (BFI, Qf fraction, TTP, etc.)    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT STAGE                                  │
├─────────────────────────────────────────────────────────────────────┤
│  17. Generate figures (publication-quality, 300 DPI)                │
│  18. Export CSV tables (events, annual max, bootstrap results)      │
│  19. Generate summary report (Markdown + HTML)                      │
│  20. Create reproducible output folder with timestamp               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Method Specifications

### 4.1 Catchment Area Calculation

```python
# Method: GeoPandas with UTM projection for accurate area
# Input: Shapefile or GeoJSON (WGS84)
# Output: Area in km²

def compute_catchment_area(gdf: gpd.GeoDataFrame) -> float:
    """
    Compute catchment area by:
    1. Determine UTM zone from centroid
    2. Reproject to UTM (meters)
    3. Compute area, convert to km²
    """
    centroid = gdf.geometry.centroid.iloc[0]
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = 'north' if centroid.y >= 0 else 'south'
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"
    gdf_utm = gdf.to_crs(utm_crs)
    area_m2 = gdf_utm.geometry.area.sum()
    return area_m2 / 1e6  # km²
```

### 4.2 Unit Conversions

```python
# Constants
CFS_TO_M3S = 0.028316847  # 1 ft³/s = 0.0283168 m³/s

def cfs_to_m3s(Q_cfs: float) -> float:
    """Convert cubic feet per second to cubic meters per second."""
    return Q_cfs * CFS_TO_M3S

def discharge_to_depth(Q_m3s: float, area_km2: float, dt_seconds: float = 86400) -> float:
    """
    Convert discharge to runoff depth.
    
    Q (m³/s) * dt (s) = volume (m³)
    depth (mm) = volume (m³) / area (m²) * 1000
    
    For daily data: dt = 86400 s
    """
    volume_m3 = Q_m3s * dt_seconds
    area_m2 = area_km2 * 1e6
    depth_m = volume_m3 / area_m2
    return depth_m * 1000  # mm
```

### 4.3 Baseflow Separation (Lyne-Hollick Filter)

```python
def lyne_hollick_filter(Q: np.ndarray, alpha: float = 0.925, 
                        passes: int = 3) -> np.ndarray:
    """
    Lyne-Hollick recursive digital filter for baseflow separation.
    
    Parameters:
    -----------
    Q : array
        Total streamflow time series
    alpha : float
        Filter parameter (0.9-0.98, higher = more smoothing)
    passes : int
        Number of passes (typically 3: forward, backward, forward)
    
    Returns:
    --------
    Qb : array
        Baseflow component
    
    Reference:
    ----------
    Lyne, V., & Hollick, M. (1979). Stochastic time-variable rainfall-runoff 
    modelling. Institute of Engineers Australia.
    
    Nathan, R. J., & McMahon, T. A. (1990). Evaluation of automated techniques 
    for base flow and recession analyses. Water Resources Research.
    """
    Qf = np.zeros_like(Q)
    
    for p in range(passes):
        # Alternate direction for each pass
        if p % 2 == 0:
            indices = range(1, len(Q))
        else:
            indices = range(len(Q) - 2, -1, -1)
        
        for i in indices:
            Qf[i] = alpha * Qf[i-1 if p % 2 == 0 else i+1] + \
                    (1 + alpha) / 2 * (Q[i] - Q[i-1 if p % 2 == 0 else i+1])
            Qf[i] = max(0, Qf[i])  # Quickflow cannot be negative
            Qf[i] = min(Qf[i], Q[i])  # Quickflow cannot exceed total
    
    Qb = Q - Qf
    return Qb
```

### 4.4 Event Extraction

```python
@dataclass
class StormEvent:
    """Container for storm event characteristics."""
    start_time: pd.Timestamp
    peak_time: pd.Timestamp
    end_time: pd.Timestamp
    peak_discharge_m3s: float
    peak_discharge_mm: float
    volume_mm: float  # Total runoff volume
    quickflow_volume_mm: float  # Above baseflow
    baseflow_volume_mm: float
    time_to_peak_hours: float
    rising_limb_slope: float  # mm/hr or m³/s/hr
    recession_constant: float  # Optional
    duration_hours: float

def extract_events(Q: pd.Series, Qb: pd.Series, 
                   threshold_percentile: float = 90,
                   min_separation_hours: float = 48,
                   min_peak_m3s: float = 0.1) -> List[StormEvent]:
    """
    Extract storm events from hydrograph.
    
    Method:
    1. Identify peaks above threshold
    2. For each peak, find event start (rising limb) and end (recession)
    3. Compute event metrics
    
    Parameters:
    -----------
    Q : pd.Series
        Total discharge (datetime index)
    Qb : pd.Series
        Baseflow component
    threshold_percentile : float
        Percentile of Q for peak detection (default: 90th)
    min_separation_hours : float
        Minimum time between independent events (default: 48h)
    min_peak_m3s : float
        Minimum peak discharge to consider (default: 0.1 m³/s)
    """
    # Implementation details in code
    pass
```

### 4.5 Flow Duration Curve

```python
def compute_fdc(Q: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute flow duration curve.
    
    Returns:
    --------
    exceedance : array
        Exceedance probability (0-100%)
    discharge : array
        Sorted discharge values (descending)
    """
    Q_sorted = np.sort(Q.dropna().values)[::-1]  # Descending
    n = len(Q_sorted)
    exceedance = (np.arange(1, n + 1) / (n + 1)) * 100
    return exceedance, Q_sorted
```

### 4.6 Flood Frequency Analysis

```python
def fit_gumbel(annual_maxima: np.ndarray) -> Tuple[float, float]:
    """
    Fit Gumbel (Type I Extreme Value) distribution using L-moments.
    
    Parameters:
    -----------
    annual_maxima : array
        Annual maximum discharge series
    
    Returns:
    --------
    location : float
        Gumbel location parameter (μ)
    scale : float
        Gumbel scale parameter (σ)
    
    Quantile function:
    Q_T = μ - σ * ln(-ln(1 - 1/T))
    """
    # L-moments estimation
    x = np.sort(annual_maxima)
    n = len(x)
    
    # L1 (mean)
    L1 = np.mean(x)
    
    # L2
    b0 = L1
    b1 = np.sum([(i * x[i]) for i in range(n)]) / (n * (n - 1) / 2)
    L2 = 2 * b1 - b0
    
    # Gumbel parameters from L-moments
    scale = L2 / np.log(2)
    location = L1 - 0.5772 * scale  # Euler-Mascheroni constant
    
    return location, scale

def gumbel_quantile(T: float, location: float, scale: float) -> float:
    """Compute T-year flood quantile from Gumbel distribution."""
    return location - scale * np.log(-np.log(1 - 1/T))

def bootstrap_flood_quantiles(annual_maxima: np.ndarray, 
                               return_periods: List[float] = [10, 50, 100],
                               n_bootstrap: int = 1000,
                               ci: float = 0.95) -> Dict:
    """
    Bootstrap confidence intervals for flood quantiles.
    
    Returns:
    --------
    dict with keys for each return period:
        'estimate': point estimate
        'ci_lower': lower CI bound
        'ci_upper': upper CI bound
        'std': bootstrap standard deviation
    """
    n = len(annual_maxima)
    results = {T: {'samples': []} for T in return_periods}
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(annual_maxima, size=n, replace=True)
        loc, scale = fit_gumbel(sample)
        
        for T in return_periods:
            Q_T = gumbel_quantile(T, loc, scale)
            results[T]['samples'].append(Q_T)
    
    alpha = (1 - ci) / 2
    for T in return_periods:
        samples = np.array(results[T]['samples'])
        results[T]['estimate'] = np.median(samples)
        results[T]['ci_lower'] = np.percentile(samples, alpha * 100)
        results[T]['ci_upper'] = np.percentile(samples, (1 - alpha) * 100)
        results[T]['std'] = np.std(samples)
        del results[T]['samples']
    
    return results
```

### 4.7 Statistical Tests

```python
def pre_post_statistical_tests(pre: np.ndarray, post: np.ndarray) -> Dict:
    """
    Comprehensive statistical comparison of pre/post distributions.
    
    Tests performed:
    1. Mann-Whitney U: Non-parametric test for distribution shift
    2. Kolmogorov-Smirnov: Test for distribution difference
    3. Welch's t-test: Difference in means (for reference)
    4. Levene's test: Difference in variance
    
    Returns:
    --------
    dict with test results including statistics and p-values
    """
    from scipy import stats
    
    results = {}
    
    # Mann-Whitney U test (robust to non-normality)
    stat, p = stats.mannwhitneyu(pre, post, alternative='two-sided')
    results['mann_whitney'] = {'statistic': stat, 'pvalue': p}
    
    # Kolmogorov-Smirnov test
    stat, p = stats.ks_2samp(pre, post)
    results['ks_test'] = {'statistic': stat, 'pvalue': p}
    
    # Welch's t-test
    stat, p = stats.ttest_ind(pre, post, equal_var=False)
    results['welch_t'] = {'statistic': stat, 'pvalue': p}
    
    # Levene's test for variance
    stat, p = stats.levene(pre, post)
    results['levene'] = {'statistic': stat, 'pvalue': p}
    
    # Descriptive statistics
    results['pre_stats'] = {
        'n': len(pre), 'mean': np.mean(pre), 'median': np.median(pre),
        'std': np.std(pre), 'min': np.min(pre), 'max': np.max(pre)
    }
    results['post_stats'] = {
        'n': len(post), 'mean': np.mean(post), 'median': np.median(post),
        'std': np.std(post), 'min': np.min(post), 'max': np.max(post)
    }
    
    # Effect size (ratio of medians, percent change)
    results['effect'] = {
        'median_ratio': np.median(post) / np.median(pre) if np.median(pre) > 0 else np.nan,
        'mean_ratio': np.mean(post) / np.mean(pre) if np.mean(pre) > 0 else np.nan,
        'percent_change_median': ((np.median(post) - np.median(pre)) / np.median(pre) * 100) 
                                  if np.median(pre) > 0 else np.nan
    }
    
    return results

def pettitt_test(x: np.ndarray) -> Tuple[int, float, float]:
    """
    Pettitt's test for change point detection.
    
    Returns:
    --------
    change_point : int
        Index of most likely change point
    K : float
        Test statistic
    p_value : float
        Approximate p-value
    """
    n = len(x)
    U = np.zeros(n)
    
    for t in range(n):
        for i in range(t + 1):
            for j in range(t + 1, n):
                U[t] += np.sign(x[j] - x[i])
    
    K = np.max(np.abs(U))
    change_point = np.argmax(np.abs(U))
    
    # Approximate p-value
    p_value = 2 * np.exp(-6 * K**2 / (n**3 + n**2))
    
    return change_point, K, p_value
```

---

## 5. Output Specifications

### 5.1 Figures

| Figure | Filename | Description |
|--------|----------|-------------|
| 1 | `fig01_hydrograph.png` | Full time series with fire date vertical line |
| 2 | `fig02_hydrograph_detail.png` | Zoomed view: 1 year pre to 1 year post fire |
| 3 | `fig03_fdc_comparison.png` | Flow duration curves (pre vs post, log scale) |
| 4 | `fig04_annual_maxima.png` | Annual maximum series with trend |
| 5 | `fig05_flood_frequency.png` | Gumbel plot with CI bands |
| 6 | `fig06_event_boxplots.png` | Boxplots of event metrics (peaks, volumes) |
| 7 | `fig07_baseflow_index.png` | Rolling BFI time series |
| 8 | `fig08_event_metrics.png` | Time-to-peak, rising slope pre/post |
| 9 | `fig09_catchment_map.png` | Catchment boundary with gauge location |
| 10 | `fig10_monthly_stats.png` | Monthly flow statistics pre/post |

### 5.2 Figure Style Guidelines

```python
FIGURE_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
}

COLORS = {
    'pre_fire': '#2166AC',    # Blue
    'post_fire': '#B2182B',   # Red
    'fire_event': '#D95F02',  # Orange
    'baseflow': '#7570B3',    # Purple
    'quickflow': '#1B9E77',   # Teal
    'confidence': '#CCCCCC',  # Gray
}
```

### 5.3 CSV Output Tables

| Table | Filename | Columns |
|-------|----------|---------|
| Events | `events.csv` | event_id, start_time, peak_time, end_time, peak_m3s, peak_mm, volume_mm, quickflow_mm, ttp_hours, period |
| Annual Maxima | `annual_maxima.csv` | water_year, max_discharge_m3s, max_date, period |
| Bootstrap Results | `flood_quantiles.csv` | return_period, pre_estimate, pre_ci_lower, pre_ci_upper, post_estimate, post_ci_lower, post_ci_upper, ratio, ratio_ci_lower, ratio_ci_upper |
| Statistics | `statistics_summary.csv` | metric, pre_value, post_value, change_percent, test_statistic, p_value, significant |
| Flow Duration | `fdc_data.csv` | exceedance_pct, pre_discharge_m3s, post_discharge_m3s |

### 5.4 Output Directory Structure

```
output/
└── run_YYYYMMDD_HHMMSS/
    ├── figures/
    │   ├── fig01_hydrograph.png
    │   ├── fig02_hydrograph_detail.png
    │   ├── ...
    │   └── fig10_monthly_stats.png
    ├── tables/
    │   ├── events.csv
    │   ├── annual_maxima.csv
    │   ├── flood_quantiles.csv
    │   ├── statistics_summary.csv
    │   └── fdc_data.csv
    ├── data/
    │   ├── processed_streamflow.csv
    │   └── baseflow_separation.csv
    ├── report.md
    ├── report.html
    └── config.json
```

---

## 6. Configuration

### 6.1 Configuration Schema

```python
@dataclass
class HydroBurnConfig:
    """Configuration for HydroBurn analysis."""
    
    # Input files
    catchment_file: str = "catchments/USGS_08390500.shp"
    streamflow_file: str = "streamflows/08390500.csv"
    
    # Key dates
    fire_date: str = "2024-06-17"
    
    # Data specifications
    discharge_column: str = "00060_Mean"
    datetime_column: str = "datetime"
    discharge_units: str = "cfs"  # 'cfs' or 'm3s'
    timezone: str = "America/Denver"
    
    # Baseflow separation
    lh_alpha: float = 0.925
    lh_passes: int = 3
    
    # Event extraction
    event_threshold_percentile: float = 90
    min_event_separation_hours: float = 48
    min_peak_m3s: float = 0.1
    
    # Flood frequency
    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    return_periods: List[int] = field(default_factory=lambda: [2, 5, 10, 25, 50, 100])
    
    # Quality control
    max_gap_days_interpolate: int = 2
    
    # Output
    output_dir: str = "output"
    figure_format: str = "png"
    figure_dpi: int = 300
```

### 6.2 Example Configuration File (config.yaml)

```yaml
# HydroBurn Configuration
# Rio Ruidoso at Hollywood, NM (USGS 08390500)
# 2024 South Fork / Salt Fires Analysis

input:
  catchment_file: "catchments/USGS_08390500.shp"
  streamflow_file: "streamflows/08390500.csv"

fire:
  date: "2024-06-17"
  name: "South Fork / Salt Fires"
  
data:
  discharge_column: "00060_Mean"
  datetime_column: "datetime"
  discharge_units: "cfs"
  timezone: "America/Denver"

baseflow:
  method: "lyne_hollick"
  alpha: 0.925
  passes: 3

events:
  threshold_percentile: 90
  min_separation_hours: 48
  min_peak_m3s: 0.1

flood_frequency:
  distribution: "gumbel"
  bootstrap_iterations: 1000
  confidence_level: 0.95
  return_periods: [2, 5, 10, 25, 50, 100]

quality_control:
  max_gap_days_interpolate: 2
  flag_negative_values: true

output:
  directory: "output"
  figure_format: "png"
  figure_dpi: 300
  generate_html_report: true
```

---

## 7. Command-Line Interface

### 7.1 CLI Specification

```bash
# Basic usage
python -m hydroburn run \
    --catchment catchments/USGS_08390500.shp \
    --streamflow streamflows/08390500.csv \
    --fire-date 2024-06-17

# With configuration file
python -m hydroburn run --config config.yaml

# Individual analysis modules
python -m hydroburn baseflow --streamflow data.csv --output baseflow.csv
python -m hydroburn events --streamflow data.csv --fire-date 2024-06-17
python -m hydroburn flood-freq --streamflow data.csv --fire-date 2024-06-17
python -m hydroburn report --input-dir output/run_xxx
```

### 7.2 CLI Arguments

```
Usage: hydroburn run [OPTIONS]

Options:
  --catchment PATH         Path to catchment shapefile/GeoJSON [required]
  --streamflow PATH        Path to streamflow CSV [required]
  --fire-date DATE         Fire date (YYYY-MM-DD) [required]
  --config PATH            Path to configuration YAML
  --output-dir PATH        Output directory [default: output]
  --units [cfs|m3s]        Discharge units [default: cfs]
  --timezone TEXT          Timezone [default: America/Denver]
  --bootstrap INT          Bootstrap iterations [default: 1000]
  --verbose / --quiet      Verbosity level
  --help                   Show this message and exit.
```

---

## 8. Dependencies

### 8.1 Required Packages

```
# Core
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# Geospatial
geopandas>=0.13.0
shapely>=2.0.0
pyproj>=3.5.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# CLI and configuration
click>=8.1.0
pyyaml>=6.0
pydantic>=2.0.0

# Reporting
jinja2>=3.1.0
markdown>=3.4.0

# Optional (for extended analysis)
# lmoments3>=1.0.0  # L-moment estimation
# pymannkendall>=1.4.0  # Mann-Kendall trend test
```

### 8.2 Python Version

- Minimum: Python 3.10
- Recommended: Python 3.11+

---

## 9. Diagnostic Indicators for Wildfire Impact

The analysis will compute and report the following indicators:

| Indicator | Pre-fire Expected | Post-fire Expected | Interpretation |
|-----------|------------------|-------------------|----------------|
| Baseflow Index (BFI) | Higher | Lower | Reduced infiltration/storage |
| Quickflow Fraction | Lower | Higher | More surface runoff |
| Mean Time-to-Peak | Longer | Shorter | Faster watershed response |
| Rising Limb Slope | Gentler | Steeper | Flashier response |
| Peak-to-Volume Ratio | Lower | Higher | Less storage, more peaky |
| FDC High-Flow Quantile (Q₁₀) | Lower | Higher | Increased high flows |
| FDC Low-Flow Quantile (Q₉₀) | May be higher | May be lower | Reduced baseflow |
| Recession Constant | Larger | Smaller | Faster recession |

---

## 10. Quality Assurance Checklist

### 10.1 Data Quality
- [ ] All dates parsed correctly
- [ ] No negative discharge values
- [ ] Gaps identified and documented
- [ ] Units verified (cfs vs m³/s)
- [ ] Timezone applied correctly

### 10.2 Analysis Quality
- [ ] Catchment area reasonable (compare to USGS metadata)
- [ ] Baseflow never exceeds total flow
- [ ] Annual maxima extracted for each water year
- [ ] Bootstrap results converged (check variance)
- [ ] Statistical tests assumptions checked

### 10.3 Output Quality
- [ ] All figures generated without errors
- [ ] CSV files have correct headers
- [ ] Report renders correctly
- [ ] Configuration saved for reproducibility

---

## 11. Limitations and Caveats

### 11.1 Current Limitations (boundary + streamflow only)

1. **No rainfall data**: Cannot compute runoff ratios or CN values directly
2. **No burn severity map**: Cannot stratify analysis by burn severity
3. **Daily resolution**: May miss sub-daily peaks; true peaks likely higher
4. **Short post-fire record**: ~1.5 years post-fire (June 2024 – Nov 2025)
5. **Confounding factors**: Climate variability, land use changes, etc.

### 11.2 Statistical Caveats

1. Post-fire period may be too short for reliable flood frequency estimates
2. Non-stationarity assumption violated (by design – we're testing for change)
3. Climate variability may confound fire effects
4. Seasonal differences in pre/post periods should be considered

### 11.3 Recommendations for Future Work

1. Acquire gridded precipitation (PRISM, Stage IV) for event-based analysis
2. Obtain MTBS or BARC burn severity maps
3. Get NLCD land cover for pre-fire conditions
4. Consider paired watershed analysis if reference gauge available

---

## 12. Expected Results Summary Table

The final report will include a summary table like:

| Metric | Pre-Fire | Post-Fire | Change (%) | p-value | Significant? |
|--------|----------|-----------|------------|---------|--------------|
| Mean Annual Max (m³/s) | X.XX | X.XX | +XX% | 0.XXX | Yes/No |
| Median Event Peak (m³/s) | X.XX | X.XX | +XX% | 0.XXX | Yes/No |
| Q₁₀₀ Estimate (m³/s) | X.XX | X.XX | +XX% | — | — |
| Baseflow Index | 0.XX | 0.XX | -XX% | 0.XXX | Yes/No |
| Mean Time-to-Peak (hrs) | X.X | X.X | -XX% | 0.XXX | Yes/No |

---

## 13. Next Steps

1. **Review this specification** – confirm scope, methods, outputs
2. **Implement core modules** – start with I/O and preprocessing
3. **Test with data** – verify catchment area, streamflow parsing
4. **Build analysis pipeline** – events, FDC, flood frequency
5. **Create visualizations** – one figure at a time
6. **Generate report template** – Markdown/HTML
7. **CLI integration** – tie everything together
8. **Documentation** – README, docstrings, examples

---

*Specification prepared for HydroBurn v1.0*  
*Ready for implementation phase*
