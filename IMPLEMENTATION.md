# HydroBurn Implementation Plan

## Phase 1: Project Setup & Core I/O

### Tasks
1. Create package structure
2. Implement configuration dataclass
3. Load catchment shapefile → compute area
4. Load streamflow CSV → parse, validate, convert units

### Deliverables
- `hydroburn/` package skeleton
- `config.py` with `HydroBurnConfig`
- `io/load_catchment.py`
- `io/load_streamflow.py`
- Unit tests for I/O

---

## Phase 2: Preprocessing

### Tasks
1. Quality control (gap detection, flagging)
2. Gap filling (linear interpolation ≤2 days)
3. Unit conversion (cfs → m³/s → mm/day)
4. Baseflow separation (Lyne-Hollick)

### Deliverables
- `preprocess/qc.py`
- `preprocess/units.py`
- `preprocess/baseflow.py`
- Processed streamflow with baseflow column

---

## Phase 3: Core Analysis

### Tasks
1. Pre/post split at fire date
2. Event extraction (peaks, volumes, timing)
3. Flow duration curves
4. Annual maxima extraction
5. Flood frequency (Gumbel + bootstrap)
6. Statistical tests

### Deliverables
- `analysis/events.py`
- `analysis/fdc.py`
- `analysis/flood_freq.py`
- `analysis/statistics.py`
- CSV output tables

---

## Phase 4: Visualization

### Tasks
1. Hydrograph with fire date
2. FDC comparison
3. Annual maxima plot
4. Flood frequency plot with CI
5. Event boxplots
6. Baseflow index time series
7. Catchment map

### Deliverables
- `visualization/*.py` modules
- All 10 publication figures

---

## Phase 5: Reporting & CLI

### Tasks
1. Markdown report template
2. HTML generation
3. CLI with Click
4. Output folder organization

### Deliverables
- `report/generate.py`
- `cli.py`
- Complete `run` command
- README with usage examples

---

## File Tree (Target)

```
HydroBurn/
├── catchments/
│   └── USGS_08390500.*        # Input shapefile
├── streamflows/
│   └── 08390500.csv           # Input streamflow
├── hydroburn/
│   ├── __init__.py
│   ├── config.py
│   ├── cli.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── load_catchment.py
│   │   └── load_streamflow.py
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── qc.py
│   │   ├── units.py
│   │   └── baseflow.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── events.py
│   │   ├── fdc.py
│   │   ├── flood_freq.py
│   │   └── statistics.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── hydrograph.py
│   │   ├── fdc_plot.py
│   │   ├── annual_max.py
│   │   ├── flood_freq_plot.py
│   │   ├── boxplots.py
│   │   ├── baseflow_plot.py
│   │   └── map_catchment.py
│   └── report/
│       ├── __init__.py
│       ├── generate.py
│       └── templates/
│           └── report.md.jinja
├── config.yaml                 # User configuration
├── output/                     # Generated outputs
├── SPECS.md                    # This specification
├── IMPLEMENTATION.md           # This file
├── README.md                   # User documentation
├── pyproject.toml              # Package definition
└── requirements.txt            # Dependencies
```

---

## Key Data Facts (from inspection)

| Item | Value |
|------|-------|
| Station | USGS 08390500 – Rio Ruidoso at Hollywood, NM |
| Record | 1939-10-01 to 2025-11-27 (~86 years) |
| Resolution | Daily mean discharge |
| Units | ft³/s (cfs) – column `00060_Mean` |
| Quality codes | A (Approved), P (Provisional) |
| Fire date | 2024-06-17 |
| Pre-fire period | 1939-10-01 to 2024-06-16 (~85 years) |
| Post-fire period | 2024-06-17 to 2025-11-27 (~1.5 years) |
| Catchment CRS | WGS84 (EPSG:4326) |

---

## Implementation Order

```
[1] config.py          → Define HydroBurnConfig dataclass
[2] io/load_catchment  → Read shapefile, compute area
[3] io/load_streamflow → Parse CSV, handle timezone
[4] preprocess/units   → CFS→m³/s, depth conversion
[5] preprocess/qc      → Gap detection, interpolation
[6] preprocess/baseflow→ Lyne-Hollick filter
[7] analysis/fdc       → Flow duration curves
[8] analysis/events    → Peak detection, extraction
[9] analysis/statistics→ Pre/post tests
[10] analysis/flood_freq→ Gumbel, bootstrap
[11] visualization/*   → All figures
[12] report/generate   → Markdown/HTML report
[13] cli.py            → Click CLI
[14] __main__.py       → Entry point
```

---

## Ready to Begin?

When you say "go", I will implement Phase 1 (project setup, config, I/O).
