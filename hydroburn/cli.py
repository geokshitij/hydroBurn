"""
Command-line interface for HydroBurn.
"""

import click
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import warnings

from .config import load_config, HydroBurnConfig
from .io import load_catchment, compute_catchment_area, load_streamflow
from .preprocess import (
    quality_control, cfs_to_m3s, separate_baseflow, 
    compute_baseflow_index, compute_annual_bfi
)
from .analysis import (
    extract_events, compare_event_distributions,
    compute_fdc, compare_fdc,
    extract_annual_maxima, bootstrap_flood_quantiles, compare_flood_quantiles,
    compute_summary_statistics
)
from .visualization import (
    set_hydroburn_style, plot_hydrograph, plot_hydrograph_detail,
    plot_fdc_comparison, plot_annual_maxima, plot_flood_frequency,
    plot_event_boxplots, plot_monthly_boxplots, plot_baseflow_index,
    plot_catchment_map
)
from .report import generate_report

# Suppress warnings for cleaner CLI output
warnings.filterwarnings("ignore")

@click.group()
@click.version_option()
def cli():
    """HydroBurn: Post-Wildfire Runoff Analysis Framework."""
    pass

@cli.command()
@click.option('--catchment', required=True, type=click.Path(exists=True), help='Path to catchment shapefile')
@click.option('--streamflow', required=True, type=click.Path(exists=True), help='Path to streamflow CSV')
@click.option('--fire-date', required=True, help='Fire date (YYYY-MM-DD)')
@click.option('--config', type=click.Path(exists=True), help='Path to config YAML')
@click.option('--output-dir', default='output', help='Output directory')
@click.option('--units', default='cfs', type=click.Choice(['cfs', 'm3s']), help='Input discharge units')
def run(catchment, streamflow, fire_date, config, output_dir, units):
    """Run full analysis pipeline."""
    
    # 1. Setup Configuration
    cfg = load_config(
        config,
        catchment_file=catchment,
        streamflow_file=streamflow,
        fire_date=fire_date,
        output_dir=output_dir,
        discharge_units=units
    )
    
    cfg.create_output_dirs()
    click.echo(f"🚀 Starting HydroBurn analysis for {cfg.fire_name}")
    click.echo(f"📂 Output directory: {cfg.output_run_dir}")
    
    # 2. Load Data
    click.echo("📥 Loading data...")
    gdf = load_catchment(cfg.catchment_file)
    cfg.catchment_area_km2 = compute_catchment_area(gdf)
    click.echo(f"   Catchment Area: {cfg.catchment_area_km2:.2f} km²")
    
    df = load_streamflow(
        cfg.streamflow_file, 
        timezone=cfg.timezone,
        discharge_column=cfg.discharge_column
    )
    click.echo(f"   Streamflow records: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")
    
    # 3. Preprocessing
    click.echo("⚙️  Preprocessing...")
    # QC
    df, qc_report = quality_control(df, max_gap_days=cfg.max_gap_days_interpolate)
    
    # Unit conversion
    if cfg.discharge_units == 'cfs':
        df['discharge'] = cfs_to_m3s(df['discharge'])
        click.echo("   Converted cfs to m³/s")
    
    # Baseflow separation
    df = separate_baseflow(df, alpha=cfg.lh_alpha, passes=cfg.lh_passes)
    
    # Save processed data
    df.to_csv(cfg.data_dir / "processed_streamflow.csv")
    
    # 4. Analysis
    click.echo("📊 Running analysis...")
    
    # Split pre/post
    fire_dt = pd.Timestamp(cfg.fire_date).tz_localize(df.index.tz)
    pre_df = df[df.index < fire_dt]
    post_df = df[df.index >= fire_dt]
    
    # Events
    click.echo("   Extracting storm events...")
    pre_events = extract_events(pre_df, area_km2=cfg.catchment_area_km2)
    post_events = extract_events(post_df, area_km2=cfg.catchment_area_km2)
    
    event_stats = compare_event_distributions(pre_events, post_events)
    event_stats.to_csv(cfg.tables_dir / "event_stats.csv", index=False)
    
    # FDC
    click.echo("   Computing flow duration curves...")
    fdc_comp = compare_fdc(pre_df['discharge'], post_df['discharge'])
    fdc_comp.to_csv(cfg.tables_dir / "fdc_comparison.csv", index=False)
    
    # Flood Frequency
    click.echo("   Analyzing flood frequency...")
    annual_max = extract_annual_maxima(df)
    annual_max.to_csv(cfg.tables_dir / "annual_maxima.csv", index=False)
    
    pre_max = annual_max[annual_max['year'] < fire_dt.year]['max_discharge'].values
    post_max = annual_max[annual_max['year'] >= fire_dt.year]['max_discharge'].values
    
    flood_quantiles = compare_flood_quantiles(
        pre_max, post_max, 
        return_periods=cfg.return_periods,
        n_bootstrap=cfg.bootstrap_iterations
    )
    flood_quantiles.to_csv(cfg.tables_dir / "flood_quantiles.csv", index=False)
    
    # Baseflow
    annual_bfi = compute_annual_bfi(df)
    
    # 5. Visualization
    click.echo("🎨 Generating figures...")
    set_hydroburn_style()
    
    plot_hydrograph(df, fire_date=cfg.fire_date, output_path=cfg.figures_dir / "fig01_hydrograph.png")
    plot_hydrograph_detail(df, fire_date=cfg.fire_date, output_path=cfg.figures_dir / "fig02_hydrograph_detail.png")
    plot_fdc_comparison(pre_df['discharge'], post_df['discharge'], output_path=cfg.figures_dir / "fig03_fdc_comparison.png")
    plot_annual_maxima(annual_max, fire_year=fire_dt.year, output_path=cfg.figures_dir / "fig04_annual_maxima.png")
    
    # Need bootstrap results for flood plot
    # Re-run bootstrap just for plotting data structure (or modify plot function to take table)
    # For now, we'll just plot the points and lines from the table
    plot_flood_frequency(pre_max, post_max, flood_quantiles, output_path=cfg.figures_dir / "fig05_flood_frequency.png")
    
    # Convert events to DF for plotting
    pre_events_df = pd.DataFrame([e.to_dict() for e in pre_events])
    post_events_df = pd.DataFrame([e.to_dict() for e in post_events])
    plot_event_boxplots(pre_events_df, post_events_df, output_path=cfg.figures_dir / "fig06_event_boxplots.png")
    
    plot_baseflow_index(annual_bfi, fire_year=fire_dt.year, output_path=cfg.figures_dir / "fig07_baseflow_index.png")
    plot_monthly_boxplots(df, fire_date=cfg.fire_date, output_path=cfg.figures_dir / "fig10_monthly_stats.png")
    plot_catchment_map(gdf, output_path=cfg.figures_dir / "fig09_catchment_map.png")
    
    # 6. Reporting
    click.echo("📝 Generating report...")
    results = {
        'fdc_comparison': fdc_comp,
        'event_stats': event_stats,
        'flood_quantiles': flood_quantiles,
        'pre_start': pre_df.index.min().date(),
        'pre_end': pre_df.index.max().date(),
        'pre_years': (pre_df.index.max() - pre_df.index.min()).days / 365.25,
        'post_start': post_df.index.min().date(),
        'post_end': post_df.index.max().date(),
        'post_years': (post_df.index.max() - post_df.index.min()).days / 365.25,
        'n_events': len(pre_events) + len(post_events),
        'n_pre_events': len(pre_events),
        'n_post_events': len(post_events),
    }
    
    generate_report(cfg, results, cfg.output_run_dir / "report.md")
    cfg.save()
    
    click.echo("✅ Analysis complete!")

if __name__ == '__main__':
    cli()
