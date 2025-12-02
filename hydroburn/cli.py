"""
Command-line interface for HydroBurn.
"""

import click
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import warnings
import numpy as np

from .config import load_config, HydroBurnConfig
from .io import load_catchment, compute_catchment_area, load_streamflow, load_fire_history, load_precipitation
from .preprocess import (
    quality_control, cfs_to_m3s, separate_baseflow, 
    compute_baseflow_index, compute_annual_bfi
)
from .analysis import (
    extract_events, compare_event_distributions,
    compute_fdc, compare_fdc,
    extract_annual_maxima, bootstrap_flood_quantiles, compare_flood_quantiles,
    compute_summary_statistics, perform_before_after_analysis,
    events_to_dataframe
)
from .visualization import (
    set_hydroburn_style, plot_hydrograph, plot_hydrograph_detail,
    plot_fdc_comparison, plot_annual_maxima, plot_flood_frequency,
    plot_event_boxplots, plot_monthly_boxplots, plot_baseflow_index,
    map_catchment, plot_before_after_summary
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
@click.option('--fire-history', required=True, type=click.Path(exists=True), help='Path to fire history shapefile')
@click.option('--fire-date', type=str, help='Specific fire date (YYYY-MM-DD) to use for pre/post split. Overrides automatic detection.')
@click.option('--burn-threshold', default=1.0, type=float, help='Minimum burned area percentage to consider a fire year.')
@click.option('--burn-threshold-km2', type=float, help='Minimum burned area in km2 to consider a fire year. Overrides percentage threshold.')
@click.option('--analysis-window', default=2, type=int, help='Years for before/after analysis window.')
@click.option('--post-fire-months', type=int, help='Limit post-fire analysis to X months after fire.')
@click.option('--precipitation', type=click.Path(exists=True), help='Path to precipitation CSV')
@click.option('--config', type=click.Path(exists=True), help='Path to config YAML')
@click.option('--output-dir', default='output', help='Output directory')
@click.option('--units', default='cfs', type=click.Choice(['cfs', 'm3s']), help='Input discharge units')
def run(catchment, streamflow, fire_history, fire_date, burn_threshold, burn_threshold_km2, analysis_window, post_fire_months, precipitation, config, output_dir, units):
    """Run full analysis pipeline."""
    
    # 1. Setup Configuration
    cfg = load_config(
        config,
        catchment_file=catchment,
        streamflow_file=streamflow,
        fire_history_file=fire_history,
        fire_date=fire_date,
        burn_area_threshold_pct=burn_threshold,
        burn_area_threshold_km2=burn_threshold_km2,
        analysis_window_years=analysis_window,
        post_fire_window_months=post_fire_months,
        precipitation_file=precipitation,
        output_dir=output_dir,
        discharge_units=units
    )
    
    cfg.create_output_dirs()
    click.echo(f"🚀 Starting HydroBurn analysis for {cfg.station_id}")
    click.echo(f"📂 Output directory: {cfg.output_run_dir}")
    
    # 2. Load Data
    click.echo("📥 Loading data...")
    gdf = load_catchment(cfg.catchment_file)
    cfg.catchment_area_km2 = compute_catchment_area(gdf)
    click.echo(f"   Catchment Area: {cfg.catchment_area_km2:.2f} km²")
    
    df = load_streamflow(
        cfg.streamflow_file, 
        timezone=cfg.timezone
    )
    click.echo(f"   Streamflow records: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")

    # Load precipitation if provided
    if cfg.precipitation_file:
        click.echo("🌧️ Loading precipitation...")
        precip_df = load_precipitation(cfg.precipitation_file)
        
        # Merge with streamflow
        # Ensure timezone awareness matches
        if df.index.tz is not None and precip_df.index.tz is None:
            precip_df.index = precip_df.index.tz_localize(df.index.tz)
        elif df.index.tz is None and precip_df.index.tz is not None:
            precip_df.index = precip_df.index.tz_convert(None)
            
        # Join
        df = df.join(precip_df, how='left')
        click.echo(f"   Merged precipitation data: {df['precipitation'].count()} records")

    # Load fire history and identify significant fire years
    click.echo("🔥 Loading fire history...")
    annual_burned_area = load_fire_history(cfg.fire_history_file, gdf)
    
    # Determine the start of the reliable fire record
    fire_record_start_year = None
    if not annual_burned_area.empty:
        fire_record_start_year = annual_burned_area['year'].min()

    # Restrict streamflow data to the fire record period if configured
    if cfg.restrict_to_fire_record and fire_record_start_year:
        click.echo(f"   Restricting analysis to fire record period (>= {fire_record_start_year})")
        df = df[df.index.year >= fire_record_start_year]
        click.echo(f"   New streamflow records: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")

    if not annual_burned_area.empty:
        annual_burned_area['burned_pct'] = (annual_burned_area['burned_area_km2'] / cfg.catchment_area_km2) * 100
        
        if cfg.burn_area_threshold_km2 is not None:
            significant_fires = annual_burned_area[annual_burned_area['burned_area_km2'] >= cfg.burn_area_threshold_km2]
            click.echo(f"   Filtering fires >= {cfg.burn_area_threshold_km2} km²")
        else:
            significant_fires = annual_burned_area[annual_burned_area['burned_pct'] >= cfg.burn_area_threshold_pct]
            click.echo(f"   Filtering fires >= {cfg.burn_area_threshold_pct}% of catchment")
            
        significant_fire_years = significant_fires['year'].tolist()
        # Get dates if available, otherwise construct from year
        if 'fire_date' in significant_fires.columns:
            significant_fire_dates = significant_fires['fire_date'].tolist()
        else:
            significant_fire_dates = [pd.Timestamp(f'{y}-01-01') for y in significant_fire_years]
    else:
        significant_fire_years = []
        significant_fire_dates = []

    if not significant_fire_years:
        click.echo("⚠️ No significant fire years found based on the threshold. Analysis will proceed without a pre/post fire split.")
        first_fire_year = None
        fire_dt = None
    else:
        click.echo(f"   Found {len(significant_fire_years)} significant fire year(s): {significant_fire_years}")
        
        if cfg.fire_date:
            fire_dt = pd.Timestamp(cfg.fire_date)
            if df.index.tz is not None and fire_dt.tz is None:
                fire_dt = fire_dt.tz_localize(df.index.tz)
            click.echo(f"   Using specified fire date for split: {fire_dt.date()}")
        else:
            # Default to the most recent significant fire if no date specified
            last_fire_year = max(significant_fire_years)
            fire_dt = pd.Timestamp(f'{last_fire_year}-01-01')
            if df.index.tz is not None and fire_dt.tz is None:
                fire_dt = fire_dt.tz_localize(df.index.tz)
            click.echo(f"   Using most recent significant fire year for split: {last_fire_year}")

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

    # New: Perform before-after analysis for each fire
    if significant_fire_years:
        click.echo("   Performing before-after analysis for each fire...")
        before_after_summary = perform_before_after_analysis(
            df,
            fire_years=significant_fire_years,
            window_years=cfg.analysis_window_years,
            discharge_col='discharge'
        )
        if not before_after_summary.empty:
            before_after_summary.to_csv(cfg.tables_dir / "before_after_summary.csv")
            click.echo("      -> Saved 'before_after_summary.csv'")
        else:
            click.echo("      -> No valid before/after periods found to analyze.")

    if cfg.post_fire_window_months and significant_fire_dates and not cfg.fire_date:
        # Multi-event analysis: Aggregate post-fire windows from ALL significant fires
        click.echo(f"   Multi-event analysis: Using {cfg.post_fire_window_months}-month window after each of {len(significant_fire_dates)} fires.")
        
        # Create boolean masks for the entire dataframe
        post_mask = pd.Series(False, index=df.index)
        pre_mask = pd.Series(False, index=df.index)
        
        for fire_date in significant_fire_dates:
            fd = pd.Timestamp(fire_date)
            if df.index.tz is not None and fd.tz is None:
                fd = fd.tz_localize(df.index.tz)
            
            # Post-fire window: [fire_date, fire_date + window]
            post_cutoff = fd + pd.DateOffset(months=cfg.post_fire_window_months)
            post_mask |= (df.index >= fd) & (df.index < post_cutoff)
            
            # Pre-fire window: [fire_date - window, fire_date]
            # User requested "last 12 month as pre"
            pre_start = fd - pd.DateOffset(months=cfg.post_fire_window_months)
            pre_mask |= (df.index >= pre_start) & (df.index < fd)
            
        # Handle superposition:
        # If a time step is marked as both Pre (for Fire B) and Post (for Fire A),
        # it is technically "disturbed" by Fire A, so it should NOT be in the Pre dataset.
        # Therefore, we exclude anything in post_mask from pre_mask.
        final_pre_mask = pre_mask & (~post_mask)
        
        post_df = df[post_mask]
        pre_df = df[final_pre_mask]
        
        click.echo(f"   Constructed composite post-fire record: {len(post_df)} days")
        click.echo(f"   Constructed composite pre-fire record: {len(pre_df)} days (filtered for overlap)")

    elif fire_dt:
        # Split pre/post based on single fire date
        pre_df = df[df.index < fire_dt]
        
        if cfg.post_fire_window_months:
            cutoff = fire_dt + pd.DateOffset(months=cfg.post_fire_window_months)
            post_df = df[(df.index >= fire_dt) & (df.index < cutoff)]
            click.echo(f"   Restricting post-fire analysis to {cfg.post_fire_window_months} months ({fire_dt.date()} to {cutoff.date()})")
        else:
            post_df = df[df.index >= fire_dt]
    else:
        # If no fire, analyze the whole period as "pre"
        pre_df = df
        post_df = pd.DataFrame()

    # Events
    click.echo("   Extracting storm events...")
    pre_events = extract_events(pre_df, area_km2=cfg.catchment_area_km2)
    post_events = extract_events(post_df, area_km2=cfg.catchment_area_km2)
    
    event_stats = compare_event_distributions(pre_events, post_events)
    event_stats.to_csv(cfg.tables_dir / "event_stats.csv", index=False)
    
    # FDC
    click.echo("   Computing flow duration curves...")
    fdc_comp = compare_fdc(pre_df['discharge'], post_df['discharge'] if not post_df.empty else None)
    fdc_comp.to_csv(cfg.tables_dir / "fdc_comparison.csv", index=False)
    
    # Flood Frequency
    click.echo("   Analyzing flood frequency...")
    pre_max = extract_annual_maxima(pre_df)
    pre_max['period'] = 'Pre-fire'
    
    if not post_df.empty:
        post_max = extract_annual_maxima(post_df)
        post_max['period'] = 'Post-fire'
    else:
        post_max = pd.DataFrame()
    
    # Combine for overall analysis
    annual_maxima = pd.concat([pre_max, post_max])
    
    # Bootstrap CI for flood frequency
    flood_quantiles = bootstrap_flood_quantiles(annual_maxima['max_discharge'])
    
    flood_quantiles_comp = compare_flood_quantiles(pre_max, post_max)
    
    # Save the overall flood frequency analysis
    flood_quantiles.to_csv(cfg.tables_dir / "flood_quantiles.csv")

    # Save the pre/post comparison separately
    if not flood_quantiles_comp.empty:
        flood_quantiles_comp.to_csv(cfg.tables_dir / "flood_quantiles_comparison.csv")
    
    # Summary Stats
    click.echo("   Computing summary statistics...")
    summary_stats = compute_summary_statistics(
        pre_df['discharge'], 
        post_df['discharge'] if not post_df.empty else None
    )
    
    # Calculate Runoff Ratio if precipitation is available
    if 'precipitation' in df.columns:
        def calc_rr(sub_df):
            if sub_df.empty: return None
            # Q (m3/s) * 86400 s/day = m3/day
            total_Q_m3 = sub_df['discharge'].sum() * 86400
            # (m3 * 1000 mm/m) / (km2 * 1e6 m2/km2) = mm
            total_Q_mm = (total_Q_m3 * 1000) / (cfg.catchment_area_km2 * 1e6)
            total_P_mm = sub_df['precipitation'].sum()
            return total_Q_mm / total_P_mm if total_P_mm > 0 else None

        summary_stats['pre_fire']['runoff_ratio'] = calc_rr(pre_df)
        summary_stats['pre_fire']['total_precip_mm'] = pre_df['precipitation'].sum() if not pre_df.empty else 0
        
        if not post_df.empty:
            summary_stats['post_fire']['runoff_ratio'] = calc_rr(post_df)
            summary_stats['post_fire']['total_precip_mm'] = post_df['precipitation'].sum()

    with open(cfg.tables_dir / "summary_stats.json", 'w') as f:
        # A custom encoder might be needed for numpy types
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        json.dump(summary_stats, f, indent=4, cls=NpEncoder)
    
    # BFI
    click.echo("   Computing baseflow index...")
    annual_bfi = compute_annual_bfi(df, baseflow_col='baseflow')
    annual_bfi.to_csv(cfg.tables_dir / "annual_bfi.csv", index=False)

    # 5. Visualization
    click.echo("🎨 Generating figures...")
    set_hydroburn_style()

    # New: Plot before-after summary
    if 'before_after_summary' in locals() and not before_after_summary.empty:
        plot_before_after_summary(
            before_after_summary,
            output_path=cfg.figures_dir / "fig10_before_after_summary.png"
        )

    # Hydrograph
    plot_hydrograph(
        df,
        fire_dates=significant_fire_dates,
        post_fire_window_months=cfg.post_fire_window_months,
        precip_col='precipitation' if 'precipitation' in df.columns else None,
        output_path=cfg.figures_dir / "fig01_hydrograph.png"
    )
    plot_hydrograph_detail(
        df,
        fire_dates=significant_fire_dates,
        output_path=cfg.figures_dir / "fig02_hydrograph_detail.png"
    )
    plot_fdc_comparison(
        pre_df['discharge'], 
        post_df['discharge'] if not post_df.empty else None, 
        output_path=cfg.figures_dir / "fig03_fdc_comparison.png"
    )
    plot_annual_maxima(
        annual_maxima,
        fire_years=significant_fire_years,
        output_path=cfg.figures_dir / "fig04_annual_maxima.png"
    )
    
    # Flood Frequency
    plot_flood_frequency(
        pre_max['max_discharge'],
        post_max['max_discharge'] if not post_max.empty else None,
        flood_quantiles_comp,
        output_path=cfg.figures_dir / "fig05_flood_frequency.png"
    )
    
    # Event Boxplots
    pre_events_df = events_to_dataframe(pre_events)
    post_events_df = events_to_dataframe(post_events)
    plot_event_boxplots(
        pre_events_df,
        post_events_df,
        output_path=cfg.figures_dir / "fig06_event_boxplots.png"
    )
    
    # Monthly boxplots
    df_for_monthly = df.copy()
    if fire_dt:
        df_for_monthly['period'] = 'Pre-fire'
        df_for_monthly.loc[df_for_monthly.index >= fire_dt, 'period'] = 'Post-fire'
    else:
        df_for_monthly['period'] = 'Full Record'
    plot_monthly_boxplots(df_for_monthly, output_path=cfg.figures_dir / "fig07_monthly_boxplots.png")

    # Baseflow plot
    if not annual_bfi.empty:
        plot_baseflow_index(annual_bfi, fire_years=significant_fire_years, output_path=cfg.figures_dir / "fig08_bfi.png")

    # Map
    map_catchment(gdf, fire_history_file=cfg.fire_history_file, output_path=cfg.figures_dir / "fig09_catchment_map.png")

    # 6. Reporting
    click.echo("📝 Generating report...")
    results = {
        'fdc_comparison': fdc_comp,
        'event_stats': event_stats,
        'flood_quantiles': flood_quantiles_comp.reset_index(),
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
