import glob, os
import numpy as np
import pandas as pd
import matplotlib as mpl
from pathlib import Path
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

""" All utils for analysis
"""

# Models used
dic_sntype = {'50':'Kasen','51':'Bulla'}

# Plotting
dic_band_colors = {
    'LSST-u': '#6A1B9A',   # slightly darker purple, distinct from blue & teal
    'LSST-g': '#3C8DFF',   # light blue (unchanged)
    'LSST-r': '#15284F',   # dark blue/navy (unchanged)
    'LSST-i': '#E69F00',   # orange (unchanged)
    'LSST-z': '#F5622E',   # reddish-orange (unchanged)
    'LSST-Y': '#FFD600'    # golden yellow, cooler than bright yellow, still high contrast
}

dic_KN_models_colors = {
    "Bulla": '#15284F',
    "Kasen": '#f5622e',
}

dic_limiting_mags = {
    'LSST-u': 23.9,
    'LSST-g': 25.0,
    'LSST-r': 24.7,
    'LSST-i': 24.0,
    'LSST-z': 23.3,
    'LSST-Y': 22.1
}


mpl.rcParams["font.size"] = 14            # default text (tick labels, legend, etc.)
mpl.rcParams["axes.titlesize"] = 14       # title font size
mpl.rcParams["axes.labelsize"] = 14       # x and y labels font size
mpl.rcParams["legend.fontsize"] = 12      # legend
mpl.rcParams["xtick.labelsize"] = 12      # x tick labels
mpl.rcParams["ytick.labelsize"] = 12      # y tick labels
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["figure.titlesize"] = "large"
mpl.rcParams["lines.linewidth"] = 3
mpl.rcParams["patch.linewidth"] = 3 

# Rubin DDFs
data = {
    'Field': ['ELAISS1', 'XMM_LSS', 'ECDFS', 'COSMOS', 'EDFS_a', 'EDFS_b'],
    'RA': [9.45, 35.57, 52.98, 150.11, 58.9, 63.6],
    'DEC': [-44.02, -4.82, -28.12, 2.23, -49.32, -47.6],
    'Gal l': [311.29, 171.1, 224.07, 236.78, 257.9, 254.48],
    'Gal b': [-72.88, -58.91, -54.6, 42.13, -48.46, -45.77],
    'Eclip l': [346.66, 31.59, 40.81, 151.39, 32, 40.97],
    'Eclip b': [-43.2, -17.92, -45.44, -9.34, -66.61, -66.6]
}
ddf_fields = pd.DataFrame(data)


def convert_fluxcal_to_mag(df):
    """_summary_

    Args:
        df (_type_): _description_
    """
    # Convert fluxcal to magnitude
    df = df.copy()
    with np.errstate(invalid='ignore', divide='ignore'):
        df["magnitude"] = np.where(
            df["FLUXCAL"] > 0,
            -2.5 * np.log10(df["FLUXCAL"]) + 27.5,
            np.nan
        )
        df["magnitude error"] = np.where(
            df["FLUXCAL"] > 0,
            2.5 / np.log(10) * df["FLUXCALERR"] / df["FLUXCAL"],
            np.nan
        )
    
    return df

def read_fits(fname, drop_separators=False):
    """Load SNANA formatted data and cast it to a PANDAS dataframe

    Args:
        fname (str): path + name to PHOT.FITS file
        drop_separators (Boolean): if -777 are to be dropped

    Returns:
        (pandas.DataFrame) dataframe from PHOT.FITS file (with ID)
        (pandas.DataFrame) dataframe from HEAD.FITS file
    """

    # load photometry
    dat = Table.read(fname, format="fits")
    df_phot = dat.to_pandas()
    # failsafe
    if df_phot.MJD.values[-1] == -777.0:
        df_phot = df_phot.drop(df_phot.index[-1])
    if df_phot.MJD.values[0] == -777.0:
        df_phot = df_phot.drop(df_phot.index[0])

    # load header
    header = Table.read(fname.replace("PHOT", "HEAD"), format="fits")
    df_header = header.to_pandas()
    # fix to make SNID unique
    
    # Decode bytes and strip whitespace
    df_header["SNID"] = df_header["SNID"].str.decode("utf-8").str.strip()
    # Map SNTYPE and concatenate
    df_header['model_name'] = df_header["SNTYPE"].astype(str).map(dic_sntype)
    suffix = df_header['model_name'][0]
    # to disentangle the SNIDs from the two models (issue with sim, not unique)
    df_header["SNID"] = df_header["SNID"] + f"_{suffix}"


    # add SNID to phot for skimming
    arr_ID = np.empty(len(df_phot), dtype=object)  # allocate as string
    arr_idx = np.where(df_phot["MJD"].values == -777.0)[0]
    arr_idx = np.hstack((np.array([0]), arr_idx, np.array([len(df_phot)])))
    for counter in range(1, len(arr_idx)):
        start, end = arr_idx[counter - 1], arr_idx[counter]
        snid_str = str(df_header.SNID.iloc[counter - 1])
        arr_ID[start:end] = snid_str
    df_phot["SNID"] = arr_ID

    if drop_separators:
        df_phot = df_phot[df_phot.MJD != -777.000]

    df_phot['SNR'] = np.abs(df_phot['FLUXCAL'] / df_phot['FLUXCALERR'])
    df_phot = convert_fluxcal_to_mag(df_phot)
    df_phot["BAND"] = df_phot["BAND"].str.decode('utf-8').str.strip()

    # Add model name (not optimised)
    df_phot["model_name"] = df_phot["SNID"].str.split("_").str[1]

    # Add DDF
    # Initialize new columns
    df_header["in_ddf_field"] = False
    df_header["ddf_field_name"] = np.nan
    df_header["ddf_field_name"] = df_header["ddf_field_name"].astype("object")  # Ensure dtype is object

    # Convert df_header RA/DEC to SkyCoord
    coords_header = SkyCoord(ra=df_header["RA"].values * u.deg,
                            dec=df_header["DEC"].values * u.deg)
    # Radius of 9.6 deg² -> ~1.75 deg
    radius_deg = np.sqrt(9.6 / np.pi)
    # Loop through DDF fields
    for _, row in ddf_fields.iterrows():
        field = row["Field"]
        ra = row["RA"]
        dec = row["DEC"]
        
        # Convert field center to SkyCoord
        field_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        
        # Compute angular separation
        separation = coords_header.separation(field_coord)
        
        # Find objects within radius
        within = separation <= radius_deg * u.deg
        
        df_header.loc[within, "in_ddf_field"] = True
        df_header.loc[within, "ddf_field_name"] = field

    return df_header, df_phot

def get_snr_sampling(df_header, df_phot,snr_threshold=3):
    dic_snr = {}
    for typ in df_header.SIM_GENTYPE.unique():
        
        # Select matching SNIDs from df_phot
        sel_snids = df_header[df_header.SIM_GENTYPE == typ].SNID
        print(f"\nNumber of light curves with SIM_GENTYPE = {typ} | {dic_sntype[str(typ)]}: {len(sel_snids)}")

        # Total SNIDs for this SIM_GENTYPE
        total_typ_snids = len(sel_snids)

        # Photometry
        sel = df_phot[df_phot.SNID.isin(sel_snids)]

        # Compute SNR thresholds
        typ_str = dic_sntype[f"{typ}"]
        dic_snr[typ_str] = sel[sel['SNR'] > snr_threshold]
        snr_gt_3 = sel[sel['SNR'] > snr_threshold]['SNID'].nunique()

        # Group by SNID, count how many SNR > 5 measurements each has
        snr_gt3_counts = dic_snr[typ_str].groupby('SNID').size()

        # Count how many SNIDs have at least two such measurements
        snr_gt3_atleast2 = (snr_gt3_counts >= 2).sum()

        # Group by SNID and check MJD separation
        def has_two_sep_days(group):
            mjds = group['MJD'].sort_values().values
            for i in range(len(mjds)):
                for j in range(i+1, len(mjds)):
                    if abs(mjds[j] - mjds[i]) > 1:
                        return True
            return False

        # Apply the function to each group
        snids_with_two_sep_days = dic_snr[typ_str].groupby('SNID').filter(has_two_sep_days)['SNID'].nunique()
        

        print(f"{snr_gt_3} light curves have 1 detection with SNR > {snr_threshold} = {snr_gt_3 / total_typ_snids * 100:.2f}% of this type")
        print(f"{snr_gt3_atleast2} light curves have 2 detections with SNR > {snr_threshold} = {snr_gt3_atleast2 / total_typ_snids * 100:.2f}% of this type")
        print(f"{snids_with_two_sep_days} light curves have 2 detections with SNR > {snr_threshold} in >1 day = {snids_with_two_sep_days / total_typ_snids * 100:.2f}% of this type")

    return dic_snr


def get_snr_sampling_table(df_header, df_phot, snr_threshold=3):
    """
    Compute percentages of light curves with detections above a given SNR threshold.
    Returns a DataFrame with SIM_GENTYPE averages and per-model percentages.
    """

    records = []

    for typ in df_header.SIM_GENTYPE.unique():
        # Select matching SNIDs
        sel_snids = df_header[df_header.SIM_GENTYPE == typ].SNID
        total_typ_snids = len(sel_snids)

        # Photometry for this type
        sel = df_phot[df_phot.SNID.isin(sel_snids)]
        typ_str = dic_sntype[str(typ)]

        # Light curves with at least 1 detection
        snr_sel = sel[sel['SNR'] > snr_threshold]
        one_det = snr_sel['SNID'].nunique()

        # Light curves with at least 2 detections
        snr_counts = snr_sel.groupby('SNID').size()
        two_det = (snr_counts >= 2).sum()

        # Light curves with 2 detections separated by >1 day
        def has_two_sep_days(group):
            mjds = group['MJD'].sort_values().values
            for i in range(len(mjds)):
                for j in range(i+1, len(mjds)):
                    if abs(mjds[j] - mjds[i]) > 1:
                        return True
            return False

        two_sep_days = snr_sel.groupby('SNID').filter(has_two_sep_days)['SNID'].nunique()

        # Store percentages
        records.append({
            "SNR": snr_threshold,
            "Model": typ_str,
            "1 detection (%)": one_det / total_typ_snids * 100,
            "2 detections (%)": two_det / total_typ_snids * 100,
            "2 detections >1 day (%)": two_sep_days / total_typ_snids * 100
        })

    # Convert to DataFrame
    df_percent = pd.DataFrame(records)

    # Compute average between models for this SNR
    avg_row = {
        "SNR": snr_threshold,
        "Model": "Average",
        "1 detection (%)": df_percent["1 detection (%)"].mean(),
        "2 detections (%)": df_percent["2 detections (%)"].mean(),
        "2 detections >1 day (%)": df_percent["2 detections >1 day (%)"].mean()
    }

    df_percent = pd.concat([df_percent, pd.DataFrame([avg_row])], ignore_index=True)

    return df_percent



def plot_lc_subplots(df_phot, df_header, idx, inmag=False, convert_date=False):
    """Plot multi-panel light curve evolution for a given SNID."""
    
    # --- Select data ---
    sel_phot = df_phot[df_phot.SNID == idx].copy()
    sel_header = df_header[df_header.SNID == idx]

    # Remove bogus mags if plotting in mag space
    if inmag:
        sel_phot = sel_phot[sel_phot["magnitude"] > 0]

    # Init
    bands = sel_phot["BAND"].unique()
    x_col = "TIME" if convert_date else "MJD"
    if convert_date:
        sel_phot["TIME"] = pd.to_datetime(sel_phot["MJD"], unit='D', origin='1858-11-17')

    # Peak time for vertical line
    peakmjd = float(sel_header.PEAKMJD.values[0])
    peak_x = pd.to_datetime(peakmjd, unit='D', origin='1858-11-17') if convert_date else peakmjd

    # Select SNR>5 detections sorted by time
    snr5_df = sel_phot[sel_phot['SNR'] > 5].sort_values('MJD')

    # No alerts
    if len(snr5_df)<1:
        print(f"No alert for {idx}")
        return None
    

    # --- Build datasets to plot
    # First alert
    first_alert = snr5_df.head(1)
    # Second alert
    # with history if Delta time> 1 day (0.6 for a night)
    second_alert = pd.DataFrame()
    if len(snr5_df) >= 2:
        if snr5_df['MJD'].iloc[1] - snr5_df['MJD'].iloc[0] > 0.6:
            second_mjd = snr5_df.iloc[1]['MJD']
            snr_lt5_before_second = sel_phot[(sel_phot['SNR'] < 5) & (sel_phot['MJD'] < second_mjd)]
            second_alert = pd.concat([snr5_df.head(2), snr_lt5_before_second])
        else:
            second_alert = snr5_df.head(2)
    # All the photometry received by the broker
    all_alerts = snr5_df
    last_alert_mjd = snr5_df['MJD'].max()
    if snr5_df['MJD'].max() - snr5_df['MJD'].min()>0.6:
        all_alerts = sel_phot[sel_phot['MJD'] < last_alert_mjd]
        
    full_Rubin_lc = sel_phot.copy()

    datasets = [first_alert, second_alert, all_alerts, full_Rubin_lc]

    # --- Global y-range ---
    if inmag:
        all_y = sel_phot["magnitude"]
        ymin, ymax = all_y.max() + 1, all_y.min() - 1  # reversed for mags
    else:
        all_y = sel_phot["FLUXCAL"]
        ymin, ymax = all_y.min() - 1, all_y.max() + 1

    # --- Subplot titles ---
    base_titles = [
        "1. First alert",
        "2. Second alert",
        "3. All broker data",
        "4. Full Rubin data"
    ]
    subplot_titles = []
    for base, df in zip(base_titles, datasets):
        if df.empty:
            latest = "No data"
        else:
            latest_mjd = df['MJD'].max()
            latest = pd.to_datetime(latest_mjd, unit='D', origin='1858-11-17').strftime('%Y-%m-%d') if convert_date else f"MJD {latest_mjd:.1f}"
        subplot_titles.append(f"{base}<br>{latest}")

    # --- Create figure ---
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=subplot_titles,
        shared_yaxes=True
    )

    plotted_labels = set()

    # --- Plot each dataset ---
    for i, dataset in enumerate(datasets):
        if dataset.empty:
            continue
        for flt in bands:
            sel_flt = dataset[dataset["BAND"] == flt]
            yvar, yerr = ("magnitude", "magnitude error") if inmag else ("FLUXCAL", "FLUXCALERR")

            # Plot SNR bins
            for snr_min, snr_max, symbol, opacity, suffix in [
                (5, np.inf, 'circle', 1, "SNR > 5"),
                (1, 5, 'circle', 0.3, "1 < SNR < 5")
            ]:
                mask = (sel_flt['SNR'] > snr_min) & (sel_flt['SNR'] < snr_max)
                label = f"{flt} ({suffix})"
                fig.add_trace(go.Scatter(
                    x=sel_flt.loc[mask, x_col],
                    y=sel_flt.loc[mask, yvar],
                    error_y=dict(
                        type='data',
                        array=sel_flt.loc[mask, yerr],
                        visible=True
                    ),
                    mode='markers',
                    name=label,
                    marker=dict(opacity=opacity, symbol=symbol, size=8, color=dic_band_colors[flt]),
                    showlegend=False, #(label not in plotted_labels)
                ), row=1, col=i+1)
                plotted_labels.add(label)

            # SNR < 1
            mask = sel_flt['SNR'] < 1
            fig.add_trace(go.Scatter(
                x=sel_flt.loc[mask, x_col],
                y=sel_flt.loc[mask, yvar],
                mode='markers',
                marker=dict(opacity=0.2, symbol='triangle-down', size=8, color=dic_band_colors[flt]),
                showlegend=False
            ), row=1, col=i+1)

        # Peak & first detection lines
        fig.add_vline(x=peak_x, line=dict(color="black", width=1, dash="dash"), row=1, col=i+1,layer="below")
        if not dataset[dataset['SNR'] > 5].empty:
            first_snr5_mjd = dataset.loc[dataset['SNR'] > 5, 'MJD'].min()
            first_snr5_x = pd.to_datetime(first_snr5_mjd, unit='D', origin='1858-11-17') if convert_date else first_snr5_mjd
            fig.add_vline(x=first_snr5_x, line=dict(color="blue", width=1.5, dash="dot"), row=1, col=i+1,layer="below")
            last_snr5_mjd = dataset.loc[dataset['SNR'] > 5, 'MJD'].max()
            last_snr5_x = pd.to_datetime(last_snr5_mjd, unit='D', origin='1858-11-17') if convert_date else last_snr5_mjd
            fig.add_vline(x=last_snr5_x, line=dict(color="blue", width=1.5, dash="dot"), row=1, col=i+1,layer="below")

    # Legend-only dummy traces
    for band, color in dic_band_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=color, size=30),
            name=band  # appears in legend
        ))

    # --- Layout ---
    z = np.round(float(sel_header.REDSHIFT_FINAL.values[0]), 2)
    sim_type = sel_header.SIM_TYPE_NAME.values[0]
    if isinstance(sim_type, bytes):
        sim_type = sim_type.decode("utf-8")
    ddf_name = sel_header.ddf_field_name.values[0] if sel_header.in_ddf_field.values[0] else "WFD"

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=30)
                              
    print(f"SNID: {idx} z: {z} {ddf_name}")
    fig.update_layout(
        # title_text=f"SNID: {idx} z: {z} {ddf_name}",
        height=500, width=1500,
        legend=dict(font=dict(size=16)),
        legend_title="Filter",
    )
    fig.update_xaxes(tickangle=30)
    fig.update_yaxes(title_text=("magnitude" if inmag else "FLUXCAL"), range=[ymin, ymax],showgrid=False,title_font=dict(size=30),tickfont=dict(size=16))
    fig.update_xaxes(title_text=("Date" if convert_date else "MJD"),showgrid=False, title_font=dict(size=30),tickfont=dict(size=16))
    fig.show()

def plot_lc_subplots_matplotlib(df_phot, df_header, idx, inmag=False, convert_date=False, savefig=False, path_plots='.'):
    """Matplotlib version of plot_lc_subplots"""

    sel_phot = df_phot[df_phot.SNID == idx].copy()
    sel_header = df_header[df_header.SNID == idx]

    if inmag:
        sel_phot = sel_phot[sel_phot["magnitude"] > 0]

    bands = sel_phot["BAND"].unique()
    x_col = "TIME" if convert_date else "MJD"
    if convert_date:
        sel_phot["TIME"] = pd.to_datetime(sel_phot["MJD"], unit="D", origin="1858-11-17")

    peakmjd = float(sel_header.PEAKMJD.values[0])
    peak_x = pd.to_datetime(peakmjd, unit="D", origin="1858-11-17") if convert_date else peakmjd

    snr5_df = sel_phot[sel_phot['SNR'] > 5].sort_values('MJD')
    if len(snr5_df) < 1:
        print(f"No alert for {idx}")
        return None

    # First alert
    first_alert = snr5_df.head(1)

    # Second alert
    second_alert = pd.DataFrame()
    if len(snr5_df) >= 2:
        if snr5_df['MJD'].iloc[1] - snr5_df['MJD'].iloc[0] > 0.6:
            second_mjd = snr5_df.iloc[1]['MJD']
            snr_lt5_before_second = sel_phot[(sel_phot['SNR'] < 5) & (sel_phot['MJD'] < second_mjd)]
            second_alert = pd.concat([snr5_df.head(2), snr_lt5_before_second])
        else:
            second_alert = snr5_df.head(2)

    # Data range second alert
    if not second_alert.empty:
        delta = second_alert[x_col].max() - second_alert[x_col].min()
        second_alert_range = (second_alert[x_col].min()-delta, second_alert[x_col].max()+delta)
    else:
        delta = first_alert[x_col].max() - first_alert[x_col].min()
        second_alert_range = (first_alert[x_col].min()-delta, first_alert[x_col].max()+delta)

    # All broker alerts
    all_alerts = snr5_df
    last_alert_mjd = snr5_df['MJD'].max()
    if snr5_df['MJD'].max() - snr5_df['MJD'].min() > 0.6:
        all_alerts = sel_phot[sel_phot['MJD'] < last_alert_mjd]

    full_Rubin_lc = sel_phot.copy()
    datasets = [first_alert, second_alert, all_alerts, full_Rubin_lc]

    # Global y-range
    if inmag:
        all_y = sel_phot["magnitude"]
        ymin, ymax = all_y.max() + 1, all_y.min() - 1
    else:
        all_y = sel_phot["FLUXCAL"]
        ymin, ymax = all_y.min() - 1, all_y.max() + 1

    # Subplot titles
    base_titles = [
        "1. First alert",
        "2. Second alert",
        "3. All broker data",
        "4. Full Rubin data"
    ]
    subplot_titles = []
    for base, df in zip(base_titles, datasets):
        if df.empty:
            latest = "No data"
        else:
            latest_mjd = df['MJD'].max()
            latest = pd.to_datetime(latest_mjd, unit='D', origin='1858-11-17').strftime('%Y-%m-%d') if convert_date else f"MJD {latest_mjd:.1f}"
        subplot_titles.append(f"{base}\n{latest}")

    # --- Matplotlib figure ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)

    for i, (ax, dataset, title) in enumerate(zip(axes, datasets, subplot_titles)):
        if dataset.empty:
            ax.set_title(title, fontsize=12)
            continue

        for flt in bands:
            sel_flt = dataset[dataset["BAND"] == flt]
            yvar, yerr = ("magnitude", "magnitude error") if inmag else ("FLUXCAL", "FLUXCALERR")

            # SNR bins
            for snr_min, snr_max, marker, alpha, suffix in [
                (5, np.inf, "o", 1, "SNR > 5"),
                (1, 5, "o", 0.3, "1 < SNR < 5")
            ]:
                mask = (sel_flt['SNR'] > snr_min) & (sel_flt['SNR'] < snr_max)
                if len(sel_flt[mask]) > 0:
                    ax.errorbar(
                        sel_flt.loc[mask, x_col],
                        sel_flt.loc[mask, yvar],
                        yerr=sel_flt.loc[mask, yerr],
                        fmt=marker,
                        ms=6,
                        alpha=alpha,
                        color=dic_band_colors[flt],
                        label=f"{flt}" if i == 0 or i==3 else None
                    )

            # SNR < 1
            mask = sel_flt['SNR'] < 1
            if len(sel_flt[mask]) > 0:
                ax.scatter(
                    sel_flt.loc[mask, x_col],
                    sel_flt.loc[mask, yvar],
                    marker="v",
                    alpha=0.2,
                    color=dic_band_colors[flt],
                    label=f"{flt} (SNR < 1)" if i == 0 else None
                )

        # Vertical lines: peak + first/last SNR>5
        ax.axvline(peak_x, color="black", lw=1.5, ls="--", zorder=0)
        if not dataset[dataset['SNR'] > 5].empty and i>0:
            first_snr5 = dataset.loc[dataset['SNR'] > 5, x_col].min()
            last_snr5 = dataset.loc[dataset['SNR'] > 5, x_col].max()
            # ax.axvline(first_snr5, color="blue", lw=1.5, ls=":", zorder=0)
            # ax.axvline(last_snr5, color="blue", lw=1.5, ls=":", zorder=0)
            ax.axvspan(first_snr5, last_snr5, color="#D5D5D3", alpha=0.8, zorder=-1)

        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", rotation=30)

    # Y-axis & labels
    axes[0].set_ylabel("magnitude" if inmag else "FLUXCAL")
    for i, ax in enumerate(axes):
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Date" if convert_date else "MJD",fontdict={'fontsize': 14})
        if i<2:
            ax.set_xlim(second_alert_range)

    import matplotlib.patches as mpatches

    # Create one legend entry per band
    handles = []
    labels = []
    for band, color in dic_band_colors.items():
        patch = mpatches.Patch(color=color, label=band)  # patch works, but for circle marker we can use Line2D
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        labels.append(band)

    # Place the legend outside, centered vertically
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        fontsize=10,
        title="Filter"
    )

    print(f"SNID {idx} z={sel_header.REDSHIFT_FINAL.values[0]:.2f}")
    

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05) 

    if savefig:
        plt.savefig(f"{path_plots}/lc_subplot_{idx}_z_{sel_header.REDSHIFT_FINAL.values[0]:.2f}_{sel_header.in_ddf_field.values[0]}.png")


def create_filtering_statistics(df_original, n_alerts_year, filtering_stages):
    """
    Create a comprehensive statistics DataFrame for all filtering stages
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Original dataframe before any cuts
    n_alerts_year : int
        Total number of alerts for the year
    filtering_stages : list of tuples
        Each tuple contains (stage_name, dataframe_after_cuts, description)
        
    Returns:
    --------
    pd.DataFrame : Statistics table with all filtering information
    """
    
    stats_data = []
    
    # Helper function to get top 3 most common finkclasses with percentages
    def get_top3_finkclasses(df):
        if len(df) == 0:
            return "No data"
        
        finkclass_counts = df['finkclass'].value_counts()
        top3 = finkclass_counts.head(3)
        
        result_parts = []
        for finkclass, count in top3.items():
            percentage = (count / len(df)) * 100
            result_parts.append(f"{finkclass} ({percentage:.1f}%)")
        
        return "; ".join(result_parts)
    
    # Add original data as first row
    top3_orig = get_top3_finkclasses(df_original)
    stats_data.append({
        'Stage': 'Original Data',
        'Description': 'All alerts from Fink broker',
        'N_Candidates': n_alerts_year, # the Fink transfer data was already pre-filtered
        'Candidates_per_Month': n_alerts_year / 12,
        'Efficiency_vs_Previous': 100.0,
        'Efficiency_vs_Original': 100.0,
        'Percentage_of_Yearly': n_alerts_year / n_alerts_year * 100,
        'Top3_Finkclasses': top3_orig
    })
    
    previous_count = n_alerts_year
    original_count = n_alerts_year
    
    # Process each filtering stage
    for stage_name, df_stage, description in filtering_stages:
        current_count = len(df_stage)
        top3_classes = get_top3_finkclasses(df_stage)
        
        stats_data.append({
            'Stage': stage_name,
            'Description': description,
            'N_Candidates': current_count,
            'Candidates_per_Month': current_count / 12,
            'Efficiency_vs_Previous': (current_count / previous_count * 100) if previous_count > 0 else 0,
            'Efficiency_vs_Original': (current_count / original_count * 100) if original_count > 0 else 0,
            'Percentage_of_Yearly': current_count / n_alerts_year * 100,
            'Top3_Finkclasses': top3_classes
        })
        
        previous_count = current_count
    
    return pd.DataFrame(stats_data)



