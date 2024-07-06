from pathlib import Path

ROOT = Path(__file__).parent.parent
PHOTOZ_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/idr5_photoz_clean.parquet')
PHOTOZ2_TABLE_PATH = Path('tables/idr5_v3.parquet')
SPEC_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/SpecZ_Catalogue_20240124.parquet')
ERASS_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/liana_erass.csv')
FULL_ERASS_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/eRASS1_min.parquet')
ERASS2_TABLE_PATH = ROOT / 'tables/Kluge_Bulbul_joint_selected_clusters_zlt0.2.csv'
HEASARC_TABLE_PATH = ROOT / 'public/heasarc_all.parquet'
TABLES_PATH = ROOT / 'clusters_members'
MEMBERS_FOLDER = ROOT / 'clusters_members/clusters'
OUT_PATH = ROOT / 'outputs_v6'
WEBSITE_PATH = ROOT / 'docs'
PLOTS_FOLDER = OUT_PATH / 'plots'
VELOCITY_PLOTS_FOLDER = OUT_PATH / 'velocity_plots'
MAGDIFF_PLOTS_FOLDER = OUT_PATH / 'magdiff_plots'
XRAY_PLOTS_FOLDER = OUT_PATH / 'xray_plots'
MAGDIFF_OUTLIERS_FOLDER = OUT_PATH / 'magdiff_outliers'
LEG_PHOTO_FOLDER = OUT_PATH / 'legacy'
PHOTOZ_FOLDER = OUT_PATH / 'photoz'
SPECZ_FOLDER = OUT_PATH / 'specz'
PHOTOZ_SPECZ_LEG_FOLDER = OUT_PATH / 'photoz+specz+legacy'
MAG_COMP_FOLDER = OUT_PATH / 'mag_comp'
Z_PHOTO_DELTA = 0.015
Z_SPEC_DELTA = 0.007
Z_SPEC_DELTA_PAULO = 0.02
MAG_RANGE = (13, 22)