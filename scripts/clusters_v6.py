import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from shutil import make_archive, rmtree

from astromodule.io import merge_pdf, read_table, write_table
from astromodule.pipeline import Pipeline, PipelineStorage
from astromodule.table import concat_tables, guess_coords_columns
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table

from splusclusters.configs import configs
from splusclusters.external import (ArchiveDownloadLegacyCatalogStage,
                                    DownloadLegacyCatalogStage,
                                    DownloadSplusPhotozStage)
from splusclusters.loaders import (LoadAllRadialStage, LoadClusterInfoStage,
                                   LoadGenericInfoStage, LoadLegacyRadialStage,
                                   LoadPauloInfoStage, LoadPhotozRadialStage,
                                   LoadSpeczRadialStage,
                                   PrepareCatalogToSubmitStage,
                                   load_catalog_v6, load_catalog_v6_hydra,
                                   load_catalog_v6_old, load_photoz2,
                                   load_spec, load_xray)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage
from splusclusters.utils import config_dask

OVERWRITE_PHOTOZ_SPECZ_LEGACY_MATCH = False


def add_xray_flag(df: pd.DataFrame, threshold: float = 1):
  df_xray = load_xray()
  ra_col, dec_col = guess_coords_columns(df_xray)
  xray_coords = SkyCoord(ra=df_xray[ra_col], dec=df_xray[dec_col], unit=u.deg)
  ra_col, dec_col = guess_coords_columns(df)
  df_coords = SkyCoord(ra=df[ra_col], dec=df[dec_col], unit=u.deg)
  _, sep, _ = match_coordinates_sky(df_coords, xray_coords)
  mask = sep > (threshold * u.arcsec)
  df['xray-flag'] = mask.astype(int)
  return df


def _log_clusters(df_clusters):
  data = {
    'cluster': [], 'total': [], 'z_min': [], 'z_max': [], 'z_null': [],
    'z_neg': [], 'z_pos': [], 'zerr_min': [], 'zerr_max': [], 'zerr_null': [],
    'zerr_neg': [], 'zerr_pos': [], 'z_flag': [],
  }
  for _, cluster in df_clusters.iterrows():
    cls_id = cluster['clsid']
    cls_name: str = cluster['name']
    clusters_path = configs.SUBMIT_FOLDER / 'clusters'
    table_path = clusters_path / f'cluster_{str(cls_id).zfill(4)}.dat'
    df = Table.read(table_path, format='ascii').to_pandas()
    flag_count = ''.join([f'{k} ({v})' for k, v in df['zspec-flag'].value_counts(dropna=False).items()])
    data['cluster'] += [cls_name]
    data['total'] += [len(df)]
    data['z_min'] += [df.zspec.min()]
    data['z_max'] += [df.zspec.max()]
    data['z_null'] += [len(df[df.zspec.isna()])]
    data['z_neg'] += [len(df[df.zspec < 0])]
    data['z_pos'] += [len(df[df.zspec > 0])]
    data['zerr_min'] += [df["zspec-err"].min()]
    data['zerr_max'] += [df["zspec-err"].max()]
    data['zerr_null'] += [len(df[df["zspec-err"].isna()])]
    data['zerr_neg'] += [len(df[df["zspec-err"] < 0])]
    data['zerr_pos'] += [len(df[df["zspec-err"] > 0])]
    data['z_flag'] += [flag_count]
  
  path = configs.SUBMIT_FOLDER / f'summary.csv'
  df = pd.DataFrame(data)
  if path.exists():
    df = concat_tables([read_table(path), df])
  write_table(df, path)
  print(df)
  


def clusters_v5_remake_pipeline(clear: bool = False):
  df_clusters = load_catalog_v6_old()
  # df_photoz, photoz_skycoord = load_photoz2()
  df_spec, specz_skycoord = load_spec()
  df_spec.rename(columns={'RA': 'ra_spec_all', 'DEC': 'dec_spec_all'}, inplace=True)
  
  configs.Z_SPEC_DELTA = 0.02
  configs.SUBMIT_FOLDER = configs.OUT_PATH / 'submit' / 'antigos'
  # rmtree(configs.SUBMIT_FOLDER, ignore_errors=True)
  configs.SUBMIT_FOLDER.mkdir(exist_ok=True, parents=True)
  
  if clear:
    for p in configs.LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
  
  pipe = Pipeline(
    LoadPauloInfoStage(df_clusters),
    DownloadSplusPhotozStage(overwrite=False, workers=5),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    DownloadLegacyCatalogStage('cls_search_radius_deg', overwrite=False, workers=6),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=OVERWRITE_PHOTOZ_SPECZ_LEGACY_MATCH),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=False, splus_only=False),
    PrepareCatalogToSubmitStage(overwrite=True),
  )
  
  # PipelineStorage().write('df_photoz', df_photoz)
  # PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  plot_paths = [configs.PLOTS_FOLDER / f'cls_{c}.pdf' for c in df_clusters.name.values]
  plot_paths = [p for p in plot_paths if p.exists()]
  concat_plot_path = configs.PLOTS_FOLDER / 'clusters_v6+antigos.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  
  df_clusters['clsid'] = df_clusters.clsid.astype(str).str.zfill(4)
  
  df_clusters = df_clusters.rename(columns={'ra': 'RA', 'dec': 'DEC', 'z_spec': 'zspec'})
  add_xray_flag(df_clusters)
  # write_table(
  #   df_clusters[['clsid', 'name', 'RA', 'DEC', 'zspec', 'xray-flag']], 
  #   configs.SUBMIT_FOLDER / 'index.dat'
  # )
  Table.from_pandas(
    df_clusters[['clsid', 'name', 'RA', 'DEC', 'zspec', 'xray-flag']]
  ).write(configs.SUBMIT_FOLDER / 'index.dat', format='ascii', overwrite=True)
  
  _log_clusters(df_clusters)





def hydra_neighbours_pipeline(clear: bool = False):
  df_clusters = load_catalog_v6_hydra()
  # df_photoz, photoz_skycoord = load_photoz2()
  df_spec, specz_skycoord = load_spec()
  df_spec.rename(columns={'RA': 'ra_spec_all', 'DEC': 'dec_spec_all'}, inplace=True)
  
  configs.Z_SPEC_DELTA = 0.02
  configs.SUBMIT_FOLDER = configs.OUT_PATH / 'submit' / 'hydra'
  # rmtree(configs.SUBMIT_FOLDER, ignore_errors=True)
  configs.SUBMIT_FOLDER.mkdir(exist_ok=True, parents=True)
  
  if clear:
    for p in configs.LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
  
  pipe = Pipeline(
    LoadGenericInfoStage(df_clusters),
    DownloadSplusPhotozStage(overwrite=False, workers=5),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    DownloadLegacyCatalogStage('cls_search_radius_deg', overwrite=False, workers=6),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=OVERWRITE_PHOTOZ_SPECZ_LEGACY_MATCH),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=False, splus_only=False),
    PrepareCatalogToSubmitStage(overwrite=True),
  )
  
  # PipelineStorage().write('df_photoz', df_photoz)
  # PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.name.values, workers=1)
  
  plot_paths = [configs.PLOTS_FOLDER / f'cls_{c}.pdf' for c in df_clusters.name.values]
  plot_paths = [p for p in plot_paths if p.exists()]
  concat_plot_path = configs.PLOTS_FOLDER / 'clusters_v6+hydra.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  
  df_clusters['clsid'] = df_clusters.clsid.astype(str).str.zfill(4)
  
  df_clusters = df_clusters.rename(columns={'z_spec': 'zspec', 'ra': 'RA', 'dec': 'DEC'})
  add_xray_flag(df_clusters)
  # write_table(
  #   df_clusters[['clsid', 'name', 'RA', 'DEC', 'zspec', 'xray-flag']], 
  #   configs.SUBMIT_FOLDER / 'index.dat'
  # )
  Table.from_pandas(
    df_clusters[['clsid', 'name', 'RA', 'DEC', 'zspec', 'xray-flag']]
  ).write(configs.SUBMIT_FOLDER / 'index.dat', format='ascii', overwrite=True)

  _log_clusters(df_clusters)



def clusters_v6_pipeline(clear: bool = False):
  df_clusters = load_catalog_v6()
  # df_photoz, photoz_skycoord = load_photoz2()
  df_spec, specz_skycoord = load_spec()
  df_spec.rename(columns={'RA': 'ra_spec_all', 'DEC': 'dec_spec_all'}, inplace=True)
  
  configs.Z_SPEC_DELTA = 0.02
  configs.SUBMIT_FOLDER = configs.OUT_PATH / 'submit' / 'novos'
  # rmtree(configs.SUBMIT_FOLDER, ignore_errors=True)
  configs.SUBMIT_FOLDER.mkdir(exist_ok=True, parents=True)
  
  if clear:
    for p in configs.LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
        
  ls10_pipe = Pipeline(
    LoadPauloInfoStage(df_clusters),
    DownloadSplusPhotozStage(overwrite=False, workers=5),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    DownloadLegacyCatalogStage('cls_search_radius_deg', overwrite=False, workers=6),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=OVERWRITE_PHOTOZ_SPECZ_LEGACY_MATCH),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=False, splus_only=False),
    PrepareCatalogToSubmitStage(overwrite=True),
  )
  
  # PipelineStorage().write('df_photoz', df_photoz)
  # PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  ls10_pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  plot_paths = [configs.PLOTS_FOLDER / f'cls_{c}.pdf' for c in df_clusters.name.values]
  plot_paths = [p for p in plot_paths if p.exists()]
  concat_plot_path = configs.PLOTS_FOLDER / 'clusters_v6+novos.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  
  df_clusters['clsid'] = df_clusters.clsid.astype(str).str.zfill(4)
  
  df_clusters = df_clusters.rename(columns={'z_spec': 'zspec', 'ra': 'RA', 'dec': 'DEC'})
  add_xray_flag(df_clusters)
  # write_table(
  #   df_clusters[['clsid', 'name', 'RA', 'DEC', 'zspec', 'xray-flag']], 
  #   configs.SUBMIT_FOLDER / 'index.dat'
  # )
  Table.from_pandas(
    df_clusters[['clsid', 'name', 'RA', 'DEC', 'zspec', 'xray-flag']]
  ).write(configs.SUBMIT_FOLDER / 'index.dat', format='ascii', overwrite=True)
  
  _log_clusters(df_clusters)
  
    

  


def create_zip():
  path = configs.OUT_PATH / 'submit' / 'clusters_v6.zip'
  if path.exists():
    path.unlink()

  make_archive(
    base_name=str(configs.OUT_PATH / 'submit' / 'clusters_v6'), 
    format='zip', 
    root_dir=configs.OUT_PATH / 'submit',
    # base_dir=configs.OUT_PATH / 'submit',
  )
  



def concat_lost_table():
  paths = list(configs.PHOTOZ_SPECZ_LEG_FOLDER.glob('*lost.csv'))
  write_table(concat_tables(paths), configs.OUT_PATH / 'lost.parquet')





def clear_comparison_path():
  for p in configs.SUBMIT_FOLDER.glob('**/comparison.csv'):
    if p.exists():
      p.unlink()
  for p in configs.SUBMIT_FOLDER.glob('**/stats*.csv'):
    if p.exists():
      p.unlink()




if __name__ == "__main__":
  config_dask()
  clear_comparison_path()
  clusters_v6_pipeline()
  clusters_v5_remake_pipeline()
  hydra_neighbours_pipeline()
  concat_lost_table()
  create_zip()