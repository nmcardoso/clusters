import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.io import merge_pdf, read_table, write_table
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.configs import configs
from splusclusters.external import (ArchiveDownloadLegacyCatalogStage,
                                    DownloadLegacyCatalogStage)
from splusclusters.loaders import (LoadAllRadialStage, LoadClusterInfoStage,
                                   LoadGenericInfoStage, LoadLegacyRadialStage,
                                   LoadPauloInfoStage, LoadPhotozRadialStage,
                                   LoadSpeczRadialStage,
                                   PrepareCatalogToSubmitStage,
                                   load_catalog_v6, load_clusters,
                                   load_photoz2, load_spec)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def clusters_v5_remake_pipeline(clear: bool = False):
  df_clusters = load_clusters()
  df_photoz, photoz_skycoord = load_photoz2()
  df_spec, specz_skycoord = load_spec()
  
  configs.Z_SPEC_DELTA = 0.02
  configs.SUBMIT_FOLDER = configs.SUBMIT_FOLDER / 'antigos'
  configs.SUBMIT_FOLDER.mkdir(exist_ok=True, parents=True)
  
  if clear:
    for p in configs.LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=True),
    SpecZRadialSearchStage(overwrite=True),
    # DownloadLegacyCatalogStage('cls_search_radius_deg', overwrite=True, workers=5),
    ArchiveDownloadLegacyCatalogStage(
      radius_key='cls_search_radius_deg', workers=15,
      overwrite=True, overwrite_bricks=False, 
    ),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=True),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=True, splus_only=False),
    PrepareCatalogToSubmitStage(overwrite=True),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  plot_paths = [configs.PLOTS_FOLDER / f'cls_{c}.pdf' for c in df_clusters.name.values]
  plot_paths = [p for p in plot_paths if p.exists()]
  concat_plot_path = configs.PLOTS_FOLDER / 'clusters_v6+antigos.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  
  df_clusters['clsid'] = df_clusters.clsid.astype(str).str.zfill(4)
  
  df_clusters = df_clusters.rename(columns={'ra': 'RA', 'dec': 'DEC', 'z_spec': 'zspec'})
  write_table(
    df_clusters[['clsid', 'name', 'RA', 'DEC', 'zspec']], 
    configs.SUBMIT_FOLDER / 'index.dat'
  )






def hydra_neighbours_pipeline(clear: bool = False):
  df_clusters = read_table(configs.CATALOG_V6_HYDRA_TABLE_PATH)
  df_photoz, photoz_skycoord = load_photoz2()
  df_spec, specz_skycoord = load_spec()
  df_clusters['clsid'] = list(range(len(df_clusters)))
  
  configs.Z_SPEC_DELTA = 0.02
  configs.SUBMIT_FOLDER = configs.SUBMIT_FOLDER / 'hydra'
  configs.SUBMIT_FOLDER.mkdir(exist_ok=True, parents=True)
  
  if clear:
    for p in configs.LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
  
  pipe = Pipeline(
    LoadGenericInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    ArchiveDownloadLegacyCatalogStage(
      radius_key='cls_search_radius_deg', workers=15,
      overwrite=True, overwrite_bricks=False, 
    ),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=False),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=False, splus_only=False),
    PrepareCatalogToSubmitStage(overwrite=False),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.name.values, workers=1)
  
  plot_paths = [configs.PLOTS_FOLDER / f'cls_{c}.pdf' for c in df_clusters.name.values]
  plot_paths = [p for p in plot_paths if p.exists()]
  concat_plot_path = configs.PLOTS_FOLDER / 'clusters_v6+hydra.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  
  df_clusters['clsid'] = df_clusters.clsid.astype(str).str.zfill(4)
  
  df_clusters = df_clusters.rename(columns={'z_spec': 'zspec'})
  write_table(
    df_clusters[['clsid', 'name', 'ra', 'dec', 'zspec']], 
    configs.SUBMIT_FOLDER / 'index.dat'
  )






def clusters_v6_pipeline(clear: bool = False):
  df_clusters = load_catalog_v6()
  df_photoz, photoz_skycoord = load_photoz2()
  df_spec, specz_skycoord = load_spec()
  
  configs.Z_SPEC_DELTA = 0.02
  configs.SUBMIT_FOLDER = configs.SUBMIT_FOLDER / 'novos'
  configs.SUBMIT_FOLDER.mkdir(exist_ok=True, parents=True)
  
  if clear:
    for p in configs.LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
        
  ls10_pipe = Pipeline(
    LoadPauloInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=True),
    SpecZRadialSearchStage(overwrite=True),
    # DownloadLegacyCatalogStage('cls_search_radius_deg', overwrite=False, workers=5),
    ArchiveDownloadLegacyCatalogStage(
      radius_key='cls_search_radius_deg', workers=15,
      overwrite=True, overwrite_bricks=False, 
    ),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=True),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=True, splus_only=False),
    PrepareCatalogToSubmitStage(overwrite=True),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  ls10_pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  plot_paths = [configs.PLOTS_FOLDER / f'cls_{c}.pdf' for c in df_clusters.name.values]
  plot_paths = [p for p in plot_paths if p.exists()]
  concat_plot_path = configs.PLOTS_FOLDER / 'clusters_v6+novos.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  
  df_clusters['clsid'] = list(range(len(df_clusters)))
  df_clusters['clsid'] = df_clusters.clsid.astype(str).str.zfill(4)
  
  write_table(
    df_clusters[['clsid', 'name', 'RA', 'DEC', 'zspec']], 
    configs.SUBMIT_FOLDER / 'index.dat'
  )


if __name__ == "__main__":
  # clusters_v5_remake_pipeline()
  # clusters_v6_pipeline()
  hydra_neighbours_pipeline()