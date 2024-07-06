import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import pandas as pd
from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.constants import *
from splusclusters.external import DownloadLegacyCatalogStage
from splusclusters.loaders import (LoadAllRadialStage, LoadClusterInfoStage,
                                   LoadLegacyRadialStage,
                                   LoadPhotozRadialStage, LoadSpeczRadialStage,
                                   load_eRASS_2, load_photoz, load_spec)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def antlia_hydra_plot_pipeline():
  df_clusters = pd.DataFrame([
    {'NAME': 'Hydra', 'ra': 159.17416, 'dec': -27.52444, 'z_spec': 0.0126},
    {'NAME': 'A636', 'ra': 157.148, 'dec': -35.642, 'z_spec': 0.0093},
  ])
  df_photoz, photoz_skycoord = load_photoz()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', workers=8),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=False),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=False, splus_only=False),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.NAME.values, workers=1)
  plot_paths = [PLOTS_FOLDER / f'cls_{c}.pdf' for c in df2.NAME.values]
  # plot_paths = [p for p in plot_paths if p.exists() and (p.stat().st_size > 100_000)]
  df2 = df2[df2.NAME.isin([p.name for p in plot_paths])]
  # for i, row in df2.iterrows():
  #   len(radial_search(SkyCoord(ra=row.ra, dec=row.dec, unit='deg'), OUT_PATH / 'photoz' / f'{row.name}.parquet'), row.) > 0
  concat_plot_path = PLOTS_FOLDER / 'antlia_hydra.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  
  
if __name__ == '__main__':
  antlia_hydra_plot_pipeline()