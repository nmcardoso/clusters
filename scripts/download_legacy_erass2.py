import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from typing import Literal

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.constants import *
from splusclusters.external import DownloadLegacyCatalogStage
from splusclusters.loaders import (LoadERASS2InfoStage, load_eRASS_2,
                                   load_photoz, load_spec)
from splusclusters.match import PhotoZRadialSearchStage, SpecZRadialSearchStage
from splusclusters.plots import ClusterPlotStage


def download_legacy_erass2_pipeline(clear: bool = False, z_type: Literal['spec', 'photo', 'both'] = 'spec'):
  df_clusters = load_eRASS_2()
  if z_type == 'spec':
    df_clusters = df_clusters[(df_clusters.BEST_Z_TYPE != 'photo_z') & (df_clusters.BEST_Z <= 0.1)].iloc[360:]
  elif z_type == 'photo':
    df_clusters = df_clusters[(df_clusters.BEST_Z_TYPE == 'photo_z') & (df_clusters.BEST_Z <= 0.1)]
  
  if clear:
    for p in LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
        
  ls10_pipe = Pipeline(
    LoadERASS2InfoStage(df_clusters),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', overwrite=False, workers=8)
  )
  ls10_pipe.map_run('cls_name', df_clusters.NAME.values, workers=1)
  
  
if __name__ == "__main__":
  download_legacy_erass2_pipeline()