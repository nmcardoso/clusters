import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.constants import *
from splusclusters.external import DownloadLegacyCatalogStage
from splusclusters.loaders import (LoadClusterInfoStage, load_clusters,
                                   load_photoz, load_spec)
from splusclusters.match import PhotoZRadialSearchStage, SpecZRadialSearchStage
from splusclusters.plots import ClusterPlotStage


def download_legacy_pipeline(clear: bool = False):
  df_clusters = load_clusters()
  
  if clear:
    for p in LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
        
  ls10_pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', overwrite=False, workers=8)
  )
  ls10_pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  
if __name__ == "__main__":
  download_legacy_pipeline()