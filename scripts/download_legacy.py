import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from argparse import ArgumentParser

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.configs import configs
from splusclusters.external import DownloadLegacyCatalogStage
from splusclusters.loaders import (LoadClusterInfoStage, load_clusters,
                                   load_photoz, load_spec)
from splusclusters.match import PhotoZRadialSearchStage, SpecZRadialSearchStage
from splusclusters.plots import ClusterPlotStage


def download_legacy_pipeline(clear: bool = False, overwrite: bool = False):
  df_clusters = load_clusters()
  
  if clear:
    for p in configs.LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
        
  ls10_pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    DownloadLegacyCatalogStage('cls_search_radius_deg', overwrite=overwrite, workers=7)
  )
  ls10_pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  
if __name__ == "__main__":
  parser = ArgumentParser(description="Legacy")
  parser.add_argument('--overwrite', action='store_true')
  args = parser.parse_args()
  
  download_legacy_pipeline(clear=args.clear, overwrite=args.overwrite)