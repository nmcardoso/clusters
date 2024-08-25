
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from argparse import ArgumentParser

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.configs import configs
from splusclusters.external import DownloadSplusPhotozStage, FixZRange
from splusclusters.loaders import (LoadClusterInfoStage, LoadLegacyRadialStage,
                                   LoadSpeczRadialStage, load_clusters,
                                   load_eRASS, load_members_index_v6,
                                   load_photoz)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def photoz_pipeline_v5(overwrite: bool = False):
  df_clusters = load_clusters()
  df_photoz, photoz_skycoord = load_photoz()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=overwrite),
  )
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)


def photoz_pipeline_v6(overwrite: bool = False):
  df_clusters = load_members_index_v6()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    DownloadSplusPhotozStage(overwrite=overwrite),
  )
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  
  
def photoz_fix_pipeline():
  df_clusters = load_members_index_v6()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    FixZRange(),
  )
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)


  
if __name__ == "__main__":
  parser = ArgumentParser(description="Website")
  parser.add_argument('--v5', action='store_true')
  parser.add_argument('--v6', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--fix', action='store_true')
  args = parser.parse_args()
  
  if args.v5:
    photoz_pipeline_v5(overwrite=args.overwrite)
  if args.v6:
    photoz_pipeline_v6(overwrite=args.overwrite)
  if args.fix:
    photoz_fix_pipeline()