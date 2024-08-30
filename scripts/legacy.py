
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from argparse import ArgumentParser

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.configs import configs
from splusclusters.external import DownloadSplusPhotozStage, FixZRange
from splusclusters.loaders import (LoadClusterInfoStage, LoadLegacyRadialStage,
                                   LoadPhotozRadialStage, LoadSpeczRadialStage,
                                   load_clusters, load_eRASS, load_legacy,
                                   load_members_index_v6, load_photoz)
from splusclusters.match import (LegacyRadialSearchStage,
                                 PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def legacy_pipeline_v5(overwrite: bool = False):
  df_clusters = load_clusters()
  df_legacy, legacy_skycoord = load_legacy()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    LegacyRadialSearchStage(overwrite=overwrite),
  )
  PipelineStorage().write('df_legacy', df_legacy)
  PipelineStorage().write('legacy_skycoord', legacy_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)


def legacy_pipeline_v6(overwrite: bool = False, compile_all: bool = False):
  df_clusters = load_members_index_v6()
  df_legacy, legacy_skycoord = load_legacy()
  
  stages = [
    LoadClusterInfoStage(df_clusters),
    LegacyRadialSearchStage(overwrite=overwrite),
  ]
  if compile_all:
    stages += [
      LoadLegacyRadialStage(),
      LoadSpeczRadialStage(),
      LoadPhotozRadialStage(),
      PhotozSpeczLegacyMatchStage(overwrite=overwrite),
    ]
    
  pipe = Pipeline(*stages)
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)


  
if __name__ == "__main__":
  parser = ArgumentParser(description="Website")
  parser.add_argument('--v5', action='store_true')
  parser.add_argument('--v6', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--all', action='store_true')
  args = parser.parse_args()
  
  if args.v5:
    legacy_pipeline_v5(overwrite=args.overwrite)
  if args.v6:
    legacy_pipeline_v6(overwrite=args.overwrite, compile_all=args.all)