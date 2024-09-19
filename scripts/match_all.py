
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from argparse import ArgumentParser

from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.configs import configs
from splusclusters.loaders import (LoadClusterInfoStage, LoadLegacyRadialStage,
                                   LoadPhotozRadialStage, LoadSpeczRadialStage,
                                   load_clusters, load_members_index_v6,
                                   load_photoz)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def match_all_pipeline(overwrite: bool = False, version: int = 6, z_photo_delta: float | None = None):
  configs.Z_SPEC_DELTA = configs.Z_SPEC_DELTA_PAULO
  if z_photo_delta is not None:
    configs.Z_PHOTO_DELTA = z_photo_delta
  else:
    configs.Z_PHOTO_DELTA = configs.Z_SPEC_DELTA_PAULO
  
  # df_clusters = load_clusters()
  df_clusters = load_members_index_v6()
  df_clusters = df_clusters[df_clusters.name.isin(['A168', 'MKW4'])]
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters, version=version),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=overwrite),
  )
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
if __name__ == "__main__":
  parser = ArgumentParser(description="Website")
  parser.add_argument('--v5', action='store_true')
  parser.add_argument('--v6', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--delta', action='store', default=None, type=float)
  args = parser.parse_args()
  
  v = 6 if args.v6 else 5
  match_all_pipeline(overwrite=args.overwrite, version=v, z_photo_delta=args.delta)