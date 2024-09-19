
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import warnings
from argparse import ArgumentParser

warnings.filterwarnings('ignore')
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
from splusclusters.utils import config_dask


def photoz_pipeline_v5(overwrite: bool = False, z_photo_delta: float | None = None):
  df_clusters = load_clusters()
  df_photoz, photoz_skycoord = load_photoz()
  
  if z_photo_delta is not None:
    configs.Z_PHOTO_DELTA = z_photo_delta
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=overwrite),
  )
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)


def photoz_pipeline_v6(overwrite: bool = False, z_photo_delta: float | None = None):
  df_clusters = load_members_index_v6()
  # df_clusters = df_clusters[df_clusters.name.isin(['A168', 'MKW4'])]
  
  if z_photo_delta is not None:
    configs.Z_PHOTO_DELTA = z_photo_delta
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    DownloadSplusPhotozStage(overwrite=overwrite),
  )
  
  config_dask()
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  
  
def photoz_fix_pipeline():
  configs.Z_PHOTO_DELTA = configs.Z_SPEC_DELTA_PAULO
  configs.Z_SPEC_DELTA = configs.Z_SPEC_DELTA_PAULO
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
  parser.add_argument('--delta', action='store', default=None, type=float)
  args = parser.parse_args()
  
  if args.v5:
    photoz_pipeline_v5(overwrite=args.overwrite, z_photo_delta=args.delta)
  if args.v6:
    photoz_pipeline_v6(overwrite=args.overwrite, z_photo_delta=args.delta)
  if args.fix:
    photoz_fix_pipeline()