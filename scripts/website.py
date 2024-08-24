import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from argparse import ArgumentParser

from astromodule.pipeline import Pipeline

from splusclusters.external import CopyXrayStage
from splusclusters.loaders import (LoadAllRadialStage, LoadClusterInfoStage,
                                   LoadPhotozRadialStage, LoadSpeczRadialStage,
                                   load_clusters, load_members_index_v6)
from splusclusters.plots import (ClusterPlotStage, MagDiffPlotStage,
                                 VelocityPlotStage)
from splusclusters.website import WebsitePagesStage


def website_pipeline(overwrite: bool = False, version: int = 6, dev: bool = False, index_only: bool = False):
  if version == 5:
    df_clusters = load_clusters()
  else:
    df_clusters = load_members_index_v6()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters, version=version),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=overwrite, fmt='jpg', separated=True, version=version),
    VelocityPlotStage(overwrite=overwrite, fmt='jpg', separated=True, version=version),
    MagDiffPlotStage(overwrite=overwrite, fmt='jpg', separated=True, version=version),
    CopyXrayStage(overwrite=overwrite, fmt='png', version=version),
    WebsitePagesStage(df_clusters=df_clusters, version=version),
  )
  
  if not index_only:
    if not dev:
      pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
    else:
      pipe.map_run('cls_id', df_clusters.clsid.values[:2], workers=1)
  
  WebsitePagesStage(df_clusters=df_clusters, version=version).make_index()
  WebsitePagesStage(df_clusters=df_clusters, version=version).make_landing()
  

if __name__ == '__main__':
  parser = ArgumentParser(description="Website")
  parser.add_argument('--v5', action='store_true')
  parser.add_argument('--v6', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--dev', action='store_true')
  parser.add_argument('--index', action='store_true')
  args = parser.parse_args()
  
  if args.v5:
    website_pipeline(overwrite=args.overwrite, version=5, dev=args.dev, index_only=args.index)
  if args.v6:
    website_pipeline(overwrite=args.overwrite, version=6, dev=args.dev, index_only=args.index)