import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from argparse import ArgumentParser

from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.configs import configs
from splusclusters.loaders import (LoadClusterInfoStage, load_clusters,
                                   load_members_index_v6, load_spec)
from splusclusters.match import SpecZRadialSearchStage
from splusclusters.plots import ClusterPlotStage


def specz_pipeline_v6(overwrite: bool = False, z_photo_delta: float | None = None, two: bool = False):
  configs.Z_SPEC_DELTA = configs.Z_SPEC_DELTA_PAULO
  if z_photo_delta is not None:
    configs.Z_PHOTO_DELTA = z_photo_delta
  else:
    configs.Z_PHOTO_DELTA = configs.Z_SPEC_DELTA_PAULO
  
  # df_clusters = load_clusters()
  df_clusters = load_members_index_v6()
  
  if two:
    df_clusters = df_clusters[df_clusters.name.isin(['MKW4', 'A168'])]
  # else:
    # df_clusters = df_clusters[df_clusters.name.str.lower().isin(['fornax', 'antlia', 'hydra'])]
    # df_clusters = df_clusters.iloc[-4:]
  
  df_spec, specz_skycoord = load_spec()
  df_spec.rename(columns={'RA': 'ra_spec_all', 'DEC': 'dec_spec_all'}, inplace=True)
  df_spec = df_spec[
    df_spec.class_spec.str.upper().str.startswith('GALAXY') &
    df_spec.f_z.str.upper().str.startswith('KEEP')
  ]
  
  print(df_spec)
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters, version=6),
    SpecZRadialSearchStage(overwrite=overwrite),
  )
  
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)



  
if __name__ == '__main__':
  parser = ArgumentParser(description="Website")
  parser.add_argument('--v5', action='store_true')
  parser.add_argument('--v6', action='store_true')
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--delta', action='store', default=None, type=float)
  parser.add_argument('--two', action='store_true')
  args = parser.parse_args()
  
  if args.v5:
    pass
  if args.v6:
    specz_pipeline_v6(overwrite=args.overwrite, z_photo_delta=args.delta, two=args.two)