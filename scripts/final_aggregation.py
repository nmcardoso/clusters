
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from argparse import ArgumentParser

import pandas as pd
from astromodule.pipeline import Pipeline, PipelineStorage
from astromodule.table import concat_tables
from pylegs.io import read_table, write_table

from splusclusters.configs import configs
from splusclusters.loaders import (LoadClusterInfoStage, LoadLegacyRadialStage,
                                   LoadPhotozRadialStage, LoadSpeczRadialStage,
                                   load_clusters, load_members_index_v6,
                                   load_photoz, load_spec)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def match_all_pipeline(overwrite: bool = False, z_photo_delta: float | None = None, two: bool = False):
  configs.Z_SPEC_DELTA = configs.Z_SPEC_DELTA_PAULO
  if z_photo_delta is not None:
    configs.Z_PHOTO_DELTA = z_photo_delta
  else:
    configs.Z_PHOTO_DELTA = configs.Z_SPEC_DELTA_PAULO
  
  df_clusters = load_members_index_v6()
  
  if two:
    df_clusters = df_clusters[df_clusters.name.isin(['MKW4', 'A168'])]
  
  df_clusters = df_clusters.sort_values(by='z_spec', ascending=True).reset_index()

  final_df = pd.DataFrame()
  for i, row in df_clusters.iterrows():
    z_cluster = row['z_spec']
    r200_Mpc = row['R200_Mpc']
    cls_id = row['clsid']
    name = row['name']
    n_memb = row['Nmemb']
    
    path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{name}.parquet'
    df = read_table(path)
    mask = (
      (df.flag_member == 0) & 
      (df.z.between(z_cluster-configs.Z_SPEC_DELTA, z_cluster+configs.Z_SPEC_DELTA)) &
      (df.radius_Mpc < 5*r200_Mpc)
    )
    df = df[mask]
    df.insert(0, 'cluster_name', name)
    df.insert(1, 'cluster_id', i+1)
    final_df = concat_tables([final_df, df])
    print(
      'cluster_id:', i+1, '\tcluster_name:', name, '\t\tz_cluster:', z_cluster, 
      '\tz_range:', f'[{df.z.min():.4f}, {df.z.max():.4f}]', '\tNmemb (total):', n_memb,
      '\tNmemb (5R200):', len(df)
    )
  write_table(final_df, configs.OUT_PATH / 'table_2.parquet')
  
  
if __name__ == "__main__":
  parser = ArgumentParser(description="Website")
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--delta', action='store', default=None, type=float)
  parser.add_argument('--two', action='store_true')
  args = parser.parse_args()
  
  match_all_pipeline(overwrite=args.overwrite, z_photo_delta=args.delta, two=args.two)