
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from argparse import ArgumentParser

import pandas as pd
from astromodule.distance import mpc2arcsec
from astromodule.pipeline import Pipeline, PipelineStorage
from astromodule.table import concat_tables
from astropy import units as u
from astropy.cosmology import LambdaCDM
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
  else:
    allowed_names = read_table(configs.ROOT / 'tables' / 'clusters_59.csv').name
    df_clusters = df_clusters[df_clusters.name.isin(allowed_names)]
    
  df_clusters = df_clusters[(df_clusters.z_spec >= 0.02) & (df_clusters.z_spec <= 0.1)]
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
      (df.flag_member.isin([0, 1])) & 
      (df.z.between(z_cluster-z_photo_delta, z_cluster+z_photo_delta)) &
      (df.radius_Mpc < 5*r200_Mpc)
    )
    df = df[mask]
    df.insert(0, 'cluster_name', name)
    # df.insert(1, 'cluster_id', i+1)
    final_df = concat_tables([final_df, df])
    print(
      'cluster_id:', i+1, '\tcluster_name:', f'{name: <17}', '\tz_cluster:', z_cluster, 
      '\tz_range:', f'[{df.z.min():.4f}, {df.z.max():.4f}]', '\tNmemb (total):', n_memb,
      '\tNmemb (5R200):', len(df[df.flag_member == 0])
    )
  print('Table size:', len(final_df))
  write_table(final_df, configs.OUT_PATH / 'table_3.parquet')
  
  df = read_table(configs.PHOTOZ_SPECZ_LEG_FOLDER / f'A168.parquet')
  cluster = df_clusters[df_clusters.name == 'A168']
  r200 = cluster['R200_Mpc'].values[0]
  z = cluster['z_spec'].values[0]
  cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
  search_radius_deg = mpc2arcsec(5*r200, z, cosmo).to(u.deg).value
  df = df[df.radius_deg < search_radius_deg]
  write_table(df, configs.OUT_PATH / 'table_4.parquet')
  
  df = read_table(configs.PHOTOZ_SPECZ_LEG_FOLDER / f'MKW4.parquet')
  cluster = df_clusters[df_clusters.name == 'MKW4']
  r200 = cluster['R200_Mpc'].values[0]
  z = cluster['z_spec'].values[0]
  cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
  search_radius_deg = mpc2arcsec(5*r200, z, cosmo).to(u.deg).value
  df = df[df.radius_deg < search_radius_deg]
  write_table(df, configs.OUT_PATH / 'table_5.parquet')
  
  
if __name__ == "__main__":
  parser = ArgumentParser(description="Website")
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--delta', action='store', default=None, type=float)
  parser.add_argument('--two', action='store_true')
  args = parser.parse_args()
  
  match_all_pipeline(overwrite=args.overwrite, z_photo_delta=args.delta, two=args.two)