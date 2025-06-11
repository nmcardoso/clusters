from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import luigi.format
import numpy as np
import pandas as pd
from astromodule.distance import mpc2arcsec
from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStage, PipelineStorage
from astromodule.table import (concat_tables, crossmatch, fast_crossmatch,
                               guess_coords_columns, radial_search, selfmatch)
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM
from astropy.table import Table
from prefect import flow, task
from pylegs.io import read_table, write_table

import luigi
from splusclusters.configs import configs
from splusclusters.utils import Timming


@dataclass
class ClusterInfo:
  name: str
  ra: float
  dec: float
  z: float
  search_radius_Mpc: float = None
  search_radius_deg: float = None
  r500_Mpc: float = None
  r500_deg: float = None
  r200_Mpc: float = None
  r200_deg: float = None
  z_photo_range: Tuple[float, float] = None
  z_spec_range: Tuple[float, float] = None
  
  @property
  def coord(self):
    return SkyCoord(ra=self.ra, dec=self.dec, unit=u.deg)
  



@task(task_run_name='cluster-params-{cls_name}', version='1.0', persist_result=False)
def cluster_params(df_clusters: pd.DataFrame, cls_name: str):
  cluster = df_clusters[df_clusters.name == cls_name]
  ra_col, dec_col = guess_coords_columns(cluster)
  ra = cluster[ra_col].values[0]
  dec = cluster[dec_col].values[0]
  z_col = 'zspec' if 'zspec' in cluster.columns else 'z_spec'
  z = cluster[z_col].values[0]
  
  cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
  if 'R500_Mpc' in cluster:
    r500_Mpc = cluster['R500_Mpc'].values[0]
    r500_deg = mpc2arcsec(r500_Mpc, z, cosmo).to(u.deg).value
  else:
    r500_Mpc = None
    r500_deg = None
  
  if 'R200_Mpc' in cluster:
    r200_Mpc = cluster['R200_Mpc'].values[0]
    r200_deg = mpc2arcsec(r200_Mpc, z, cosmo).to(u.deg).value
  else:
    r200_Mpc = None
    r200_deg = None
  
  if 'search_radius_Mpc' in df_clusters.columns:
    search_radius_Mpc = cluster['search_radius_Mpc'].values[0]
  else:
    search_radius_Mpc = 15
  search_radius_deg = min(mpc2arcsec(search_radius_Mpc, z, cosmo).to(u.deg).value, 17)
  if search_radius_deg > 17:
    print(f'Cluster angular radius @ 15Mpc = {search_radius_deg:.2f} deg, limiting to 17 deg')
    search_radius_deg = min(search_radius_deg, 17)
    
  print('Cluster Name:', cls_name)
  print(f'RA: {ra:.3f}, DEC: {dec:.3f}, z: {z:.2f}, search radius: {search_radius_deg:.2f}')
  
  return ClusterInfo(
    name=cls_name,
    ra=ra,
    dec=dec,
    z=z,
    search_radius_Mpc=search_radius_Mpc,
    search_radius_deg=search_radius_deg,
    r500_Mpc=r500_Mpc,
    r500_deg=r500_deg,
    r200_Mpc=r200_Mpc,
    r200_deg=r200_deg,
    z_photo_range=(z - configs.Z_PHOTO_DELTA, z + configs.Z_PHOTO_DELTA),
    z_spec_range=(z - configs.Z_SPEC_DELTA, z + configs.Z_SPEC_DELTA),
  )



# class ComputeClusterInfo(luigi.Task):
#   cls_name = luigi.Parameter()
#   version = luigi.IntParameter()
  
#   def requires(self):
#     return LoadClusterCatalog(version=self.version)

#   def output(self):
#     return luigi.LocalTarget(configs.LUIGI_FOLDER / f'info-{self.cls_name}.pckl')
  
#   def run(self):
#     import pickle
#     with self.input().open() as f:
#       df_clusters = pickle.load(f)
#     print(f'{df_clusters=}')
#     info = cluster_params(df_clusters, self.cls_name)
#     with self.output().open('w') as f:
#       pickle.dump(info, f)