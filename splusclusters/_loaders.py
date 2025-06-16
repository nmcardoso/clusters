from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import List, Tuple

import dagster as dg
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
from pylegs.io import read_table, write_table

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
  z_photo_delta: float = 0.05
  z_spec_delta: float = 0.02
  z_photo_range: Tuple[float, float] = None
  z_spec_range: Tuple[float, float] = None
  magnitude_range: Tuple[float, float] = None
  version: int = 7
  plot_format: str = 'png'
  
  @property
  def coord(self):
    return SkyCoord(ra=self.ra, dec=self.dec, unit=u.deg)
  
  @property
  def output_folder(self):
    return configs.ROOT / f'outputs_v{self.version}'
  
  @property
  def photoz_path(self):
    return self.output_folder / 'photoz' / f'{self.name}.parquet'
  
  @property
  def photoz_df(self):
    return _load_cluster_product(self, self.photoz_path)
  
  @property
  def specz_path(self):
    return self.output_folder / 'specz' / f'{self.name}.parquet'
  
  @property
  def specz_df(self):
    return _load_cluster_product(self, self.specz_path)
  
  @property
  def specz_outrange_path(self):
    return self.output_folder / 'specz_outrange' / f'{self.name}.parquet'
  
  @property
  def specz_outrange_df(self):
    return _load_cluster_product(self, self.specz_outrange_path)
  
  @property
  def compilation_path(self):
    return self.output_folder / 'comp' / f'{self.name}.parquet'
  
  @property
  def compilation_df(self):
    return _load_cluster_product(self, self.compilation_path)
  
  @property
  def legacy_path(self):
    return self.output_folder / 'legacy' / f'{self.name}.parquet'
  
  @property
  def legacy_df(self):
    return _load_cluster_product(self, self.legacy_path)
  
  @property
  def legacy_bricks_folder(self):
    return self.output_folder / 'legacy_bricks'
  
  @property
  def submit_folder(self):
    return self.output_folder / 'submit'

  @property
  def plots_folder(self):
    return self.output_folder / 'plots'
  
  @property
  def plot_xray_vector_path(self):
    return self.plots_folder / f'xray_{self.name}.eps'
  
  @property
  def plot_xray_raster_path(self):
    return self.plots_folder / f'xray_{self.name}.{self.plot_format}'
  
  @property
  def plot_specz_path(self):
    return self.plots_folder / f'specz_{self.name}.{self.plot_format}'
  
  @property
  def plot_photoz_path(self):
    return self.plots_folder / f'photoz_{self.name}.{self.plot_format}'
  
  @property
  def plot_legacy_coverage_path(self):
    return self.plots_folder / f'legacy_coverage_{self.name}.{self.plot_format}'
  
  @property
  def plot_photoz_specz_path(self):
    return self.plots_folder / f'photoz_specz_{self.name}.{self.plot_format}'
  
  @property
  def plot_specz_contours_path(self):
    return self.plots_folder / f'specz_contours_{self.name}.{self.plot_format}'
  
  @property
  def plot_redshift_diagonal_path(self):
    return self.plots_folder / f'redshift_diagonal_{self.name}.{self.plot_format}'
  
  @property
  def plot_redshift_diff_mag_path(self):
    return self.plots_folder / f'redshift_diff_mag_{self.name}.{self.plot_format}'
  
  @property
  def plot_redshift_diff_distance_path(self):
    return self.plots_folder / f'redshift_diff_distance_{self.name}.{self.plot_format}'
  
  @property
  def plot_redshift_diff_odds_path(self):
    return self.plots_folder / f'redshift_diff_odds_{self.name}.{self.plot_format}'
  
  @property
  def plot_redshift_hist_members_path(self):
    return self.plots_folder / f'redshift_hist_members_{self.name}.{self.plot_format}'
  
  @property
  def plot_redshift_hist_interlopers_path(self):
    return self.plots_folder / f'redshift_hist_interlopers_{self.name}.{self.plot_format}'
  
  @property
  def plot_redshift_hist_all_path(self):
    return self.plots_folder / f'redshift_hist_all_{self.name}.{self.plot_format}'
  
  @property
  def plot_specz_velocity_path(self):
    return self.plots_folder / f'specz_velocity_{self.name}.{self.plot_format}'
  
  @property
  def plot_specz_distance_path(self):
    return self.plots_folder / f'specz_distance_{self.name}.{self.plot_format}'
  
  @property
  def plot_photoz_velocity_path(self):
    return self.plots_folder / f'photoz_velocity_{self.name}.{self.plot_format}'
  
  @property
  def plot_photoz_distance_path(self):
    return self.plots_folder / f'photoz_distance_{self.name}.{self.plot_format}'
  
  @property
  def plot_specz_velocity_rel_position_path(self):
    return self.plots_folder / f'specz_velocity_rel_position_{self.name}.{self.plot_format}'
  
  @property
  def plot_mag_diff_path(self):
    return self.plots_folder / f'mag_diff_{self.name}.{self.plot_format}'
  
  @property
  def plot_mag_diff_hist_path(self):
    return self.plots_folder / f'mag_diff_hist_{self.name}.{self.plot_format}'
  
  @property
  def plot_agg_velocity_path(self):
    return self.plots_folder / f'agg_velocity_{self.name}.{self.plot_format}'
  
  @property
  def plot_agg_info_path(self):
    return self.plots_folder / f'agg_info_{self.name}.{self.plot_format}'
  
  @property
  def plot_agg_mag_diff_path(self):
    return self.plots_folder / f'agg_mag_diff_{self.name}.{self.plot_format}'
  
  @property
  def plot_xray_raster_path(self):
    return self.plots_folder / f'xray_{self.name}.{self.plot_format}'
  
  @property
  def website_root(self):
    return configs.ROOT / 'docs'
  
  @property
  def website_version_root(self):
    return self.website_root / f'clusters_v{self.version}'
  
  @property
  def website_cluster_page(self):
    return self.website_version_root / self.name




def _cluster_name_std(df: pd.DataFrame):
  df['name'] = df.name.str.replace(' ', '')
  df['name'] = df.name.str.replace(r'^ABELL0+', 'A', regex=True)
  df['name'] = df.name.str.replace(r'^ABELL', 'A', regex=True)
  df['name'] = df.name.str.replace(r'^Abell0+', 'A', regex=True)
  df['name'] = df.name.str.replace(r'^Abell', 'A', regex=True)
  df['name'] = df.name.str.replace(r'^A0+', 'A', regex=True)
  df['name'] = df.name.str.replace(r'^AS', 'A', regex=True)
  df['name'] = df.name.str.replace(r'^ACO', 'A', regex=True)
  df['name'] = df.name.str.replace(r'^A(\d+)\w*', r'A\1', regex=True)
  df['name'] = df.name.str.replace(r'^RXC', 'MCXC', regex=True)
  df['name'] = df.name.str.replace(r'^MKW0+', 'MKW', regex=True)
  return df



def _df_subset(df, subset):
  if isinstance(subset, list):
    return df[df['name'].isin(subset)]
  elif subset is True:
    return df[df['name'].isin(['MKW4', 'A168'])]
  return df



def _load_clusters_v5(subset: bool = False):
  df_clusters = read_table(configs.MEMBERS_V5_PATH / 'index.dat')
  df_clusters['clsid'] = df_clusters['clsid'].astype('int')
  df_index = read_table(configs.MEMBERS_V5_PATH / 'names.dat', columns=['clsid', 'name'])
  df = df_clusters.set_index('clsid').join(df_index.set_index('clsid'), how='inner', rsuffix='_r')
  df_filter = read_table(configs.ROOT / 'tables/catalog_v6_splus_only_filter.csv')
  df = df[df.name.isin(df_filter.name)].copy().reset_index()
  return _df_subset(df, subset)



def _load_catalog_v6(subset: bool = False):
  df = _cluster_name_std(read_table(configs.CATALOG_V6_TABLE_PATH, comment='#'))
  df['clsid'] = list(range(1, len(df)+1))
  return _df_subset(df, subset)



def _load_catalog_v6_old():
  df = _cluster_name_std(read_table(configs.CATALOG_V6_OLD_TABLE_PATH, comment='#'))
  return df



def _load_catalog_v6_hydra():
  df = _cluster_name_std(read_table(configs.CATALOG_V6_HYDRA_TABLE_PATH, comment='#'))
  df['clsid'] = list(range(1, len(df)+1))
  return df



def _load_catalog_v7(subset: bool = False):
  df = _load_shiftgap_index_v7()
  # df = df[['name', 'ra', 'dec', 'z_spec']]
  df = df.rename({'z_spec': 'zspec'})
  return _df_subset(df, subset)



def _load_shiftgap_index_v5():
  df_index = read_table(configs.MEMBERS_V5_PATH / 'names.dat')
  return df_index



def _load_shiftgap_v5(cls_id):
  path = configs.MEMBERS_V5_FOLDER / f'cluster.gals.sel.shiftgap.iter.{str(cls_id).zfill(5)}'
  col_names = [
    'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
    'radius_Mpc', 'v_offset', 'flag_member'
  ] # 0 - member; 1 - interloper
  return read_table(path, fmt='dat', col_names=col_names)



def _load_shiftgap_index_v6():
  cols = [
    'clsid', 'ra', 'dec', 'z_spec', 'sigma_cl_kms', 'sigma_cl_lower', 
    'sigma_cl_upper', 'r200', 'r200_lower', 'r200_upper', 'm200',
    'm200_lower', 'm200_upper', 'nwcls', 'n_memb', 'n_memb_wR200', 'name'
  ]
  path = configs.MEMBERS_V6_PATH / 'info_cls_shiftgap_iter_10.0hmpcf_nrb.dat'
  names_df = read_table(path, col_names=cols, comment='#')
  path = configs.MEMBERS_V6_PATH / 'info_cls_shiftgap_iter_10.0hmpcf.dat_nrb'
  cols = [
    'clsid', 'ra', 'dec', 'z_spec', 'veli', 'velf', 'Nwcls', 'Nmemb', 'sigma_p',
    'sigma_p_lower', 'sigma_p_upper', 'R500_Mpc', 'R500_lower', 'R500_upper',
    'M500_solar', 'M500_lower', 'M500_upper', 'R200_Mpc', 'R200_lower',
    'R200_upper', 'M200_solar', 'M200_lower', 'M200_upper', 'znew', 'znew_err',
    'Rap', 'Nmemb_wR200', 'col1', 'col2', 'col3', 'col4'
  ]
  info_df = read_table(path, fmt='dat', col_names=cols, comment='#')
  info_df['name'] = names_df['name'].values
  return info_df



def _load_shiftgap_v6(cls_name: str = None, cls_id: str | int = None):
  if cls_name is not None:
    index_df = _load_shiftgap_index_v6()
    if cls_name not in index_df.name: 
      return None
    cls_id = index_df[index_df.name == cls_name].clsid.values[0]
  prefixed_id = str(int(cls_id)).zfill(5)
  path = configs.MEMBERS_V6_FOLDER / f'cluster.gals.sel.shiftgap.iter.{prefixed_id}'
  cols = [
    'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
    'radius_Mpc', 'v_offset', 'flag_member'
  ]
  return read_table(path, fmt='dat', col_names=cols)



def _load_shiftgap_index_v7():
  cols = [
    'clsid', 'ra', 'dec', 'z_spec', 'sigma_cl_kms', 'sigma_cl_lower', 
    'sigma_cl_upper', 'r200', 'r200_lower', 'r200_upper', 'm200',
    'm200_lower', 'm200_upper', 'nwcls', 'n_memb', 'n_memb_wR200', 'name'
  ]
  path = configs.MEMBERS_V7_PATH / 'info_cls_shiftgap_iter_10.0hmpcf_nrb.dat'
  names_df = read_table(path, col_names=cols, comment='#')
  path = configs.MEMBERS_V7_PATH / 'info_cls_shiftgap_iter_10.0hmpcf.dat_nrb'
  cols = [
    'clsid', 'ra', 'dec', 'z_spec', 'veli', 'velf', 'Nwcls', 'Nmemb', 'sigma_p',
    'sigma_p_lower', 'sigma_p_upper', 'R500_Mpc', 'R500_lower', 'R500_upper',
    'M500_solar', 'M500_lower', 'M500_upper', 'R200_Mpc', 'R200_lower',
    'R200_upper', 'M200_solar', 'M200_lower', 'M200_upper', 'znew', 'znew_err',
    'Rap', 'Nmemb_wR200', 'col1', 'col2', 'col3', 'col4'
  ]
  info_df = read_table(path, fmt='dat', col_names=cols, comment='#')
  info_df['name'] = names_df['name'].values
  return info_df



def _load_shiftgap_v7(cls_name: str = None, cls_id: str | int = None):
  if cls_name is not None:
    index_df = _load_shiftgap_index_v7()
    cls_id = index_df[index_df.name == cls_name].clsid.values[0]
  prefixed_id = str(int(cls_id)).zfill(5)
  path = configs.MEMBERS_V7_FOLDER / f'cluster.gals.sel.shiftgap.iter.{prefixed_id}'
  cols = [
    'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
    'radius_Mpc', 'v_offset', 'flag_member'
  ]
  return read_table(path, fmt='dat', col_names=cols)




def _load_eRASS():
  df_erass = read_table(configs.ERASS_TABLE_PATH)
  return df_erass



def _load_full_eRASS():
  df_full_eras = read_table(configs.FULL_ERASS_TABLE_PATH)
  return df_full_eras



def _load_eRASS_2():
  return read_table(configs.ERASS2_TABLE_PATH)



def _load_heasarc():
  df = _cluster_name_std(read_table(configs.HEASARC_TABLE_PATH))
  df = df.drop_duplicates('name', keep='last')
  return df



def _load_xray():
  df = _cluster_name_std(read_table(configs.XRAY_TABLE_PATH, comment='#'))
  return df



def _load_cluster_product(info: ClusterInfo, path: Path):
  if not path.exists(): 
    print(f'Table {path} not found!')
    return None
  t = Timming()
  df = read_table(path)
  print(f'loaded table {str(path)} in {t.end()}.')
  print(f'number of objects: {len(df)}')
  print('columns:')
  pprint(df.columns)
  if 'z' in df.columns:
    print(f'z range: [{df.z.min()}, {df.z.max()}]')
  if 'zml' in df.columns:
    print(f'zml range: [{df.zml.min()}, {df.zml.max()}]')
  print()
  return df



def load_catalog(version: int, subset: bool = False):
  version_map = {
    5: _load_clusters_v5,
    6: _load_catalog_v6,
    7: _load_catalog_v7,
  }
  df = version_map[version](subset)
  print('Columns:', ', '.join(df.columns))
  print('Rows:', len(df))
  return df



def load_shiftgap_cone(info: ClusterInfo, version: int):
  version_map = {
    5: _load_shiftgap_v5,
    6: _load_shiftgap_v6,
    7: _load_shiftgap_v7,
  }
  df_sg = version_map[version](info.name)
  df_members = df_sg[df_sg.flag_member == 0]
  df_interlopers = df_sg[df_sg.flag_member == 1]
  print('df_sg columns:', ', '.join(df_sg.columns))
  print('df_members columns:', ', '.join(df_members.columns))
  print('df_interlopers columns:', ', '.join(df_interlopers.columns))
  return df_sg, df_members, df_interlopers



def load_legacy_cone(info: ClusterInfo):
  return _load_cluster_product(info, info.legacy_path)



def load_photoz_cone(info: ClusterInfo):
  return _load_cluster_product(info, info.photoz_path)



def load_specz_cone(info: ClusterInfo):
  return _load_cluster_product(info, info.specz_path)



def load_combined_cone(info: ClusterInfo):
  return _load_cluster_product(info, info.compilation_path)



def load_mag_cone(info: ClusterInfo):
  return _load_cluster_product(info, configs.MAG_COMP_FOLDER)




def load_spec(coords: bool = True):
  df_spec = read_table(configs.SPEC_TABLE_PATH)
  if 'original_f_z' in df_spec.columns:
    df_spec['f_z_new'] = ''
    df_spec.loc[df_spec['f_z'] == 1, 'f_z_new'] = 'KEEP'
    del df_spec['f_z']
    df_spec = df_spec.rename(columns={'f_z_new': 'f_z'})
    del df_spec['original_f_z']
  if 'class' in df_spec.columns:
    if 'class_spec' in df_spec.columns:
      del df_spec['class_spec']
    df_spec = df_spec.rename(columns={'class': 'class_spec'})
  if 'original_class' in df_spec.columns:
    if 'original_class_spec' in df_spec.columns:
      del df_spec['original_class_spec']
    df_spec = df_spec.rename(columns={'original_class': 'original_class_spec'})
  df_spec['source'] = df_spec['source'].fillna('')
  ra, dec = guess_coords_columns(df_spec)
  df_spec = df_spec.rename(columns={ra: 'ra_spec_all', dec: 'dec_spec_all'})
  print('columns: ', ', '.join(df_spec.columns))
  if coords:
    skycoords = SkyCoord(
      ra=df_spec['ra_spec_all'].values, 
      dec=df_spec['dec_spec_all'].values, 
      unit=u.deg, 
      frame='icrs'
    )
    return df_spec, skycoords
  return df_spec, None



def compute_cluster_info(
  cls_name: str,
  z_spec_delta: float,
  z_photo_delta: float,
  plot_format: str,
  version: int,
  magnitude_range: Tuple[float, float],
  subset: bool,
) -> ClusterInfo:
  df_clusters = load_catalog(version, subset=subset)
  cluster = df_clusters[df_clusters.name == cls_name]
  if len(cluster) == 0:
    pprint(df_clusters['name'].values)
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
  
  info = ClusterInfo(
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
    z_photo_delta=z_photo_delta,
    z_spec_delta=z_spec_delta,
    z_photo_range=(z - z_photo_delta, z + z_photo_delta),
    z_spec_range=(z - z_spec_delta, z + z_spec_delta),
    plot_format=plot_format,
    version=version,
    magnitude_range=magnitude_range,
  )
  pprint(info)
  return info


if __name__ == '__main__':
  from pprint import pprint
  info = ClusterInfo(name='A168', ra=1.89, dec=69.90, z=0.10, search_radius_Mpc=15)
  pprint(info)
  pprint(info.compilation_path)