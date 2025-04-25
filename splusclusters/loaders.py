from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from astromodule.distance import mpc2arcsec
from astromodule.io import merge_pdf, read_table, write_table
from astromodule.pipeline import Pipeline, PipelineStage, PipelineStorage
from astromodule.table import (concat_tables, crossmatch, fast_crossmatch,
                               guess_coords_columns, radial_search, selfmatch)
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM

from splusclusters.configs import configs
from splusclusters.utils import Timming


def load_clusters():
  df_clusters = read_table(configs.MEMBERS_V5_PATH / 'index.dat')
  df_clusters['clsid'] = df_clusters['clsid'].astype('int')
  df_index = read_table(configs.ROOT / 'outputs_v5/paulo/G/index.dat', columns=['clsid', 'name'])
  df = df_clusters.set_index('clsid').join(df_index.set_index('clsid'), how='inner', rsuffix='_r')
  df_filter = read_table(configs.ROOT / 'tables/catalog_v6_splus_only_filter.csv')
  df = df[df.name.isin(df_filter.name)].copy().reset_index()
  return df

def load_index():
  df_index = read_table(configs.ROOT / 'outputs_v5/paulo/G/index.dat')
  return df_index

def load_members_v5(cls_id):
  path = configs.MEMBERS_V5_FOLDER / f'cluster.gals.sel.shiftgap.iter.{str(cls_id).zfill(5)}'
  col_names = [
    'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
    'radius_Mpc', 'v_offset', 'flag_member'
  ] # 0 - member; 1 - interloper
  return read_table(path, fmt='dat', col_names=col_names)

def load_members_index_v6():
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

def load_members_v6(cls_name: str = None, cls_id: str | int = None):
  if cls_name is not None:
    index_df = load_members_index_v6()
    cls_id = index_df[index_df.name == cls_name].clsid.values[0]
  prefixed_id = str(int(cls_id)).zfill(5)
  path = configs.MEMBERS_V6_FOLDER / f'cluster.gals.sel.shiftgap.iter.{prefixed_id}'
  cols = [
    'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
    'radius_Mpc', 'v_offset', 'flag_member'
  ]
  return read_table(path, fmt='dat', col_names=cols)

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
  if coords:
    ra, dec = guess_coords_columns(df_spec)
    coords = SkyCoord(
      ra=df_spec[ra].values, 
      dec=df_spec[dec].values, 
      unit=u.deg, 
      frame='icrs'
    )
    return df_spec, coords
  return df_spec

def load_photoz(coords: bool = True):
  df_photoz = read_table(configs.PHOTOZ_TABLE_PATH)
  if coords:
    ra, dec = guess_coords_columns(df_photoz)
    coords = SkyCoord(
      ra=df_photoz[ra].values, 
      dec=df_photoz[dec].values, 
      unit=u.deg, 
      frame='icrs'
    )
    return df_photoz, coords
  return df_photoz

def load_photoz2(coords: bool = True):
  df_photoz = read_table(configs.PHOTOZ2_TABLE_PATH)
  if coords:
    ra, dec = guess_coords_columns(df_photoz)
    coords = SkyCoord(
      ra=df_photoz[ra].values, 
      dec=df_photoz[dec].values, 
      unit=u.deg, 
      frame='icrs'
    )
    return df_photoz, coords
  return df_photoz.rename(columns={'RA': 'ra', 'DEC': 'dec'})

def load_legacy(coords: bool = True):
  df = read_table(configs.LEGACY_TABLE_PATH)
  if coords:
    ra, dec = guess_coords_columns(df)
    coords = SkyCoord(
      ra=df[ra].values, 
      dec=df[dec].values, 
      unit=u.deg, 
      frame='icrs'
    )
    return df, coords
  return df

def load_eRASS():
  df_erass = read_table(configs.ERASS_TABLE_PATH)
  return df_erass


def load_full_eRASS():
  df_full_eras = read_table(configs.FULL_ERASS_TABLE_PATH)
  return df_full_eras


def load_eRASS_2():
  return read_table(configs.ERASS2_TABLE_PATH)


def load_heasarc():
  df = read_table(configs.HEASARC_TABLE_PATH)
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
  df = df.drop_duplicates('name', keep='last')
  return df


def load_catalog_v6():
  df = read_table(configs.CATALOG_V6_TABLE_PATH, comment='#')
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
  df['clsid'] = list(range(1, len(df)+1))
  return df

def load_catalog_v6_old():
  df = read_table(configs.CATALOG_V6_OLD_TABLE_PATH, comment='#')
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

def load_catalog_v6_hydra():
  df = read_table(configs.CATALOG_V6_HYDRA_TABLE_PATH, comment='#')
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
  df['clsid'] = list(range(1, len(df)+1))
  return df


def load_xray():
  df = read_table(configs.XRAY_TABLE_PATH, comment='#')
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


class LoadHeasarcInfoStage(PipelineStage):
  products = [
    'cls_name', 'cls_z', 'cls_ra', 'cls_dec', 'cls_search_radius_deg',
    'cls_r500_Mpc', 'cls_r500_deg', 'cls_r200_Mpc', 'cls_r200_deg',
    'z_photo_range', 'z_spec_range', 'df_members', 'df_interlopers',
    'cls_search_radius_Mpc',
  ]
  
  def __init__(
    self,
    df_heasarc: pd.DataFrame,
    cls_name: str = None,
  ):
    self.df = df_heasarc
    self.cls_name = cls_name
    
  def run(self, cls_name: str = None):
    cls_name = cls_name or self.cls_name
    cluster = self.df[self.df.name == cls_name]
    ra = cluster['ra'].values[0]
    dec = cluster['dec'].values[0]
    z = cluster['redshift'].values[0]
    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
    if 'search_radius_Mpc' in self.df.columns:
      search_radius_Mpc = cluster['search_radius_Mpc'].values[0]
    else:
      search_radius_Mpc = 15
    search_radius_deg = min(mpc2arcsec(search_radius_Mpc, z, cosmo).to(u.deg).value, 17)
    print('Cluster Name:', cls_name)
    print(f'RA: {ra:.3f}, DEC: {dec:.3f}, z: {z:.2f}, search radius: {search_radius_deg:.2f}')
    return {
      'cls_name': cls_name,
      'cls_z': z,
      'cls_ra': ra,
      'cls_dec': dec,
      'cls_search_radius_Mpc': search_radius_Mpc,
      'cls_search_radius_deg': search_radius_deg,
      'cls_r500_Mpc': None,
      'cls_r500_deg': None,
      'cls_r200_Mpc': None,
      'cls_r200_deg': None,
      'z_photo_range': (z - configs.Z_PHOTO_DELTA, z + configs.Z_PHOTO_DELTA),
      'z_spec_range': (z - configs.Z_SPEC_DELTA, z + configs.Z_SPEC_DELTA),
      'df_members': None,
      'df_interlopers': None,
      'df_ret': None,
    }
    


class LoadClusterInfoStage(PipelineStage):
  products = [
    'cls_name', 'cls_z', 'cls_ra', 'cls_dec', 'cls_search_radius_deg',
    'cls_r500_Mpc', 'cls_r500_deg', 'cls_r200_Mpc', 'cls_r200_deg',
    'z_photo_range', 'z_spec_range', 'df_members', 'df_interlopers',
    'cls_search_radius_Mpc', 'df_ret'
  ]
  
  def __init__(
    self, 
    df_clusters: pd.DataFrame = None,
    cls_id: int = None, 
    version: int = 5,
  ):
    self.df_clusters = df_clusters
    self.cls_id = cls_id
    self.version = version
    
  def run(self, cls_id: int = None):
    df_clusters = self.df_clusters
    cls_id = cls_id or self.cls_id
    cluster = df_clusters[df_clusters.clsid == cls_id]
    name = cluster['name'].values[0]
    ra = cluster['ra'].values[0]
    dec = cluster['dec'].values[0]
    z = cluster['z_spec'].values[0]
    
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
    
    base_path = configs.MEMBERS_V5_FOLDER if self.version == 5 else configs.MEMBERS_V6_FOLDER
    ret_path = base_path / f'cluster.gals.sel.shiftgap.iter.{str(cls_id).zfill(5)}'
    print('ret_path:', str(ret_path))
    if ret_path.exists():
      col_names = [
        'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
        'radius_Mpc', 'v_offset', 'flag_member'
      ] # 0 - member; 1 - interloper
      df_ret = read_table(ret_path, fmt='dat', col_names=col_names)
      df_members = df_ret[df_ret.flag_member == 0]
      df_interlopers = df_ret[df_ret.flag_member == 1]
    else:
      df_members = None
      df_interlopers = None
      df_ret = None
    
    print(f'{df_ret = }')
    
    print('Cluster Name:', name)
    print(f'RA: {ra:.3f}, DEC: {dec:.3f}, z: {z:.2f}, search radius: {search_radius_deg:.2f}')
    print(f'R200: {r200_Mpc:.2f} ({r200_deg:.2f}), R500: {r500_Mpc:.2f} ({r500_deg:.2f})')
    print(f'{configs.Z_SPEC_DELTA = }; {configs.Z_PHOTO_DELTA = }')
    
    return {
      'cls_name': name,
      'cls_z': z,
      'cls_ra': ra,
      'cls_dec': dec,
      'cls_search_radius_Mpc': search_radius_Mpc,
      'cls_search_radius_deg': search_radius_deg,
      'cls_r500_Mpc': r500_Mpc,
      'cls_r500_deg': r500_deg,
      'cls_r200_Mpc': r200_Mpc,
      'cls_r200_deg': r200_deg,
      'z_photo_range': (z - configs.Z_PHOTO_DELTA, z + configs.Z_PHOTO_DELTA),
      'z_spec_range': (z - configs.Z_SPEC_DELTA, z + configs.Z_SPEC_DELTA),
      'df_members': df_members,
      'df_interlopers': df_interlopers,
      'df_ret': df_ret,
    }
    
    
class LoadERASSInfoStage(PipelineStage):
  products = [
    'cls_name', 'cls_z', 'cls_ra', 'cls_dec', 'cls_search_radius_deg',
    'cls_r500_Mpc', 'cls_r500_deg', 'cls_r200_Mpc', 'cls_r200_deg',
    'z_photo_range', 'z_spec_range', 'df_members', 'df_interlopers',
    'cls_search_radius_Mpc', 'df_ret'
  ]
  
  def __init__(self, df_clusters: pd.DataFrame):
    self.df_clusters = df_clusters
  
  def run(self, cls_name: str):
    df_clusters = self.df_clusters
    cluster = df_clusters[df_clusters.Cluster == cls_name]
    z = cluster['BEST_Z'].values[0]
    ra = cluster['RA_OPT'].values[0]
    dec = cluster['DEC_OPT'].values[0]
    r500_Mpc = cluster['R500_Mpc'].values[0]
    r500_deg = cluster['R500_deg'].values[0]
    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
    if 'search_radius_Mpc' in df_clusters.columns:
      search_radius_Mpc = cluster['search_radius_Mpc'].values[0]
    else:
      search_radius_Mpc = 15
    search_radius_deg = min(mpc2arcsec(search_radius_Mpc, z, cosmo).to(u.deg).value, 17)
    print('Cluster Name:', cls_name)
    print(f'RA: {ra:.3f}, DEC: {dec:.3f}, z: {z:.2f}, search radius: {search_radius_deg:.2f}')
    return {
      'cls_name': cls_name,
      'cls_z': z,
      'cls_ra': ra,
      'cls_dec': dec,
      'cls_search_radius_Mpc': search_radius_Mpc,
      'cls_search_radius_deg': search_radius_deg,
      'cls_r500_Mpc': r500_Mpc,
      'cls_r500_deg': r500_deg,
      'cls_r200_Mpc': None,
      'cls_r200_deg': None,
      'z_photo_range': (z - configs.Z_PHOTO_DELTA, z + configs.Z_PHOTO_DELTA),
      'z_spec_range': (z - configs.Z_SPEC_DELTA, z + configs.Z_SPEC_DELTA),
      'df_members': None,
      'df_interlopers': None,
      'df_ret': None,
    }
    
    

class LoadGenericInfoStage(PipelineStage):
  products = [
    'cls_name', 'cls_id', 'cls_z', 'cls_ra', 'cls_dec', 'cls_search_radius_deg',
    'cls_r500_Mpc', 'cls_r500_deg', 'cls_r200_Mpc', 'cls_r200_deg',
    'z_photo_range', 'z_spec_range', 'df_members', 'df_interlopers',
    'cls_search_radius_Mpc', 'df_ret'
  ]
  
  def __init__(self, df_clusters: pd.DataFrame):
    self.df_clusters = df_clusters
  
  def run(self, cls_name: str):
    df_clusters = self.df_clusters
    cluster = df_clusters[df_clusters.name == cls_name]
    cls_id = cluster['clsid'].values[0]
    z = cluster['z_spec'].values[0]
    ra = cluster['ra'].values[0]
    dec = cluster['dec'].values[0]
    # r500_Mpc = cluster['R500_Mpc'].values[0]
    # r500_deg = cluster['R500_deg'].values[0]
    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
    if 'search_radius_Mpc' in df_clusters.columns:
      search_radius_Mpc = cluster['search_radius_Mpc'].values[0]
    else:
      search_radius_Mpc = 15
    search_radius_deg = min(mpc2arcsec(search_radius_Mpc, z, cosmo).to(u.deg).value, 17)
    print('Cluster Name:', cls_name)
    print(f'RA: {ra:.3f}, DEC: {dec:.3f}, z: {z:.2f}, search radius: {search_radius_deg:.2f}')
    return {
      'cls_id': cls_id,
      'cls_name': cls_name,
      'cls_z': z,
      'cls_ra': ra,
      'cls_dec': dec,
      'cls_search_radius_Mpc': search_radius_Mpc,
      'cls_search_radius_deg': search_radius_deg,
      # 'cls_r500_Mpc': r500_Mpc,
      # 'cls_r500_deg': r500_deg,
      'cls_r500_Mpc': None,
      'cls_r500_deg': None,
      'cls_r200_Mpc': None,
      'cls_r200_deg': None,
      'z_photo_range': (z - configs.Z_PHOTO_DELTA, z + configs.Z_PHOTO_DELTA),
      'z_spec_range': (z - configs.Z_SPEC_DELTA, z + configs.Z_SPEC_DELTA),
      'df_members': None,
      'df_interlopers': None,
      'df_ret': None,
    }


class LoadPauloInfoStage(PipelineStage):
  products = [
    'cls_name', 'cls_z', 'cls_ra', 'cls_dec', 'cls_search_radius_deg',
    'cls_r500_Mpc', 'cls_r500_deg', 'cls_r200_Mpc', 'cls_r200_deg',
    'z_photo_range', 'z_spec_range', 'df_members', 'df_interlopers',
    'cls_search_radius_Mpc', 'df_ret',
  ]
  
  def __init__(
    self, 
    df_clusters: pd.DataFrame = None,
    cls_id: int = None, 
  ):
    self.df_clusters = df_clusters
    self.cls_id = cls_id
    
  def run(self, cls_id: int = None):
    df_clusters = self.df_clusters
    cls_id = cls_id or self.cls_id
    cluster = df_clusters[df_clusters.clsid == cls_id]
    ra_col, dec_col = guess_coords_columns(cluster)
    name = cluster['name'].values[0]
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
    
    df_class = load_clusters()
    df_members = None
    df_interlopers = None
    df_ret = None
    if name in df_class.name.values:
      _cls_id = df_class[df_class.name == name].clsid.values[0]
      paulo_path = configs.MEMBERS_V5_FOLDER / f'cluster.gals.sel.shiftgap.iter.{str(_cls_id).zfill(5)}'
      if paulo_path.exists():
        col_names = [
          'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
          'radius_Mpc', 'v_offset', 'flag_member'
        ] # 0 - member; 1 - interloper
        df_ret = read_table(paulo_path, fmt='dat', col_names=col_names)
        df_members = df_ret[df_ret.flag_member == 0]
        df_interlopers = df_ret[df_ret.flag_member == 1]
    
    print('Cluster Name:', name)
    print(f'RA: {ra:.3f}, DEC: {dec:.3f}, z: {z:.2f}, search radius: {search_radius_deg:.2f}')
    
    return {
      'cls_name': name,
      'cls_z': z,
      'cls_ra': ra,
      'cls_dec': dec,
      'cls_search_radius_Mpc': search_radius_Mpc,
      'cls_search_radius_deg': search_radius_deg,
      'cls_r500_Mpc': r500_Mpc,
      'cls_r500_deg': r500_deg,
      'cls_r200_Mpc': r200_Mpc,
      'cls_r200_deg': r200_deg,
      'z_photo_range': (z - configs.Z_PHOTO_DELTA, z + configs.Z_PHOTO_DELTA),
      'z_spec_range': (z - configs.Z_SPEC_DELTA, z + configs.Z_SPEC_DELTA),
      'df_members': df_members,
      'df_interlopers': df_interlopers,
      'df_ret': df_ret,
    }


    
class LoadERASS2InfoStage(PipelineStage):
  products = [
    'cls_name', 'cls_z', 'cls_ra', 'cls_dec', 'cls_search_radius_deg', 
    'cls_search_radius_Mpc', 'cls_r500_Mpc', 'cls_r500_deg', 'cls_r200_Mpc', 
    'cls_r200_deg', 'z_photo_range', 'z_spec_range', 'df_members', 
    'df_interlopers', 'df_ret'
  ]
  
  def __init__(self, df_clusters: pd.DataFrame):
    self.df_clusters = df_clusters
  
  def run(self, cls_name: str):
    df_clusters = self.df_clusters
    cluster = df_clusters[df_clusters.NAME == cls_name]
    z = cluster['BEST_Z'].values[0]
    ra = cluster['RA_XFIT'].values[0]
    dec = cluster['DEC_XFIT'].values[0]
    r500_Mpc = cluster['R500'].values[0] * 1e-3 # kpc -> Mpc
    # r500_deg = cluster['R500_deg'].values[0]
    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
    r500_deg = mpc2arcsec(r500_Mpc, z, cosmo).to(u.deg).value
    if 'search_radius_Mpc' in df_clusters.columns:
      search_radius_Mpc = cluster['search_radius_Mpc'].values[0]
    else:
      search_radius_Mpc = 15
    search_radius_deg = min(mpc2arcsec(search_radius_Mpc, z, cosmo).to(u.deg).value, 17)
    print('Cluster Name:', cls_name)
    print(f'RA: {ra:.3f}, DEC: {dec:.3f}, z: {z:.2f}, search radius: {search_radius_deg:.2f}')
    return {
      'cls_name': cls_name,
      'cls_z': z,
      'cls_ra': ra,
      'cls_dec': dec,
      'cls_search_radius_Mpc': search_radius_Mpc,
      'cls_search_radius_deg': search_radius_deg,
      'cls_r500_Mpc': r500_Mpc,
      'cls_r500_deg': r500_deg,
      'cls_r200_Mpc': None,
      'cls_r200_deg': None,
      'z_photo_range': (z - configs.Z_PHOTO_DELTA, z + configs.Z_PHOTO_DELTA),
      'z_spec_range': (z - configs.Z_SPEC_DELTA, z + configs.Z_SPEC_DELTA),
      'df_members': None,
      'df_interlopers': None,
      'df_ret': None,
    }



def remove_bad_objects(df: pd.DataFrame):
  filter_df = read_table(configs.ROOT / 'tables' / 'objects_to_exclude.csv', comment='#')
  return crossmatch(df, filter_df, radius=1*u.arcsec, join='1not2')



class PrepareCatalogToSubmitStage(PipelineStage):
  def __init__(self, overwrite: bool = True):
    super().__init__()
    self.overwrite = overwrite
    
  def run(self, cls_id: int, df_all_radial: pd.DataFrame, z_spec_range: Tuple[float, float]):
    clusters_path = configs.SUBMIT_FOLDER / 'clusters'
    clusters_path.mkdir(parents=True, exist_ok=True)
    out_path = clusters_path / f'cluster_{str(cls_id).zfill(4)}.dat'
    if out_path.exists() and not self.overwrite:
      return
    
    df_submit = df_all_radial[(~df_all_radial.z.isna()) & df_all_radial.z.between(*z_spec_range)]
    df_submit = df_submit.reset_index(drop=True).copy(deep=True)
    del df_submit['field'] # causa desalinhamento na tabela, união com objs sem SPLUS
    # df_submit['ls10_photo'] = np.zeros(shape=((len(df_submit),)), dtype=int)
    # df_submit.ls10_photo[~df_submit.mag_r.isna()] = 1
    df_submit['class_spec'] = df_submit.class_spec.str.replace(' ', '')
    df_submit.loc[df_submit.class_spec == '', 'class_spec'] = '-999'
    df_submit.loc[df_submit.type == '', 'type'] = '-999'
    df_submit.loc[df_submit.f_z == '', 'f_z'] = '-999'
    df_submit.loc[df_submit.f_z == 'None', 'f_z'] = '-999'
    df_submit['ls10_photo'] = (~df_submit['mag_r'].isna()).astype(int)
    df_submit = df_submit.fillna(-999)
    df_submit = df_submit.rename(columns={
      'ra': 'RA',
      'dec': 'DEC',
      'z': 'zspec',
      'e_z': 'zspec-err',
      'f_z': 'zspec-flag',
      'zml': 'z_phot',
      'odds': 'z_phot_odds',
      'mag_r': 'ls10_r',
      'type': 'ls10_morpho',
      'class_spec': 'zspec-class',
    })
    objects_before = len(df_submit)
    df_submit = remove_bad_objects(df_submit)
    print('Inspection filter:', objects_before, '->', len(df_submit))
    write_table(df_submit, out_path)




class LoadDataFrameStage(PipelineStage):
  def __init__(self, key: str, base_path: str | Path):
    self.key = key
    self.base_path = base_path
    self.products = [key]
    
  def run(self, cls_name: str, z_spec_range: Tuple[float, float], cls_z: float):
    t = Timming()
    df = read_table(self.base_path / f'{cls_name}.parquet')
    if 'z' in df.columns:
      z_delta1 = cls_z - df.z.min()
      z_delta2 = df.z.max() - cls_z
      print(f'z range: [{cls_z:.3f} - {z_delta1:.3f}, {cls_z:.3f} + {z_delta2:.3f}]')
    if 'zml' in df.columns:
      zml_delta1 = cls_z - df.zml.min()
      zml_delta2 = df.zml.max() - cls_z
      print(f'z range: [{cls_z:.3f} - {zml_delta1:.3f}, {cls_z:.3f} + {zml_delta2:.3f}]')
    # if 'z' in df.columns:
    #   df = df[df.z.between(*z_spec_range)].reset_index(drop=True)
    print(f'Table loaded. Duration: {t.end()}. Number of objects: {len(df)}')
    return {self.key: df}


class LoadLegacyRadialStage(LoadDataFrameStage):
  def __init__(self):
    super().__init__('df_legacy_radial', configs.LEG_PHOTO_FOLDER)


class LoadPhotozRadialStage(LoadDataFrameStage):
  def __init__(self):
    super().__init__('df_photoz_radial', configs.PHOTOZ_FOLDER)
  
  
class LoadSpeczRadialStage(LoadDataFrameStage):
  def __init__(self):
    super().__init__('df_specz_radial', configs.SPECZ_FOLDER)


class LoadAllRadialStage(LoadDataFrameStage):
  def __init__(self):
    super().__init__('df_all_radial', configs.PHOTOZ_SPECZ_LEG_FOLDER)
    

class LoadMagRadialStage(LoadDataFrameStage):
  def __init__(self):
    super().__init__('df_mag_radial', configs.MAG_COMP_FOLDER)