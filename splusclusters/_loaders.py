from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
from prefect import flow, task
from prefect.logging import get_run_logger
from pylegs.io import read_table, write_table

import luigi
from splusclusters._info import ClusterInfo, cluster_params
from splusclusters.configs import configs
from splusclusters.utils import Timming


def load_shiftgap_index_v5():
  df_index = read_table(configs.MEMBERS_V5_PATH / 'names.dat')
  return df_index


def load_shiftgap_v5(cls_id):
  path = configs.MEMBERS_V5_FOLDER / f'cluster.gals.sel.shiftgap.iter.{str(cls_id).zfill(5)}'
  col_names = [
    'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
    'radius_Mpc', 'v_offset', 'flag_member'
  ] # 0 - member; 1 - interloper
  return read_table(path, fmt='dat', col_names=col_names)


def load_shiftgap_index_v6():
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


def load_shiftgap_v6(cls_name: str = None, cls_id: str | int = None):
  if cls_name is not None:
    index_df = load_shiftgap_index_v6()
    cls_id = index_df[index_df.name == cls_name].clsid.values[0]
  prefixed_id = str(int(cls_id)).zfill(5)
  path = configs.MEMBERS_V6_FOLDER / f'cluster.gals.sel.shiftgap.iter.{prefixed_id}'
  cols = [
    'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
    'radius_Mpc', 'v_offset', 'flag_member'
  ]
  return read_table(path, fmt='dat', col_names=cols)


def load_shiftgap_index_v7():
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


def load_shiftgap_v7(cls_name: str = None, cls_id: str | int = None):
  if cls_name is not None:
    index_df = load_shiftgap_index_v6()
    cls_id = index_df[index_df.name == cls_name].clsid.values[0]
  prefixed_id = str(int(cls_id)).zfill(5)
  path = configs.MEMBERS_V7_FOLDER / f'cluster.gals.sel.shiftgap.iter.{prefixed_id}'
  cols = [
    'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
    'radius_Mpc', 'v_offset', 'flag_member'
  ]
  return read_table(path, fmt='dat', col_names=cols)


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


def load_heasarc():
  df = _cluster_name_std(read_table(configs.HEASARC_TABLE_PATH))
  df = df.drop_duplicates('name', keep='last')
  return df


def load_clusters_v5():
  df_clusters = read_table(configs.MEMBERS_V5_PATH / 'index.dat')
  df_clusters['clsid'] = df_clusters['clsid'].astype('int')
  df_index = read_table(configs.MEMBERS_V5_PATH / 'names.dat', columns=['clsid', 'name'])
  df = df_clusters.set_index('clsid').join(df_index.set_index('clsid'), how='inner', rsuffix='_r')
  df_filter = read_table(configs.ROOT / 'tables/catalog_v6_splus_only_filter.csv')
  df = df[df.name.isin(df_filter.name)].copy().reset_index()
  return df


def load_catalog_v6():
  df = _cluster_name_std(read_table(configs.CATALOG_V6_TABLE_PATH, comment='#'))
  df['clsid'] = list(range(1, len(df)+1))
  return df


def load_catalog_v6_old():
  df = _cluster_name_std(read_table(configs.CATALOG_V6_OLD_TABLE_PATH, comment='#'))
  return df


def load_catalog_v6_hydra():
  df = _cluster_name_std(read_table(configs.CATALOG_V6_HYDRA_TABLE_PATH, comment='#'))
  df['clsid'] = list(range(1, len(df)+1))
  return df


def load_catalog_v7():
  return load_shiftgap_index_v7()[['name', 'ra', 'dec', 'z_spec']].rename({'z_spec': 'zspec'})


def load_xray():
  df = _cluster_name_std(read_table(configs.XRAY_TABLE_PATH, comment='#'))
  return df



def _load_cluster_product(info: ClusterInfo, base_path: Path):
  path = base_path / f'{info.name}.parquet'
  if not path.exists(): 
    get_run_logger().warning(f'Table {path} not found!')
    return None
  t = Timming()
  df = read_table(path)
  if 'z' in df.columns:
    z_delta1 = info.z - df.z.min()
    z_delta2 = df.z.max() - info.z
    print(f'z range: [{info.z:.3f} - {z_delta1:.3f}, {info.z:.3f} + {z_delta2:.3f}]')
  if 'zml' in df.columns:
    zml_delta1 = info.z - df.zml.min()
    zml_delta2 = df.zml.max() - info.z
    print(f'z range: [{info.z:.3f} - {zml_delta1:.3f}, {info.z:.3f} + {zml_delta2:.3f}]')
  # if 'z' in df.columns:
  #   df = df[df.z.between(*z_spec_range)].reset_index(drop=True)
  print(f'Table loaded. Duration: {t.end()}. Number of objects: {len(df)}')
  return df


@task(task_run_name='load-legacy-cone-{info.name}', version='1.0', persist_result=False)
def load_legacy_cone(info: ClusterInfo):
  return _load_cluster_product(info, configs.LEG_PHOTO_FOLDER)


@task(task_run_name='load-photoz-cone-{info.name}', version='1.0', persist_result=False)
def load_photoz_cone(info: ClusterInfo):
  return _load_cluster_product(info, configs.PHOTOZ_FOLDER)


@task(task_run_name='load-specz-cone-{info.name}', version='1.0', persist_result=False)
def load_specz_cone(info: ClusterInfo):
  return _load_cluster_product(info, configs.SPECZ_FOLDER)


@task(task_run_name='load-all-cone-{info.name}', version='1.0', persist_result=False)
def load_all_cone(info: ClusterInfo):
  return _load_cluster_product(info, configs.PHOTOZ_SPECZ_LEG_FOLDER)


@task(task_run_name='load-mag-cone-{info.name}', version='1.0', persist_result=False)
def load_mag_cone(info: ClusterInfo):
  return _load_cluster_product(info, configs.MAG_COMP_FOLDER)



@task(task_run_name='load-catalog-v{version}', version='1.0', persist_result=False)
def load_catalog(version: int):
  version_map = {
    5: load_clusters_v5,
    6: load_catalog_v6,
    7: load_catalog_v7,
  }
  return version_map[version]()



@task(task_run_name='load-shiftgap-v{version}-{cls_name}', version='1.0')
def load_shiftgap_tables(cls_name: str, version: int):
  version_map = {
    5: load_shiftgap_v5,
    6: load_shiftgap_v6,
    7: load_shiftgap_v7,
  }
  df_sg = version_map[version](cls_name)
  df_members = df_sg[df_sg.flag_member == 0]
  df_interlopers = df_sg[df_sg.flag_member == 1]
  return df_sg, df_members, df_interlopers



@task(task_run_name='load-spec', version='1.1', persist_result=False)
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
  return df_spec


@dataclass
class ConesContainer:
  photoz: pd.DataFrame
  specz: pd.DataFrame
  legacy: pd.DataFrame
  shiftgap: pd.DataFrame
  interlopers: pd.DataFrame
  members: pd.DataFrame


@flow(flow_run_name='load-all-cones-{info.name}', version='1.0', persist_result=False)
def load_cones(info: ClusterInfo, version: int) -> ConesContainer:
  df_shiftgap, df_members, df_interlopers = load_shiftgap_tables(
    cls_name=info.name, 
    version=version,
  )
  df_cone_photoz = load_photoz_cone(info)
  df_cone_specz = load_specz_cone(info)
  df_cone_legacy = load_legacy_cone(info)
  
  return ConesContainer(
    photoz=df_cone_photoz,
    specz=df_cone_specz,
    legacy=df_cone_legacy,
    shiftgap=df_shiftgap,
    interlopers=df_interlopers,
    members=df_members,
  )



class LoadClusterCatalog(luigi.Task):
  version = luigi.IntParameter()
  
  def output(self):
    return luigi.LocalTarget(configs.LUIGI_FOLDER / f'cluster-catalog-v{self.version}.pckl')
  
  def run(self):
    import pickle
    with self.output().open('w') as f:
      pickle.dump(load_catalog(self.version), f)
      



@dg.op(out={'specz_df': dg.Out(pd.DataFrame), 'specz_skycoord': dg.Out(SkyCoord)})
def dg_load_spec(version: int):
  return load_spec(version)


@dg.op
def dg_load_cluster_catalog(version: int):
  return load_catalog(version)


@dg.op
def dg_cluster_info(cls_name: str, version: int):
  df_clusters = load_catalog(version)
  return cluster_params(df_clusters, cls_name)