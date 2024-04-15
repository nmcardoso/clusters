from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import matplotlib.pyplot as plt
import mpl_scatter_density
import numpy as np
import pandas as pd
from astromodule.distance import mpc2arcsec
from astromodule.io import merge_pdf, read_table, write_table
from astromodule.legacy import LegacyService
from astromodule.pipeline import Pipeline, PipelineStage, PipelineStorage
from astromodule.table import (concat_tables, crossmatch, fast_crossmatch,
                               guess_coords_columns, radial_search, selfmatch)
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from tqdm import tqdm

PHOTOZ_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/idr5_photoz_clean.parquet')
SPEC_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/SpecZ_Catalogue_20240124.parquet')
ERASS_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/liana_erass.csv')
FULL_ERASS_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/eRASS1_min.parquet')
HEASARC_TABLE_PATH = Path('public/heasarc_all.parquet')
TABLES_PATH = Path('clusters_members')
MEMBERS_FOLDER = Path('clusters_members/clusters')
OUT_PATH = Path('outputs_v6')
WEBSITE_PATH = Path('docs')
PLOTS_FOLDER = OUT_PATH / 'plots'
VELOCITY_PLOTS_FOLDER = OUT_PATH / 'velocity_plots'
MAGDIFF_PLOTS_FOLDER = OUT_PATH / 'magdiff_plots'
MAGDIFF_OUTLIERS_FOLDER = OUT_PATH / 'magdiff_outliers'
LEG_PHOTO_FOLDER = OUT_PATH / 'legacy'
PHOTOZ_FOLDER = OUT_PATH / 'photoz'
SPECZ_FOLDER = OUT_PATH / 'specz'
PHOTOZ_SPECZ_LEG_FOLDER = OUT_PATH / 'photoz+specz+legacy'
MAG_COMP_FOLDER = OUT_PATH / 'mag_comp'
Z_PHOTO_DELTA = 0.015
Z_SPEC_DELTA = 0.007
MAG_RANGE = (13, 22)


class Timming:
  def __init__(self, start: bool = True):
    self.start_time = None
    self.end_time = None
    if start:
      self.start()


  def __repr__(self) -> str:
    return self.duration()


  def start(self):
    self.start_time = datetime.now()


  def end(self) -> str:
    self.end_time = datetime.now()
    return self.duration()


  def duration(self) -> str:
    if not self.end_time:
      duration = self.end_time - self.start_time
    else:
      end_time = datetime.now()
      duration = end_time - self.start_time

    return self._format_time(duration)


  def _format_time(self, dt: timedelta) -> str:
    hours, remainder = divmod(dt.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))




def load_clusters():
  df_clusters = read_table(TABLES_PATH / 'index.dat')
  df_clusters['clsid'] = df_clusters['clsid'].astype('int')
  df_index = read_table('outputs_v5/paulo/G/index.dat', columns=['clsid', 'name'])
  df = df_clusters.set_index('clsid').join(df_index.set_index('clsid'), how='inner', rsuffix='_r')
  df_filter = read_table('tables/catalog_v6_splus_only_filter.csv')
  df = df[df.name.isin(df_filter.name)].copy().reset_index()
  return df

def load_index():
  df_index = read_table('outputs_v5/paulo/G/index.dat')
  return df_index

def load_spec(coords: bool = True):
  df_spec = read_table(SPEC_TABLE_PATH)
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
  df_photoz = read_table(PHOTOZ_TABLE_PATH)
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

def load_eRASS():
  df_erass = read_table(ERASS_TABLE_PATH)
  return df_erass


def load_full_eRASS():
  df_full_eras = read_table(FULL_ERASS_TABLE_PATH)
  return df_full_eras


def load_heasarc():
  df = read_table(HEASARC_TABLE_PATH)
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



class LoadHeasarcInfoStage(PipelineStage):
  products = [
    'cls_name', 'cls_z', 'cls_ra', 'cls_dec', 'cls_15Mpc_deg',
    'cls_r500_Mpc', 'cls_r500_deg', 'cls_r200_Mpc', 'cls_r200_deg',
    'z_photo_range', 'z_spec_range', 'df_members', 'df_interlopers',
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
    r15Mpc_deg = mpc2arcsec(15, z).to(u.deg).value
    print('Cluster Name:', cls_name)
    print(f'RA: {ra:.3f}, DEC: {dec:.3f}, z: {z:.2f}, search radius: {r15Mpc_deg:.2f}')
    return {
      'cls_name': cls_name,
      'cls_z': z,
      'cls_ra': ra,
      'cls_dec': dec,
      'cls_15Mpc_deg': r15Mpc_deg,
      'cls_r500_Mpc': None,
      'cls_r500_deg': None,
      'cls_r200_Mpc': None,
      'cls_r200_deg': None,
      'z_photo_range': (z - Z_PHOTO_DELTA, z + Z_PHOTO_DELTA),
      'z_spec_range': (z - Z_SPEC_DELTA, z + Z_SPEC_DELTA),
      'df_members': None,
      'df_interlopers': None,
    }
    


class LoadClusterInfoStage(PipelineStage):
  products = [
    'cls_name', 'cls_z', 'cls_ra', 'cls_dec', 'cls_15Mpc_deg',
    'cls_r500_Mpc', 'cls_r500_deg', 'cls_r200_Mpc', 'cls_r200_deg',
    'z_photo_range', 'z_spec_range', 'df_members', 'df_interlopers',
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
    name = cluster['name'].values[0]
    ra = cluster['ra'].values[0]
    dec = cluster['dec'].values[0]
    z = cluster['z_spec'].values[0]
    r500_Mpc = cluster['R500_Mpc'].values[0]
    r200_Mpc = cluster['R200_Mpc'].values[0]
    r500_deg = mpc2arcsec(r500_Mpc, z).to(u.deg).value
    r200_deg = mpc2arcsec(r200_Mpc, z).to(u.deg).value
    r15Mpc_deg = mpc2arcsec(15, z).to(u.deg).value
    if r15Mpc_deg > 17:
      print(f'Cluster angular radius @ 15Mpc = {r15Mpc_deg:.2f} deg, limiting to 17 deg')
      r15Mpc_deg = min(r15Mpc_deg, 17)
    paulo_path = MEMBERS_FOLDER / f'cluster.gals.sel.shiftgap.iter.{str(cls_id).zfill(5)}'
    col_names = [
      'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
      'radius_Mpc', 'v_offset', 'flag_member'
    ] # 0 - member; 1 - interloper
    df_paulo = read_table(paulo_path, fmt='dat', col_names=col_names)
    df_members = df_paulo[df_paulo.flag_member == 0]
    df_interlopers = df_paulo[df_paulo.flag_member == 1]
    print('Cluster Name:', name)
    print(f'RA: {ra:.3f}, DEC: {dec:.3f}, z: {z:.2f}, search radius: {r15Mpc_deg:.2f}')
    return {
      'cls_name': name,
      'cls_z': z,
      'cls_ra': ra,
      'cls_dec': dec,
      'cls_15Mpc_deg': r15Mpc_deg,
      'cls_r500_Mpc': r500_Mpc,
      'cls_r500_deg': r500_deg,
      'cls_r200_Mpc': r200_Mpc,
      'cls_r200_deg': r200_deg,
      'z_photo_range': (z - Z_PHOTO_DELTA, z + Z_PHOTO_DELTA),
      'z_spec_range': (z - Z_SPEC_DELTA, z + Z_SPEC_DELTA),
      'df_members': df_members,
      'df_interlopers': df_interlopers,
    }
    
    
class LoadERASSInfoStage(PipelineStage):
  products = [
    'cls_name', 'cls_z', 'cls_ra', 'cls_dec', 'cls_15Mpc_deg',
    'cls_r500_Mpc', 'cls_r500_deg', 'cls_r200_Mpc', 'cls_r200_deg',
    'z_photo_range', 'z_spec_range', 'df_members', 'df_interlopers',
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
    r15Mpc_deg = min(mpc2arcsec(15, z).to(u.deg).value, 17)
    print('Cluster Name:', cls_name)
    print(f'RA: {ra:.3f}, DEC: {dec:.3f}, z: {z:.2f}, search radius: {r15Mpc_deg:.2f}')
    return {
      'cls_name': cls_name,
      'cls_z': z,
      'cls_ra': ra,
      'cls_dec': dec,
      'cls_15Mpc_deg': r15Mpc_deg,
      'cls_r500_Mpc': r500_Mpc,
      'cls_r500_deg': r500_deg,
      'cls_r200_Mpc': None,
      'cls_r200_deg': None,
      'z_photo_range': (z - Z_PHOTO_DELTA, z + Z_PHOTO_DELTA),
      'z_spec_range': (z - Z_SPEC_DELTA, z + Z_SPEC_DELTA),
      'df_members': None,
      'df_interlopers': None,
    }


class LoadDataFrameStage(PipelineStage):
  def __init__(self, key: str, base_path: str | Path):
    self.key = key
    self.base_path = base_path
    self.products = [key]
    
  def run(self, cls_name: str):
    t = Timming()
    df = read_table(self.base_path / f'{cls_name}.parquet')
    print(f'Table loaded. Duration: {t.end()}. Number of objects: {len(df)}')
    return {self.key: df}


class LoadLegacyRadialStage(LoadDataFrameStage):
  def __init__(self):
    super().__init__('df_legacy_radial', LEG_PHOTO_FOLDER)


class LoadPhotozRadialStage(LoadDataFrameStage):
  def __init__(self):
    super().__init__('df_photoz_radial', PHOTOZ_FOLDER)
  
  
class LoadSpeczRadialStage(LoadDataFrameStage):
  def __init__(self):
    super().__init__('df_specz_radial', SPECZ_FOLDER)


class LoadAllRadialStage(LoadDataFrameStage):
  def __init__(self):
    super().__init__('df_all_radial', PHOTOZ_SPECZ_LEG_FOLDER)
    

class LoadMagRadialStage(LoadDataFrameStage):
  def __init__(self):
    super().__init__('df_mag_radial', MAG_COMP_FOLDER)


class RadialSearchStage(PipelineStage):
  def __init__(
    self, 
    df_name: str,
    radius_key: str, 
    save_folder: str | Path,
    kind: Literal['spec', 'photo'],
    overwrite: bool = False,
    skycoord_name: str = None,
  ):
    self.df_name = df_name
    self.radius_key = radius_key
    self.save_folder = Path(save_folder)
    self.overwrite = overwrite
    self.kind = kind
    self.skycoord_name = skycoord_name
    self.save_folder.mkdir(parents=True, exist_ok=True)
    
  def run(
    self, 
    cls_ra: float, 
    cls_dec: float, 
    cls_name: str, 
    z_spec_range: Tuple[float, float],
  ):
    out_path = self.save_folder / f'{cls_name}.parquet'
    if not self.overwrite and out_path.exists():
      if self.get_data('cls_15Mpc_deg') < 10.17:
        return
    
    radius = self.get_data(self.radius_key)
    t = Timming()
    print(f'Starting radial search with radius: {radius:.2f} deg')
    pos = SkyCoord(ra=cls_ra, dec=cls_dec, unit=u.deg, frame='icrs')
    df_search = radial_search(
      position=pos, 
      table=self.get_data(self.df_name), 
      radius=radius*u.deg,
      cached_catalog=self.get_data(self.skycoord_name),
    )
    
    if self.kind == 'spec':
      df_search = df_search[
        df_search.z.between(*z_spec_range) &
        df_search.class_spec.str.startswith('GALAXY')
      ]
    elif self.kind == 'photo':
      df_search = df_search[
        # df_search.zml.between(*z_photo_range) &
        df_search.r_auto.between(*MAG_RANGE)
      ]
    
    print(f'Radial search finished. Elapsed time: {t.end()}')
    
    if self.save_folder:
      table_name = f'{cls_name}.parquet'
      write_table(df_search, self.save_folder / table_name)
      print(f'Table "{table_name}" saved')



class SpecZRadialSearchStage(RadialSearchStage):
  def __init__(
    self, 
    save_folder: str | Path = SPECZ_FOLDER, 
    radius_key: str = 'cls_15Mpc_deg', 
    overwrite: bool = False,
  ):
    super().__init__(
      df_name='df_spec',
      radius_key=radius_key, 
      save_folder=save_folder, 
      kind='spec', 
      overwrite=overwrite, 
      skycoord_name='specz_skycoord',
    )
  

class PhotoZRadialSearchStage(RadialSearchStage):
  def __init__(
    self, 
    save_folder: str | Path = PHOTOZ_FOLDER, 
    radius_key: str = 'cls_15Mpc_deg', 
    overwrite: bool = False,
  ):
    super().__init__(
      df_name='df_photoz',
      radius_key=radius_key, 
      save_folder=save_folder, 
      kind='photo', 
      overwrite=overwrite, 
      skycoord_name='photoz_skycoord',
    )




class FastCrossmatchStage(PipelineStage):
  def __init__(
    self, 
    left_table: str,
    right_table: str, 
    out_key: str,
    join: Literal['left', 'inner'] = 'inner'
  ):
    self.left_table = left_table
    self.right_table = right_table
    self.out_key = out_key
    self.join = join
    self.products = [out_key]
    
  def run(self):
    df_left = self.get_data(self.left_table)
    df_right = self.get_data(self.right_table)
    df_match = fast_crossmatch(df_left, df_right, join=self.join)
    return {self.out_key: df_match}



class StarsRemovalStage(PipelineStage):
  products = ['df_photoz_radial']
  def run(self, df_photoz_radial: pd.DataFrame, df_legacy_radial: pd.DataFrame):
    df_legacy_gal = df_legacy_radial[df_legacy_radial.type != 'PSF']
    df = fast_crossmatch(df_photoz_radial, df_legacy_gal, include_sep=False)
    return {'df_photoz_radial': df}



class DownloadLegacyCatalogStage(PipelineStage):
  def __init__(self, radius_key: str, overwrite: bool = False, workers: int = 3):
    self.radius_key = radius_key
    self.overwrite = overwrite
    self.workers = workers
    
  def run(self, cls_ra: float, cls_dec: float, cls_name: str):
    out_path = LEG_PHOTO_FOLDER / f'{cls_name}.parquet'
    if not self.overwrite and out_path.exists():
      if self.get_data('cls_15Mpc_deg') < 10.17:
        return
    
    sql = """
      SELECT t.ra, t.dec, t.type, t.mag_r
      FROM ls_dr10.tractor AS t
      WHERE (ra BETWEEN {ra_min} AND {ra_max}) AND 
      (dec BETWEEN {dec_min} AND {dec_max}) AND 
      (brick_primary = 1) AND 
      (mag_r BETWEEN {r_min:.2f} AND {r_max:.2f})
    """.strip()
    
    radius = self.get_data(self.radius_key)
    queries = [
      sql.format(
        ra_min=cls_ra-radius, 
        ra_max=cls_ra+radius, 
        dec_min=cls_dec-radius,
        dec_max=cls_dec+radius,
        r_min=_r,
        r_max=_r+.05
      )
      for _r in np.arange(*MAG_RANGE, .05)
    ]
    service = LegacyService(replace=True, workers=self.workers)
    service.batch_sync_query(
      queries=queries, 
      save_paths=out_path, 
      join_outputs=True, 
      workers=self.workers
    )



class PhotozSpeczLegacyMatchStage(PipelineStage):
  def __init__(self, overwrite: bool = False):
    self.overwrite = overwrite
    
  def run(
    self, 
    cls_name: str, 
    df_specz_radial: pd.DataFrame,
    df_photoz_radial: pd.DataFrame, 
    df_legacy_radial: pd.DataFrame
  ):
    out_path = PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}.parquet'
    if out_path.exists() and not self.overwrite:
      if self.get_data('cls_15Mpc_deg') < 10.17:
        return
    
    df_specz_radial['f_z'] = df_specz_radial['f_z'].astype('str')
    df_specz_radial['original_class_spec'] = df_specz_radial['original_class_spec'].astype('str')
    
    print('Photo-z objects:', len(df_photoz_radial))
    print('Spec-z objects:', len(df_specz_radial))
    print('Legacy objects:', len(df_legacy_radial))
    print('Starting first crossmatch: photo-z UNION spec-z')
    
    t = Timming()
    df = crossmatch(
      table1=df_photoz_radial,
      table2=df_specz_radial,
      join='1or2',
    )
    df['ra_1'].fillna(df['RA_2'], inplace=True)
    df['dec_1'].fillna(df['DEC_2'], inplace=True)
    
    print(f'First crossmatch finished. Duration: {t.end()}')
    print('Objects with photo-z only:', len(df[~df.zml.isna() & df.z.isna()]))
    print('Objects with spec-z only:', len(df[df.zml.isna() & ~df.z.isna()]))
    print('Objects with photo-z and spec-z:', len(df[~df.zml.isna() & ~df.z.isna()]))
    print('Total of objects after first match:', len(df))
    print('starting second crossmatch: match-1 LEFT OUTER JOIN legacy')
    
    t = Timming()
    df = crossmatch(
      table1=df,
      table2=df_legacy_radial,
      join='all1',
      ra1='ra_1',
      dec1='dec_1',
    )
    
    print(f'Second crossmatch finished. Duration: {t.end()}')
    print('Objects with legacy:', len(df[~df.type.isna()]))
    print('Objects without legacy:', len(df[df.type.isna()]))
    print('Galaxies:', len(df[df.type != 'PSF']), ', Stars:', len(df[df.type == 'PSF']))
    print('Total of objects after second match:', len(df))
    
    del df['ra'] # legacy ra
    del df['dec'] # legacy dec
    df = df.rename(columns={'ra_1': 'ra', 'dec_1': 'dec'}) # use photoz ra/dec
    
    photoz_cols = ['ra', 'dec', 'zml', 'odds', 'r_auto', 'field']
    specz_cols = ['z', 'e_z', 'f_z', 'class_spec']
    legacy_cols = ['mag_r', 'type']
    cols = photoz_cols + specz_cols + legacy_cols
    df = df[cols]
    
    write_table(df, out_path)

    


def get_plot_title(
  cls_name: str,
  cls_ra: float,
  cls_dec: float,
  cls_z: float,
  cls_15Mpc_deg: float,
  z_spec_range: Tuple[float, float],
  z_photo_range: Tuple[float, float],
):
  return (
    f'Cluster: {cls_name} (RA: {cls_ra:.5f}, DEC: {cls_dec:.5f})\n'
    f'Search Radius: 15Mpc = {cls_15Mpc_deg:.3f}$^\\circ$ ($z_{{cluster}}={cls_z:.4f}$)\n'
    f'Spec Z Range: $z_{{cluster}} \pm 0.007$ = [{z_spec_range[0]:.4f}, {z_spec_range[1]:.4f}]\n'
    f'Good Photo Z: $z_{{cluster}} \pm 0.015$ = [{z_photo_range[0]:.4f}, {z_photo_range[1]:.4f}]\n'
    f'R Mag Range: [13, 22] $\cdot$ Spec Class = GALAXY*\n'
  )



class ClusterPlotStage(PipelineStage):
  def __init__(
    self, 
    fmt: Literal['pdf', 'jpg', 'png'] = 'pdf', 
    overwrite: bool = False, 
    separated: bool = False,
  ):
    self.fmt = fmt
    self.overwrite = overwrite
    self.separated = separated
    
  def add_circle(
    self, 
    ra: float, 
    dec: float, 
    radius: float,
    color: str,
    ax,
    label: str = '',
    ls: str = '-',
  ):
    circle = Circle(
      (ra, dec), 
      radius,
      fc='none', 
      lw=2, 
      linestyle=ls,
      ec=color, 
      transform=ax.get_transform('icrs'), 
      label=label,
    )
    ax.add_patch(circle)
    
  def add_all_circles(
    self,
    cls_ra: float,
    cls_dec: float,
    r200_deg: float,
    r200_Mpc: float,
    r500_deg: float,
    r500_Mpc: float,
    r15Mpc_deg: float,
    ax,
  ):
    if r200_deg:
      self.add_circle(
        ra=cls_ra,
        dec=cls_dec,
        radius=5*r200_deg,
        color='tab:green',
        label=f'5 $\\times$ R200 ({5*r200_Mpc:.2f}Mpc)',
        ax=ax
      )
    if r500_deg:
      self.add_circle(
        ra=cls_ra,
        dec=cls_dec,
        radius=5*r500_deg,
        color='tab:green',
        ls='--',
        label=f'5 $\\times$ R500 ({5*r500_Mpc:.2f}Mpc)',
        ax=ax
      )
    if r15Mpc_deg:
      self.add_circle(
        ra=cls_ra,
        dec=cls_dec,
        radius=r15Mpc_deg,
        color='tab:brown',
        label=f'15Mpc',
        ax=ax
      )
    
  def add_cluster_center(self, ra: float, dec: float, ax):
    ax.scatter(
      ra, 
      dec, 
      marker='+', 
      linewidths=2, 
      s=140, 
      c='tab:red', 
      rasterized=True, 
      transform=ax.get_transform('icrs'),
      label='Cluster Center'
    )
    
  def plot_specz(
    self,
    cls_ra: float, 
    cls_dec: float, 
    cls_r200_deg: float, 
    cls_r500_deg: float, 
    cls_r200_Mpc: float, 
    cls_r500_Mpc: float, 
    cls_15Mpc_deg: float,
    df_specz_radial: pd.DataFrame,
    df_members: pd.DataFrame,
    df_interlopers: pd.DataFrame,
    ax: plt.Axes,
  ):
    ra_col, dec_col = guess_coords_columns(df_specz_radial)
    if df_members is not None and df_interlopers is not None:
      ax.scatter(
        df_specz_radial[ra_col].values, 
        df_specz_radial[dec_col].values, 
        c='tab:red', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'Spec Unclassif.'
      )
      ra_col, dec_col = guess_coords_columns(df_members)
      ax.scatter(
        df_members[ra_col].values, 
        df_members[dec_col].values, 
        c='tab:blue', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'Spec Member ({len(df_members)})'
      )
      ax.scatter(
        df_interlopers[ra_col].values, 
        df_interlopers[dec_col].values, 
        c='tab:orange', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'Spec Interloper ({len(df_interlopers)})'
      )
    else:
      ax.scatter(
        df_specz_radial[ra_col].values, 
        df_specz_radial[dec_col].values, 
        c='tab:blue', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'Spec Z'
      )
    self.add_cluster_center(cls_ra, cls_dec, ax)
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      r15Mpc_deg=cls_15Mpc_deg,
      ax=ax
    )
    ax.set_title(f'Spec Z Only - Objects: {len(df_specz_radial)}')
    ax.invert_xaxis()
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
  
  def plot_photoz(
    self,
    cls_ra: float, 
    cls_dec: float, 
    cls_r200_deg: float, 
    cls_r500_deg: float, 
    cls_r200_Mpc: float, 
    cls_r500_Mpc: float, 
    cls_15Mpc_deg: float,
    df_photoz_radial: pd.DataFrame,
    z_photo_range: Tuple[float, float],
    ax: plt.Axes,
  ):
    df_photoz_radial.loc[:,'idx'] = range(len(df_photoz_radial))
    df_photoz_good = df_photoz_radial[df_photoz_radial.zml.between(*z_photo_range)]
    df_photoz_bad = df_photoz_radial[~df_photoz_radial.zml.between(*z_photo_range)]
    ra_col, dec_col = guess_coords_columns(df_photoz_radial)
    if len(df_photoz_bad) > 0:
      ax.scatter(
        df_photoz_bad[ra_col].values, 
        df_photoz_bad[dec_col].values, 
        c='silver', 
        s=6, 
        alpha=0.5, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'Bad Photo Z ({len(df_photoz_bad)} obj)'
      )
    if len(df_photoz_good) > 0:
      ax.scatter(
        df_photoz_good[ra_col].values, 
        df_photoz_good[dec_col].values,
        c='tab:blue', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'Good Photo Z ({len(df_photoz_good)} obj)'
      )
    self.add_cluster_center(cls_ra, cls_dec, ax)
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      r15Mpc_deg=cls_15Mpc_deg,
      ax=ax
    )
    ax.set_title(f'S-PLUS Photo Z Only - Objects: {len(df_photoz_radial)}')
    ax.invert_xaxis()
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
  
  def plot_photoz_specz(
    self,
    cls_ra: float, 
    cls_dec: float, 
    cls_r200_deg: float, 
    cls_r500_deg: float, 
    cls_r200_Mpc: float, 
    cls_r500_Mpc: float, 
    cls_15Mpc_deg: float,
    df_specz_radial: pd.DataFrame,
    df_photoz_radial: pd.DataFrame,
    z_photo_range: Tuple[float, float],
    ax: plt.Axes,
  ):
    if not 'idx' in df_photoz_radial.columns:
      df_photoz_radial.loc[:,'idx'] = range(len(df_photoz_radial))
    df_photoz_good = df_photoz_radial[df_photoz_radial.zml.between(*z_photo_range)]
    
    if len(df_specz_radial) > 0 and len(df_photoz_radial) > 0:
      df_match = fast_crossmatch(df_specz_radial, df_photoz_radial)
      df_photoz_good_with_spec = df_match[df_match.zml.between(*z_photo_range)]
      df_photoz_good_wo_spec = df_photoz_good[~df_photoz_good.idx.isin(df_match.idx)]
      df_photoz_bad_with_spec = df_match[~df_match.zml.between(*z_photo_range)]
      ra_col, dec_col = guess_coords_columns(df_photoz_radial)
      if len(df_photoz_good_wo_spec) > 0:
        ax.scatter(
          df_photoz_good_wo_spec[ra_col].values, 
          df_photoz_good_wo_spec[dec_col].values, 
          c='tab:olive', 
          s=6, 
          rasterized=True, 
          transform=ax.get_transform('icrs'),
          label=f'Good Photo Z wo/ Spec Z ({len(df_photoz_good_wo_spec)} obj)'
        )
      if len(df_photoz_bad_with_spec) > 0:
        ax.scatter(
          df_photoz_bad_with_spec[ra_col].values, 
          df_photoz_bad_with_spec[dec_col].values, 
          c='tab:orange', 
          s=6, 
          rasterized=True, 
          transform=ax.get_transform('icrs'),
          label=f'Bad Photo Z w/ Spec Z ({len(df_photoz_bad_with_spec)} obj)'
        )
      if len(df_photoz_good_with_spec) > 0:
        ax.scatter(
          df_photoz_good_with_spec[ra_col].values, 
          df_photoz_good_with_spec[dec_col].values, 
          c='tab:blue', 
          s=6, 
          rasterized=True, 
          transform=ax.get_transform('icrs'),
          label=f'Good Photo Z w/ Spec Z ({len(df_photoz_good_with_spec)} obj)'
        )
    self.add_cluster_center(cls_ra, cls_dec, ax)
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      r15Mpc_deg=cls_15Mpc_deg,
      ax=ax
    )
    ax.set_title(f'Photo Z $\\cap$ Spec Z (CrossMatch Distance: 1 arcsec)')
    ax.invert_xaxis()
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    
  def run(
    self, 
    cls_name: str,
    cls_ra: float, 
    cls_dec: float, 
    cls_z: float,
    cls_r200_deg: float, 
    cls_r500_deg: float, 
    cls_r200_Mpc: float, 
    cls_r500_Mpc: float, 
    cls_15Mpc_deg: float,
    z_photo_range: Tuple[float, float],
    z_spec_range: Tuple[float, float],
    df_photoz_radial: pd.DataFrame,
    df_specz_radial: pd.DataFrame,
    df_members: pd.DataFrame,
    df_interlopers: pd.DataFrame,
  ):
    wcs_spec =  {
      # 'CDELT1': -1.0,
      # 'CDELT2': 1.0,
      # 'CRPIX1': 8.5,
      # 'CRPIX2': 8.5,
      'CRVAL1': cls_ra,
      'CRVAL2': cls_dec,
      'CTYPE1': 'RA---AIT',
      'CTYPE2': 'DEC--AIT',
      'CUNIT1': 'deg',
      'CUNIT2': 'deg'
    }
    wcs = WCS(wcs_spec)
    
    title = get_plot_title(
        cls_name=cls_name,
        cls_ra=cls_ra,
        cls_dec=cls_dec,
        cls_z=cls_z,
        cls_15Mpc_deg=cls_15Mpc_deg,
        z_spec_range=z_spec_range,
        z_photo_range=z_photo_range,
      )
    
    if self.separated:
      out = WEBSITE_PATH / 'clusters' / cls_name / f'specz.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot(projection=wcs)
        self.plot_specz(
          cls_ra=cls_ra, 
          cls_dec=cls_dec,
          cls_r200_deg=cls_r200_deg, 
          cls_r500_deg=cls_r500_deg, 
          cls_r200_Mpc=cls_r200_Mpc, 
          cls_r500_Mpc=cls_r500_Mpc, 
          cls_15Mpc_deg=cls_15Mpc_deg,
          df_members=df_members,
          df_interlopers=df_interlopers,
          df_specz_radial=df_specz_radial,
          ax=ax,
        )
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = WEBSITE_PATH / 'clusters' / cls_name / f'photoz.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot(projection=wcs)
        self.plot_photoz(
          cls_ra=cls_ra, 
          cls_dec=cls_dec,
          cls_r200_deg=cls_r200_deg, 
          cls_r500_deg=cls_r500_deg, 
          cls_r200_Mpc=cls_r200_Mpc, 
          cls_r500_Mpc=cls_r500_Mpc, 
          cls_15Mpc_deg=cls_15Mpc_deg,
          df_photoz_radial=df_photoz_radial,
          z_photo_range=z_photo_range,
          ax=ax,
        )
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = WEBSITE_PATH / 'clusters' / cls_name / f'photoz_specz.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot(projection=wcs)
        self.plot_photoz_specz(
          cls_ra=cls_ra, 
          cls_dec=cls_dec,
          cls_r200_deg=cls_r200_deg, 
          cls_r500_deg=cls_r500_deg, 
          cls_r200_Mpc=cls_r200_Mpc, 
          cls_r500_Mpc=cls_r500_Mpc, 
          cls_15Mpc_deg=cls_15Mpc_deg,
          df_specz_radial=df_specz_radial,
          df_photoz_radial=df_photoz_radial,
          z_photo_range=z_photo_range,
          ax=ax,
        )
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
      out_path = self.output_folder / f'cls_{cls_name}.{self.fmt}'
      if not self.overwrite and out_path.exists():
        return
      
      fig, axs = plt.subplots(
        nrows=3, 
        ncols=1, 
        figsize=(12, 27), 
        subplot_kw={'projection': wcs}, 
        dpi=300
      )
      
      self.plot_specz(
        cls_ra=cls_ra, 
        cls_dec=cls_dec,
        cls_r200_deg=cls_r200_deg, 
        cls_r500_deg=cls_r500_deg, 
        cls_r200_Mpc=cls_r200_Mpc, 
        cls_r500_Mpc=cls_r500_Mpc, 
        cls_15Mpc_deg=cls_15Mpc_deg,
        df_members=df_members,
        df_interlopers=df_interlopers,
        df_specz_radial=df_specz_radial,
        ax=axs[0],
      )
      
      self.plot_photoz(
        cls_ra=cls_ra, 
        cls_dec=cls_dec,
        cls_r200_deg=cls_r200_deg, 
        cls_r500_deg=cls_r500_deg, 
        cls_r200_Mpc=cls_r200_Mpc, 
        cls_r500_Mpc=cls_r500_Mpc, 
        cls_15Mpc_deg=cls_15Mpc_deg,
        df_photoz_radial=df_photoz_radial,
        z_photo_range=z_photo_range,
        ax=axs[1],
      )
      
      self.plot_photoz_specz(
        cls_ra=cls_ra, 
        cls_dec=cls_dec,
        cls_r200_deg=cls_r200_deg, 
        cls_r500_deg=cls_r500_deg, 
        cls_r200_Mpc=cls_r200_Mpc, 
        cls_r500_Mpc=cls_r500_Mpc, 
        cls_15Mpc_deg=cls_15Mpc_deg,
        df_specz_radial=df_specz_radial,
        df_photoz_radial=df_photoz_radial,
        z_photo_range=z_photo_range,
        ax=axs[2],
      )
      
      fig.suptitle(title, size=18)
      plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)




class VelocityPlotStage(PipelineStage):
  def __init__(
    self, 
    overwrite: bool = False, 
    fmt: Literal['pdf', 'jpg', 'png'] = 'pdf',
    separated: bool = False,
  ):
    self.overwrite = overwrite
    self.separated = separated
    self.fmt = fmt
    
    
  def add_circle(
    self, 
    ra: float, 
    dec: float, 
    radius: float,
    color: str,
    ax,
    label: str = '',
    ls: str = '-',
  ):
    circle = Circle(
      (ra, dec), 
      radius,
      fc='none', 
      lw=2, 
      linestyle=ls,
      ec=color, 
      transform=ax.get_transform('icrs'), 
      label=label,
    )
    ax.add_patch(circle)
    
  def add_all_circles(
    self,
    cls_ra: float,
    cls_dec: float,
    r200_deg: float,
    r200_Mpc: float,
    r500_deg: float,
    r500_Mpc: float,
    r15Mpc_deg: float,
    ax,
  ):
    if r200_deg:
      self.add_circle(
        ra=cls_ra,
        dec=cls_dec,
        radius=5*r200_deg,
        color='tab:green',
        label=f'5 $\\times$ R200 ({5*r200_Mpc:.2f}Mpc)',
        ax=ax
      )
    if r500_deg:
      self.add_circle(
        ra=cls_ra,
        dec=cls_dec,
        radius=5*r500_deg,
        color='tab:green',
        ls='--',
        label=f'5 $\\times$ R500 ({5*r500_Mpc:.2f}Mpc)',
        ax=ax
      )
    if r15Mpc_deg:
      self.add_circle(
        ra=cls_ra,
        dec=cls_dec,
        radius=r15Mpc_deg,
        color='tab:brown',
        label=f'15Mpc',
        ax=ax
      )
    
  def add_cluster_center(self, ra: float, dec: float, ax):
    ax.scatter(
      ra, 
      dec, 
      marker='+', 
      linewidths=1.5, 
      s=80, 
      c='k', 
      rasterized=True, 
      transform=ax.get_transform('icrs'),
      label='Cluster Center'
    )
    
  
  def plot_velocity(self, df_members: pd.DataFrame, df_interlopers: pd.DataFrame, ax: plt.Axes):
    ax.scatter(df_members.radius_Mpc, df_members.v_offset, c='tab:red', s=5, label='Members', rasterized=True)  
    ax.scatter(df_interlopers.radius_Mpc, df_interlopers.v_offset, c='tab:blue', s=5, label='Interlopers', rasterized=True)
    ax.legend()
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('$\\Delta v [km/s]$')
    ax.set_title('Spectroscoptic velocity x distance')
    
  def plot_specz(self, df_members: pd.DataFrame, df_interlopers: pd.DataFrame, cls_z: float, ax: plt.Axes):
    df_members['z_offset'] = df_members['z'] - cls_z
    df_interlopers['z_offset'] = df_interlopers['z'] - cls_z
    ax.scatter(df_members.radius_Mpc, df_members.z_offset, c='tab:red', s=5, label='Members', rasterized=True)  
    ax.scatter(df_interlopers.radius_Mpc, df_interlopers.z_offset, c='tab:blue', s=5, label='Interlopers', rasterized=True)
    ax.legend()
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('$\\Delta z_{{spec}}$')
    ax.set_title('Spectroscoptic redshift x distance')
    ymin = min(df_members.z_offset.min(), df_interlopers.z_offset.min()) - 0.0015
    ymax = max(df_members.z_offset.max(), df_interlopers.z_offset.max()) + 0.0015
    ax.set_ylim(ymin, ymax)
    
  def plot_photoz(self, df_members: pd.DataFrame, df_interlopers: pd.DataFrame, df_photoz_radial: pd.DataFrame, cls_z: float, ax: plt.Axes):
    df_members_match = fast_crossmatch(df_members, df_photoz_radial)
    df_interlopers_match = fast_crossmatch(df_interlopers, df_photoz_radial)
    df_members_match['zml_offset'] = df_members_match['zml'] - cls_z
    df_interlopers_match['zml_offset'] = df_interlopers_match['zml'] - cls_z
    ax.scatter(df_members_match.radius_Mpc, df_members_match.zml_offset, c='tab:red', s=5, label='Members', rasterized=True)  
    ax.scatter(df_interlopers_match.radius_Mpc, df_interlopers_match.zml_offset, c='tab:blue', s=5, label='Interlopers', rasterized=True)
    ax.legend()
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('$\\Delta z_{{photo}}$')
    ax.set_title('Photometric redshift x distance')
    ymin = min(df_members.z_offset.min(), df_interlopers.z_offset.min()) - 0.0015
    ymax = max(df_members.z_offset.max(), df_interlopers.z_offset.max()) + 0.0015
    ax.set_ylim(ymin, ymax)
  
  def plot_ra_dec(
    self, 
    cls_ra: float,
    cls_dec: float,
    cls_r200_deg: float,
    cls_r200_Mpc: float,
    cls_r500_deg: float,
    cls_r500_Mpc: float,
    cls_15Mpc_deg: float,
    df_members: pd.DataFrame, 
    df_interlopers: pd.DataFrame, 
    ax: plt.Axes
  ):
    # ax.set_title(f'Spec Z Only - Objects: {len(df_specz_radial)}')
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      r15Mpc_deg=cls_15Mpc_deg,
      ax=ax
    )
    ax.scatter(
      df_members.ra, 
      df_members.dec, 
      c='tab:red', 
      s=5,
      label='Members', 
      transform=ax.get_transform('icrs'), 
      rasterized=True,
    )
    ax.scatter(
      df_interlopers.ra, 
      df_interlopers.dec, 
      c='tab:blue', 
      s=5,
      label='Interlopers', 
      transform=ax.get_transform('icrs'), 
      rasterized=True,
    )
    self.add_cluster_center(cls_ra, cls_dec, ax)
    ax.invert_xaxis()
    ax.set_aspect('equal', adjustable='datalim', anchor='C')
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    ax.legend()
    ax.set_title('Spatial distribution of spectroscopic members')
  
  
  def run(
    self, 
    cls_name: str, 
    cls_ra: float, 
    cls_dec: float, 
    cls_z: float,
    cls_15Mpc_deg: float,
    cls_r200_deg: float,
    cls_r200_Mpc: float,
    cls_r500_deg: float,
    cls_r500_Mpc: float,
    z_photo_range: Tuple[float, float],
    z_spec_range: Tuple[float, float],
    df_members: pd.DataFrame,
    df_interlopers: pd.DataFrame,
    df_photoz_radial: pd.DataFrame,
  ):
    wcs_spec =  {
      # 'CDELT1': -1.0,
      # 'CDELT2': 1.0,
      # 'CRPIX1': 8.5,
      # 'CRPIX2': 8.5,
      'CRVAL1': cls_ra,
      'CRVAL2': cls_dec,
      'CTYPE1': 'RA---AIT',
      'CTYPE2': 'DEC--AIT',
      'CUNIT1': 'deg',
      'CUNIT2': 'deg'
    }
    wcs = WCS(wcs_spec)
    title = get_plot_title(
      cls_name=cls_name,
      cls_ra=cls_ra,
      cls_dec=cls_dec,
      cls_z=cls_z,
      cls_15Mpc_deg=cls_15Mpc_deg,
      z_spec_range=z_spec_range,
      z_photo_range=z_photo_range,
    )
      
    if self.separated:
      out = WEBSITE_PATH / 'clusters' / cls_name / f'spec_velocity.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot()
        self.plot_velocity(df_members, df_interlopers, ax)
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = WEBSITE_PATH / 'clusters' / cls_name / f'specz_distance.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot()
        self.plot_specz(df_members, df_interlopers, cls_z, ax)
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = WEBSITE_PATH / 'clusters' / cls_name / f'photoz_distance.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot()
        self.plot_photoz(df_members, df_interlopers, df_photoz_radial, cls_z, ax)
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = WEBSITE_PATH / 'clusters' / cls_name / f'spec_velocity_position.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot(projection=wcs)
        self.plot_ra_dec(cls_ra, cls_dec, cls_r200_deg, cls_r200_Mpc, cls_r500_deg, 
                         cls_r500_Mpc, cls_15Mpc_deg, df_members, df_interlopers, ax)
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
      out_path = VELOCITY_PLOTS_FOLDER / f'cls_{cls_name}.{self.fmt}'
      if not self.overwrite and out_path.exists():
        return
      
      fig = plt.figure(figsize=(7.5, 16), dpi=300)
      ax1 = fig.add_subplot(211)
      ax2 = fig.add_subplot(212, projection=wcs)
      self.plot_velocity(df_members, df_interlopers, ax1)
      self.plot_ra_dec(cls_ra, cls_dec, cls_r200_deg, cls_r200_Mpc, cls_r500_deg, 
                      cls_r500_Mpc, cls_15Mpc_deg, df_members, df_interlopers, ax2)
      
      fig.suptitle(title, size=18)
      plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)




class MagDiffPlotStage(PipelineStage):
  def __init__(
    self, 
    overwrite: bool = False, 
    fmt: Literal['pdf', 'jpg', 'png'] = 'pdf', 
    separated: bool = False
  ):
    self.overwrite = overwrite
    self.separated = separated
    self.fmt = fmt
    self.white_viridis = LinearSegmentedColormap.from_list(
      'white_viridis', 
      [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
      ], 
      N=256
    )
    
  def plot_mag_diff(
    self,
    df: pd.DataFrame,
    mag1: str, 
    mag2: str, 
    ax: plt.Axes, 
    xlabel: str = None,
    ylabel: str = None,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
  ):
    df_spec = df[~df.z.isna()]
    df_photo = df[df.z.isna()]
    ax.scatter(
      df_photo[mag1], 
      df_photo[mag1] - df_photo[mag2], 
      s=0.8, 
      label=f'Without Spec ({len(df_photo)})', 
      color='tab:blue', 
      alpha=0.8, 
      rasterized=True
    )
    ax.scatter(
      df_spec[mag1], 
      df_spec[mag1] - df_spec[mag2], 
      s=0.8, 
      label=f'With Spec ({len(df_spec)})', 
      color='tab:red', 
      alpha=0.8,
      rasterized=True
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
      ax.set_ylim(*ylim)
    if xlim is not None:
      ax.set_xlim(*xlim)
    ax.grid('on', color='k', linestyle='--', alpha=.2)
    ax.tick_params(direction='in')
    ax.legend()
    ax.set_title('SP-LS r-mag difference')
    
  def plot_histogram(
    self,
    x: np.ndarray | pd.Series | pd.DataFrame,
    ax: plt.Axes,
    xlabel: str = '',
    xrange: Tuple[float, float] = None,
  ):
    ax.hist(x, bins=60, range=xrange)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of galaxies')
    if xrange is not None:
      ax.set_xlim(*xrange)
    ax.grid('on', color='k', linestyle='--', alpha=.2)
    ax.tick_params(direction='in')
    ax.set_title('SP-LS r-mag difference distribution')
  
  def run(
    self, 
    df_all_radial: pd.DataFrame, 
    cls_ra: float, 
    cls_dec: float, 
    cls_name: str, 
    cls_z: float, 
    z_spec_range: Tuple[float, float], 
    z_photo_range: Tuple[float, float], 
    cls_15Mpc_deg: float
  ):
    df = df_all_radial[
      (df_all_radial.type != 'PSF') & 
      df_all_radial.r_auto.between(*MAG_RANGE) & 
      df_all_radial.mag_r.between(*MAG_RANGE) &
      (df_all_radial.z.between(*z_spec_range) | df_all_radial.z.isna()) &
      df_all_radial.zml.between(*z_photo_range)
    ]
    title = get_plot_title(
      cls_name=cls_name,
      cls_ra=cls_ra,
      cls_dec=cls_dec,
      cls_z=cls_z,
      cls_15Mpc_deg=cls_15Mpc_deg,
      z_spec_range=z_spec_range,
      z_photo_range=z_photo_range,
    )
    
    if self.separated:
      out = WEBSITE_PATH / 'clusters' / cls_name / f'mag_diff.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(figsize=(7.5, 7.5), dpi=150)
        self.plot_mag_diff(df, 'r_auto', 'mag_r', axs, '$iDR5_r$', '$iDR5_r - LS10_r$')
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
      
      out = WEBSITE_PATH / 'clusters' / cls_name / f'mag_diff_hist.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(figsize=(7.5, 7.5), dpi=150)
        self.plot_histogram(df['r_auto'] - df['mag_r'], axs, '$iDR5_r - LS10_r$')
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
      out_path = MAGDIFF_PLOTS_FOLDER / f'cls_{cls_name}.pdf'
      if not self.overwrite and out_path.exists():
        return

      fig, axs = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(5, 9),
        dpi=300,
        # subplot_kw={'projection': 'scatter_density'}, 
      )
      self.plot_mag_diff(df, 'r_auto', 'mag_r', axs[0], '$iDR5_r$', '$iDR5_r - LS10_r$')
      self.plot_histogram(df['r_auto'] - df['mag_r'], axs[1], '$iDR5_r - LS10_r$')
      plt.suptitle(title, size=10)
      plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)


class MagdiffOutlierStage(PipelineStage):
  def __init__(self, overwrite: bool = False):
    self.overwrite = overwrite
    
  def run(
    self, 
    df_all_radial: pd.DataFrame, 
    cls_name: str, 
    z_spec_range: Tuple[float, float], 
    z_photo_range: Tuple[float, float]
  ):
    out_path = MAGDIFF_OUTLIERS_FOLDER / f'{cls_name}.csv'
    if out_path.exists() and not self.overwrite:
      return
    
    df_all_radial['idr5_r-ls10_r'] = df_all_radial['r_auto'] - df_all_radial['mag_r']
    df = df_all_radial[
      (df_all_radial.type != 'PSF') & 
      df_all_radial.r_auto.between(13, 19) & 
      # df_all_radial.mag_r.between(*MAG_RANGE) &
      (df_all_radial.z.between(*z_spec_range) | df_all_radial.z.isna()) &
      df_all_radial.zml.between(*z_photo_range)
    ]
    df = df[(df['idr5_r-ls10_r'] > -0.2) & (df['idr5_r-ls10_r'] < 0.5)]
    df = df.sort_values('idr5_r-ls10_r')
    write_table(df, out_path)




class MagnitudeCrossmatch(PipelineStage):
  def __init__(self, overwrite: bool = False):
    self.overwrite = overwrite
    
  def run(self, cls_name: str, df_magnitudes: pd.DataFrame, df_photoz_radial: pd.DataFrame):
    out_path = MAG_COMP_FOLDER / f'{cls_name}.parquet'
    if out_path.exists() and not self.overwrite:
      if self.get_data('cls_15Mpc_deg') < 10.17:
        return 
    
    df = fast_crossmatch(
      left=df_photoz_radial,
      right=df_magnitudes,
      include_sep=False
    )
    del df['RA_1']
    del df['DEC_1']
    df = df.rename(columns={'ra_1': 'ra', 'dec_1': 'dec'})
    write_table(df, out_path)
    print('Crossmatch objects:', len(df))
    




class MagnitudePlotStage(PipelineStage):
  def plot_mag_diff(self, ax: plt.Axes):
    pass
  
  def run(self, cls_ra: float, cls_dec: float):
    wcs_spec =  {
      # 'CDELT1': -1.0,
      # 'CDELT2': 1.0,
      # 'CRPIX1': 8.5,
      # 'CRPIX2': 8.5,
      'CRVAL1': cls_ra,
      'CRVAL2': cls_dec,
      'CTYPE1': 'RA---AIT',
      'CTYPE2': 'DEC--AIT',
      'CUNIT1': 'deg',
      'CUNIT2': 'deg'
    }
    wcs = WCS(wcs_spec)
    fig, axs = plt.subplots(
      nrows=1, 
      ncols=2, 
      figsize=(12, 27), 
      subplot_kw={'projection': WCS(wcs)}, 
      dpi=150
    )
    
    


class WebsitePagesStage(PipelineStage):
  def __init__(self, clusters: Sequence[str], fmt: str = 'png'):
    self.clusters = sorted(clusters)
    self.fmt = fmt
  
  def get_paginator(self, back: bool = True):
    if back:
      links = [f'<a href="../{name}/index.html">{name}</a>' for name in self.clusters]
    else:
      links = [f'<a href="clusters/{name}/index.html">{name}</a>' for name in self.clusters]
      
    return ' &nbsp;&bullet;&nbsp; '.join(links)
  
  def make_index(self):
    page = f'''
    <html>
    <head>
      <title>Clusters Index</title>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.4/css/lightbox.min.css" />
    </head>
    <body>
      <h2>Clusters Index</h2>
      {self.get_paginator(back=False)}
      <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.4/js/lightbox-plus-jquery.min.js"></script>
    </body>
    </html>
    '''
    index_path = WEBSITE_PATH / 'index.html'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(page)
  
  def run(
    self, 
    cls_name: str, 
    cls_ra: float, 
    cls_dec: float, 
    cls_z: float,
    cls_15Mpc_deg: float,
    cls_r200_deg: float,
    z_spec_range: Tuple[float, float],
    z_photo_range: Tuple[float, float],
  ):
    width = 400
    height = 400
    images = [
      'specz', 'photoz', 'photoz_specz', 'spec_velocity_position', 
      'spec_velocity', 'specz_distance', 'photoz_distance', 
      'mag_diff', 'mag_diff_hist',
    ]
    gallery = [
      f'<a href="{img}.{self.fmt}" class="gallery" data-lightbox="images"><img src="{img}.{self.fmt}" width="{width}" height="{height}" /></a>'
      for img in images
    ]
    page = f'''
    <html>
    <head>
      <title>{cls_name}</title>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.4/css/lightbox.min.css" />
      <style type="text/css">
      a.gallery:hover {{
        cursor: -moz-zoom-in; 
        cursor: -webkit-zoom-in;
        cursor: zoom-in;
      }}
      </style>
    </head>
    <body>
      <b>Clusters Index</b><br />
      {self.get_paginator()}
      <hr />
      <h2>Cluster: {cls_name}</h2>
      <i>
        RA: {cls_ra:.4f}&deg;, DEC: {cls_dec:.4f}&deg;,
        z: {cls_z:.4f} &bullet; search radius: 15Mpc ({cls_15Mpc_deg:.3f}&deg;)
      </i>
      <br />
      <i>
        Spec Z Range: z<sub>cluster</sub> &plusmn; 0.007 = [{z_spec_range[0]:.4f}, {z_spec_range[1]:.4f}]
      </i>
      <br />
      <i>
        Good Photo Z: z<sub>cluster</sub> &plusmn; 0.015 = [{z_photo_range[0]:.4f}, {z_photo_range[1]:.4f}]
      </i>
      <br />
      <i>
        R Mag Range: [13, 22] &bullet; spec class = GALAXY*
      </i>
      <br />
      <p><b>Gallery</b></p>
      {' '.join(gallery)}
      <p><b>Legacy DR10</b></p>
      <div id='aladin-lite-div' style='width: 850px; height: 700px; margin:0 auto;'></div>
      
      <script src="https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.4/js/lightbox-plus-jquery.min.js"></script>
      <script>
      lightbox.option({{
        'resizeDuration': 0,
        'fadeDuration': 0,
        'imageFadeDuration': 0,
        'wrapAround': true,
        'fitImagesInViewport': true,
      }})
      </script>
      
      <script src='https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js' charset='utf-8'></script>
      <script>
      const catFilter = (source) => {{
        return !isNaN(parseFloat(source.data['redshift'])) && (source.data['redshift'] > {z_spec_range[0]}) && (source.data['redshift'] < {z_spec_range[1]})
      }}
      
      var aladin;
      A.init.then(() => {{
        // Init Aladin
        aladin = A.aladin('#aladin-lite-div', {{
          source: 'CDS/P/DESI-Legacy-Surveys/DR10/color',
          target: '{cls_ra:.6f} {cls_dec:.6f}', 
          fov: {5*cls_r200_deg + 0.3},
          cooFrame: 'ICRSd',
        }});
        aladin.setImageSurvey('CDS/P/DESI-Legacy-Surveys/DR10/color');
        
        // Add 5R200 Circle
        var overlay = A.graphicOverlay({{color: '#ee2345', lineWidth: 2}});
        aladin.addOverlay(overlay);
        overlay.add(A.ellipse({cls_ra:.6f}, {cls_dec:.6f}, {2.5*cls_r200_deg}, {2.5*cls_r200_deg}, 0));
        
        // Add redshift catalog
        const cat_url = 'http://cdsxmatch.u-strasbg.fr/QueryCat/QueryCat?catName=SIMBAD&mode=cone&pos={cls_ra:.6f}%20{cls_dec:.6f}&r={5*cls_r200_deg:.3f}deg&format=votable&limit=6000'
        const cat = A.catalogFromURL(cat_url, {{
          name: 'redshift',
          sourceSize:12, 
          color: '#f72525', 
          displayLabel: true, 
          labelColumn: 'redshift', 
          labelColor: '#31c3f7', 
          labelFont: '14px sans-serif', 
          onClick: 'showPopup', 
          shape: 'circle',
          filter: catFilter,
        }})
        aladin.addCatalog(cat)
      }});
      </script>
    </body>
    </html>
    '''
    index_path = WEBSITE_PATH / 'clusters' / cls_name / 'index.html'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(page)
    


    
    

def prepare_idr5():
  fields_path = Path('/mnt/hd/natanael/astrodata/idr5_fields/')
  output_path = Path('/mnt/hd/natanael/astrodata/idr5_photoz_clean.parquet')
  cols = ['RA', 'DEC', 'zml', 'odds', 'r_auto', 'remove_flag']
  for path in (pbar := tqdm(list(fields_path.glob('*.csv')))):
    pbar.set_description(path.stem)
    df = read_table(path, columns=cols)
    df = df[df['remove_flag'] == False]
    del df['remove_flag']
    df = df.rename(columns={'RA': 'ra', 'DEC': 'dec'})
    df['field'] = path.stem
    write_table(df, str(path.absolute()).replace('.csv', '.parquet'))
  write_table(concat_tables(list(fields_path.glob('*.parquet'))), output_path)



def download_legacy_pipeline(clear: bool = False):
  df_clusters = load_clusters()
  
  if clear:
    for p in LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
        
  ls10_pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', overwrite=False, workers=8)
  )
  ls10_pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)



def download_legacy_erass_pipeline(clear: bool = False):
  df_clusters = load_eRASS()
  
  if clear:
    for p in LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
        
  ls10_pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', overwrite=False, workers=8)
  )
  ls10_pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1)



def match_all_pipeline():
  df_clusters = load_clusters()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=False),
  )
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  
def match_all_erosita_pipeline():
  df_clusters = load_eRASS()
  
  pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=False),
  )
  
  pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1)



def photoz_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  df_photoz, photoz_skycoord = load_photoz()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=overwrite),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)



def spec_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    SpecZRadialSearchStage(overwrite=overwrite),
  )
  
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  


def spec_plots_pipeline():
  df_clusters = load_clusters()
  df_photoz, photoz_skycoord = load_photoz()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=True),
    SpecZRadialSearchStage(overwrite=False),
    ClusterPlotStage(overwrite=True)
  )
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  df2 = df_clusters.sort_values('ra')
  plot_paths = [PLOTS_FOLDER / f'cls_{c}.pdf' for c in df2.name.values]
  concat_plot_path = PLOTS_FOLDER / 'clusters_v6_RA.pdf'
  merge_pdf(plot_paths, concat_plot_path)




def photoz_erass_pipeline(overwrite: bool = False):
  df_clusters = load_eRASS()
  df_photoz, photoz_skycoord = load_photoz()
  
  pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=overwrite),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1)


  
def spec_erass_pipeline(overwrite: bool = False):
  df_clusters = load_eRASS()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    SpecZRadialSearchStage(overwrite=overwrite)
  )
  
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1)



def erass_plots_pipeline():
  df_clusters = load_eRASS()
  df_photoz, photoz_skycoord = load_photoz()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    ClusterPlotStage(overwrite=False)
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1)
  # df2 = selfmatch(df_clusters, 1*u.deg, 'identify', ra='RA_OPT', dec='DEC_OPT')
  # df2 = df2.sort_values('GroupID')
  df2 = df_clusters.sort_values('RA_OPT')
  plot_paths = [PLOTS_FOLDER / f'cls_{c}.pdf' for c in df2.Cluster.values]
  concat_plot_path = PLOTS_FOLDER / 'eRASS_v1.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  



def erass_website_plots_pipeline():
  # df_clusters = load_full_eRASS()
  df_clusters = load_eRASS()
  df_photoz, photoz_skycoord = load_photoz()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    LoadLegacyRadialStage(),
    StarsRemovalStage(),
    ClusterPlotStage(overwrite=False, fmt='jpg', output_folder='outputs_v6/website_plots')
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1, validate=False)




def magdiff_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  # df_erass = load_eRASS()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    LoadAllRadialStage(),
    MagDiffPlotStage(overwrite=overwrite),
  )
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  
  # pipe = Pipeline(
  #   LoadERASSInfoStage(df_erass),
  #   LoadAllRadialStage(),
  #   MagDiffPlotStage(overwrite=overwrite),
  # )
  # pipe.map_run('cls_name', df_erass.Cluster.values, workers=1)
  
  plot_paths = sorted(MAGDIFF_PLOTS_FOLDER.glob('cls_*.pdf'))
  concat_plot_path = MAGDIFF_PLOTS_FOLDER / 'magdiff_v2.pdf'
  merge_pdf(plot_paths, concat_plot_path)




def heasarc_plot_pipeline(overwrite: bool = False):
  df_heasarc = load_heasarc()
  df_photoz, photoz_skycoord = load_photoz()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadHeasarcInfoStage(df_heasarc),
    PhotoZRadialSearchStage(overwrite=overwrite),
    SpecZRadialSearchStage(overwrite=overwrite),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', overwrite=overwrite),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(),
    LoadAllRadialStage(),
    ClusterPlotStage(),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', ['[YMV2007]1854', '[YMV2007]337642', 'ACT-CLJ0006.9-0041', '[YMV2007]15744', '[YMV2007]337638',], workers=1)




def magdiff_outliers_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    LoadAllRadialStage(),
    MagdiffOutlierStage(overwrite),
  )
  
  pipe.map_run('cls_id', [12, 27], workers=1)



def velocity_plots_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    VelocityPlotStage(overwrite=overwrite)
  )
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  plot_paths = sorted(VELOCITY_PLOTS_FOLDER.glob('cls_*.pdf'))
  concat_plot_path = VELOCITY_PLOTS_FOLDER / 'velocity_plots_v1.pdf'
  merge_pdf(plot_paths, concat_plot_path)



def website_pipeline(overwrite: bool = False):
  df_clusters = load_clusters()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=overwrite, fmt='jpg', separated=True),
    VelocityPlotStage(overwrite=overwrite, fmt='jpg', separated=True),
    MagDiffPlotStage(overwrite=overwrite, fmt='jpg', separated=True),
    WebsitePagesStage(clusters=df_clusters.name.values, fmt='jpg'),
  )
  
  pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  WebsitePagesStage(clusters=df_clusters.name.values).make_index()




def main():
  # prepare_idr5()
  # download_legacy_erass_pipeline(clear=True)
  # download_legacy_pipeline(clear=True)
  # spec_pipeline()
  # spec_erass_pipeline()
  # photoz_pipeline()
  # photoz_erass_pipeline()
  # match_all_pipeline()
  # match_all_erosita_pipeline()
  # spec_plots_pipeline()
  # erass_plots_pipeline()
  # erass_website_plots_pipeline()
  # print(load_clusters())
  # magdiff_pipeline(False)
  # heasarc_plot_pipeline(True)
  # magdiff_outliers_pipeline(True)
  # velocity_plots_pipeline(True)
  website_pipeline(False)
  
  
  
if __name__ == '__main__':
  main()