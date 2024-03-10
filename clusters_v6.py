from pathlib import Path
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
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
from matplotlib.patches import Circle
from tqdm import tqdm

PHOTOZ_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/idr5_photoz_clean.parquet')
SPEC_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/SpecZ_Catalogue_20240124.parquet')
ERASS_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/liana_erass.csv')
TABLES_PATH = Path('clusters_members')
MEMBERS_FOLDER = Path('clusters_members/clusters')
OUT_PATH = Path('outputs_v6')
PLOTS_FOLDER = OUT_PATH / 'plots'
LEG_PHOTO_FOLDER = OUT_PATH / 'legacy'
PHOTOZ_FOLDER = OUT_PATH / 'photo'
SPECZ_FOLDER = OUT_PATH / 'spec'
Z_PHOTO_DELTA = 0.015
Z_SPEC_DELTA = 0.007
MAG_RANGE = (13, 22)

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
  df_erass['Cluster'].str.replace(' ', '_')
  return df_erass


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
    r15Mpc_deg = min(mpc2arcsec(15, z).to(u.deg).value, 10.17)
    paulo_path = MEMBERS_FOLDER / f'cluster.gals.sel.shiftgap.iter.{str(cls_id).zfill(5)}'
    col_names = [
      'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
      'radius_Mpc', 'v_offset', 'flag_member'
    ] # 0 - member; 1 - interloper
    df_paulo = read_table(paulo_path, fmt='dat', col_names=col_names)
    df_members = df_paulo[df_paulo.flag_member == 0]
    df_interlopers = df_paulo[df_paulo.flag_member == 1]
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
    r15Mpc_deg = min(mpc2arcsec(15, z).to(u.deg).value, 10.17)
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


class RadialSearchStage(PipelineStage):
  def __init__(
    self, 
    df_name: str,
    out_key: str, 
    radius_key: str, 
    save_folder: str | Path,
    kind: Literal['spec', 'photo'],
    overwrite: bool = False,
    skycoord_name: str = None,
  ):
    self.df_name = df_name
    self.out_key = out_key
    self.products = [out_key]
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
      df_search = read_table(out_path)
    else:
      pos = SkyCoord(ra=cls_ra, dec=cls_dec, unit=u.deg, frame='icrs')
      radius = self.get_data(self.radius_key)
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
        
      if self.save_folder:
        write_table(df_search, self.save_folder / f'{cls_name}.parquet')
    return {self.out_key: df_search}



class SpecZRadialSearchStage(RadialSearchStage):
  def __init__(
    self, 
    save_folder: str | Path = SPECZ_FOLDER, 
    radius_key: str = 'cls_15Mpc_deg', 
    overwrite: bool = False,
  ):
    super().__init__(
      df_name='df_spec', 
      out_key='df_spec_radial', 
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
      out_key='df_photoz_radial', 
      radius_key=radius_key, 
      save_folder=save_folder, 
      kind='photo', 
      overwrite=overwrite, 
      skycoord_name='photoz_skycoord',
    )




class CrossmatchStage(PipelineStage):
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
    df_match = fast_crossmatch(df_left, self.df_spec, join=self.join)
    return {self.out_key: df_match}




class DownloadLegacyCatalogStage(PipelineStage):
  def __init__(self, radius_key: str, overwrite: bool = False, workers: int = 3):
    self.radius_key = radius_key
    self.overwrite = overwrite
    self.workers = workers
    
  def run(self, cls_ra: float, cls_dec: float, cls_name: str):
    out_path = LEG_PHOTO_FOLDER / f'{cls_name}+leg.parquet'
    if not self.overwrite and out_path.exists():
      # df = read_table(out_path)
      # return {'df_ls10': df}
      return
    
    sql = """
      SELECT t.ra, t.dec, t.type, t.mag_r
      FROM ls_dr10.tractor AS t
      WHERE (ra BETWEEN {ra_min} AND {ra_max}) AND 
      (dec BETWEEN {dec_min} AND {dec_max}) AND 
      (brick_primary = 1) AND 
      (mag_r BETWEEN {r_min:.2f} AND {r_max:.2f})
    """
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
      for _r in np.arange(13, 22, .05)
    ]
    service = LegacyService(replace=self.overwrite, workers=self.workers)
    service.batch_sync_query(
      queries=queries, 
      save_paths=out_path, 
      join_outputs=True, 
      workers=self.workers
    )
    

class ClusterPlotStage(PipelineStage):
  def __init__(self, overwrite: bool = False):
    self.overwrite = overwrite
    
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
    df_spec_radial: pd.DataFrame,
    df_members: pd.DataFrame,
    df_interlopers: pd.DataFrame,
  ):
    out_path = PLOTS_FOLDER / f'cls_{cls_name}.pdf'
    if not self.overwrite and out_path.exists():
      return
    
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
      nrows=3, 
      ncols=1, 
      figsize=(12, 27), 
      subplot_kw={'projection': WCS(wcs_spec)}, 
      dpi=300
    )

    ra_col, dec_col = guess_coords_columns(df_spec_radial)
    if df_members is not None and df_interlopers is not None:
      axs[0].scatter(
        df_spec_radial[ra_col].values, 
        df_spec_radial[dec_col].values, 
        c='tab:red', 
        s=6, 
        rasterized=True, 
        transform=axs[0].get_transform('icrs'),
        label=f'Spec Unclassif.'
      )
      ra_col, dec_col = guess_coords_columns(df_members)
      axs[0].scatter(
        df_members[ra_col].values, 
        df_members[dec_col].values, 
        c='tab:blue', 
        s=6, 
        rasterized=True, 
        transform=axs[0].get_transform('icrs'),
        label=f'Spec Member ({len(df_members)})'
      )
      axs[0].scatter(
        df_interlopers[ra_col].values, 
        df_interlopers[dec_col].values, 
        c='tab:orange', 
        s=6, 
        rasterized=True, 
        transform=axs[0].get_transform('icrs'),
        label=f'Spec Interloper ({len(df_interlopers)})'
      )
    else:
      axs[0].scatter(
        df_spec_radial[ra_col].values, 
        df_spec_radial[dec_col].values, 
        c='tab:blue', 
        s=6, 
        rasterized=True, 
        transform=axs[0].get_transform('icrs'),
        label=f'Spec Z'
      )
    self.add_cluster_center(cls_ra, cls_dec, axs[0])
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      r15Mpc_deg=cls_15Mpc_deg,
      ax=axs[0]
    )
    axs[0].set_title(f'Spec Z Only - Objects: {len(df_spec_radial)}')
    axs[0].invert_xaxis()
    axs[0].legend(loc='upper left')
    axs[0].set_aspect('equal')
    axs[0].grid('on', color='k', linestyle='--', alpha=.5)
    axs[0].tick_params(direction='in')
    axs[0].set_xlabel('RA')
    axs[0].set_ylabel('DEC')
    
    
    df_photoz_radial.loc[:,'idx'] = range(len(df_photoz_radial))
    df_photoz_good = df_photoz_radial[df_photoz_radial.zml.between(*z_photo_range)]
    df_photoz_bad = df_photoz_radial[~df_photoz_radial.zml.between(*z_photo_range)]
    ra_col, dec_col = guess_coords_columns(df_photoz_radial)
    if len(df_photoz_bad) > 0:
      axs[1].scatter(
        df_photoz_bad[ra_col].values, 
        df_photoz_bad[dec_col].values, 
        c='silver', 
        s=6, 
        alpha=0.5, 
        rasterized=True, 
        transform=axs[1].get_transform('icrs'),
        label=f'Bad Photo Z ({len(df_photoz_bad)} obj)'
      )
    if len(df_photoz_good) > 0:
      axs[1].scatter(
        df_photoz_good[ra_col].values, 
        df_photoz_good[dec_col].values,
        c='tab:blue', 
        s=6, 
        rasterized=True, 
        transform=axs[1].get_transform('icrs'),
        label=f'Good Photo Z ({len(df_photoz_good)} obj)'
      )
    self.add_cluster_center(cls_ra, cls_dec, axs[1])
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      r15Mpc_deg=cls_15Mpc_deg,
      ax=axs[1]
    )
    axs[1].set_title(f'S-PLUS Photo Z Only - Objects: {len(df_photoz_radial)}')
    axs[1].invert_xaxis()
    axs[1].legend(loc='upper left')
    axs[1].set_aspect('equal')
    axs[1].grid('on', color='k', linestyle='--', alpha=.5)
    axs[1].tick_params(direction='in')
    axs[1].set_xlabel('RA')
    axs[1].set_ylabel('DEC')
    
    
    if len(df_spec_radial) > 0 and len(df_photoz_radial) > 0:
      df_match = fast_crossmatch(df_spec_radial, df_photoz_radial)
      df_photoz_good_with_spec = df_match[df_match.zml.between(*z_photo_range)]
      df_photoz_good_wo_spec = df_photoz_good[~df_photoz_good.idx.isin(df_match.idx)]
      df_photoz_bad_with_spec = df_match[~df_match.zml.between(*z_photo_range)]
      ra_col, dec_col = guess_coords_columns(df_photoz_radial)
      if len(df_photoz_good_wo_spec) > 0:
        axs[2].scatter(
          df_photoz_good_wo_spec[ra_col].values, 
          df_photoz_good_wo_spec[dec_col].values, 
          c='tab:olive', 
          s=6, 
          rasterized=True, 
          transform=axs[2].get_transform('icrs'),
          label=f'Good Photo Z wo/ Spec Z ({len(df_photoz_good_wo_spec)} obj)'
        )
      if len(df_photoz_bad_with_spec) > 0:
        axs[2].scatter(
          df_photoz_bad_with_spec[ra_col].values, 
          df_photoz_bad_with_spec[dec_col].values, 
          c='tab:orange', 
          s=6, 
          rasterized=True, 
          transform=axs[2].get_transform('icrs'),
          label=f'Bad Photo Z w/ Spec Z ({len(df_photoz_bad_with_spec)} obj)'
        )
      if len(df_photoz_good_with_spec) > 0:
        axs[2].scatter(
          df_photoz_good_with_spec[ra_col].values, 
          df_photoz_good_with_spec[dec_col].values, 
          c='tab:blue', 
          s=6, 
          rasterized=True, 
          transform=axs[2].get_transform('icrs'),
          label=f'Good Photo Z w/ Spec Z ({len(df_photoz_good_with_spec)} obj)'
        )
    self.add_cluster_center(cls_ra, cls_dec, axs[2])
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      r15Mpc_deg=cls_15Mpc_deg,
      ax=axs[2]
    )
    axs[2].set_title(f'Photo Z $\\cap$ Spec Z (CrossMatch Distance: 1 arcsec)')
    axs[2].invert_xaxis()
    axs[2].legend(loc='upper left')
    axs[2].set_aspect('equal')
    axs[2].grid('on', color='k', linestyle='--', alpha=.5)
    axs[2].tick_params(direction='in')
    axs[2].set_xlabel('RA')
    axs[2].set_ylabel('DEC')
    
    title = (
      f'Cluster: {cls_name} (RA: {cls_ra:.5f}, DEC: {cls_dec:.5f})\n'
      f'Search Radius: 15Mpc = {cls_15Mpc_deg:.3f}$^\\circ$ ($z_{{cluster}}={cls_z:.4f}$)\n'
      f'Spec Z Range: [$z_{{cluster}} - 0.007$, $z_{{cluster}} + 0.007$] = [{z_spec_range[0]:.4f}, {z_spec_range[1]:.4f}]\n'
      f'Good Photo Z: [$z_{{cluster}} - 0.015$, $z_{{cluster}} + 0.015$] = [{z_photo_range[0]:.4f}, {z_photo_range[1]:.4f}]\n'
      f'R Mag Range: [13, 22] | Spec Class = GALAXY*\n'
    )
    fig.suptitle(title, size=18)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)




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



def download_legacy_pipeline():
  df_clusters = load_clusters()
  df_index = load_index()
  ls10_pipe = Pipeline(
    LoadClusterInfoStage(df_clusters, df_index),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', overwrite=False, workers=8)
  )
  ls10_pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)


def download_legacy_erass_pipeline():
  df_clusters = load_eRASS()
  ls10_pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', overwrite=False, workers=8)
  )
  ls10_pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1)



def spec_plots_pipeline():
  df_clusters = load_clusters()
  df_index = load_index()
  df_photoz, photoz_skycoord = load_photoz()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadClusterInfoStage(df_clusters, df_index),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    # RadialSearchStage('df_photoz', 'df_photoz_radial', 'cls_15Mpc_deg', 'outputs_v6/photo', 'photo'),
    # RadialSearchStage('df_spec', 'df_spec_radial', 'cls_15Mpc_deg', 'outputs_v6/spec', 'spec'),
    ClusterPlotStage(overwrite=False)
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
  



def main():
  # prepare_idr5()
  # download_legacy_erass_pipeline()
  spec_plots_pipeline()
  # erass_plots_pipeline()
  # print(load_clusters())
  
  
  
if __name__ == '__main__':
  main()