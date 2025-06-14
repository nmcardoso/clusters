from pathlib import Path
from typing import Dict, Literal, Sequence, Tuple

import dagster as dg
import numpy as np
import pandas as pd
from astromodule.io import merge_pdf, read_table, write_table
from astromodule.table import (concat_tables, crossmatch, fast_crossmatch,
                               guess_coords_columns, radial_search, selfmatch)
from astropy import units as u
from astropy.coordinates import SkyCoord

from splusclusters._loaders import ClusterInfo
from splusclusters.configs import configs
from splusclusters.loaders import remove_bad_objects
from splusclusters.utils import Timming, cond_overwrite, return_table_if_exists

PHOTOZ_COLUMNS = [
  'RA', 'DEC', 'A', 'B', 'THETA', 'ELLIPTICITY',
  'PETRO_RADIUS', 'FLUX_RADIUS_20', 'FLUX_RADIUS_50', 'FLUX_RADIUS_90', 
  'MU_MAX_g', 'MU_MAX_r', 'BACKGROUND_g', 'BACKGROUND_r',
  's2n_g_auto', 's2n_r_auto',
  # auto mags
  'J0378_auto', 'J0395_auto', 'J0410_auto', 'J0430_auto', 'J0515_auto',
  'J0660_auto', 'J0861_auto', 'g_auto', 'i_auto', 'r_auto', 'u_auto', 
  'z_auto',
  # PStotal mags
  'J0378_PStotal', 'J0395_PStotal', 'J0410_PStotal', 'J0430_PStotal', 
  'J0515_PStotal', 'J0660_PStotal', 'J0861_PStotal', 'g_PStotal', 
  'i_PStotal', 'r_PStotal', 'u_PStotal', 'z_PStotal',
  # aper_6 mags
  'J0378_aper_6', 'J0395_aper_6', 'J0410_aper_6', 'J0430_aper_6', 'J0515_aper_6',
  'J0660_aper_6', 'J0861_aper_6', 'g_aper_6', 'i_aper_6', 'r_aper_6', 'u_aper_6', 
  'z_aper_6',
  # auto error
  'e_J0378_auto', 'e_J0395_auto', 'e_J0410_auto', 'e_J0430_auto', 
  'e_J0515_auto', 'e_J0660_auto', 'e_J0861_auto', 'e_g_auto', 'e_i_auto', 
  'e_r_auto', 'e_u_auto', 'e_z_auto',
  # PStotal error
  'e_J0378_PStotal', 'e_J0395_PStotal', 'e_J0410_PStotal', 'e_J0430_PStotal', 
  'e_J0515_PStotal', 'e_J0660_PStotal', 'e_J0861_PStotal', 'e_g_PStotal', 
  'e_i_PStotal', 'e_r_PStotal', 'e_u_PStotal', 'e_z_PStotal',
  # aper_6 mags
  'e_J0378_aper_6', 'e_J0395_aper_6', 'e_J0410_aper_6', 'e_J0430_aper_6', 'e_J0515_aper_6',
  'e_J0660_aper_6', 'e_J0861_aper_6', 'e_g_aper_6', 'e_i_aper_6', 'e_r_aper_6', 'e_u_aper_6', 
  'e_z_aper_6',
  # G mags
  'g_aper_3', 'g_res', 'g_iso', 'g_petro', 
  # R mags
  'r_aper_3', 'r_res', 'r_iso', 'r_petro', 
  'Field'
]

SHIFTGAP_COLUMNS = [
  'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
  'radius_Mpc', 'v_offset', 'flag_member'
]



def _fix_coordinates(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  
  options = [
    ('ra', 'dec'), ('ra_r', 'dec_r'), ('ra_spec', 'dec_spec'), 
    ('ra_photo', 'dec_photo'), ('ra_legacy', 'dec_legacy'), 
    ('ra_spec_all', 'dec_spec_all')
  ]
  
  df['ra_aux'] = np.nan
  df['dec_aux'] = np.nan
    
  for ra_col, dec_col in options:
    if ra_col in df.columns and dec_col in df.columns:
      df['ra_aux'] = df.ra_aux.fillna(df[ra_col])
      df['dec_aux'] = df.dec_aux.fillna(df[dec_col]) 
  
  if 'ra' in df.columns:
    del df['ra']
  if 'dec' in df.columns:
    del df['dec']
  
  df.insert(0, 'ra', df.ra_aux)
  df.insert(1, 'dec', df.dec_aux)
  del df['ra_aux']
  del df['dec_aux']
  
  return df



def _remove_separation_columns(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  suffixes = ('', '_1', '_2', '_final', '_finala')
  for suffix in suffixes:
    col = f'xmatch_sep{suffix}'
    if col in df.columns: del df[col]
  
  suffixes = ('r', 'spec', 'photo', 'legacy', 'spec_all')
  for suffix in suffixes:
    ra_col, dec_col = f'ra_{suffix}', f'dec_{suffix}'
    if ra_col in df.columns: del df[ra_col]
    if dec_col in df.columns: del df[dec_col]
  return df
  


def _log_columns(df: pd.DataFrame, tab: int = 0):
  cols = ', '.join(df.columns)
  print(' ' * tab, '- Current columns: ', cols, ' (objects: ', len(df), ')', sep='')
  


def _sanitize_columns(df: pd.DataFrame):
  for col in df.columns:
    if df[col].dtype == 'int64' or df[col].dtype == 'int64[pyarrow]':
      df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).astype('int32')
    if (df[col].dtype == 'float64' or df[col].dtype == 'float64[pyarrow]' or 
        df[col].dtype == 'float[pyarrow]' or df[col].dtype == 'double[pyarrow]'):
      df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).astype('float64')
  df['flag_member'] = df.flag_member.fillna(-1)
  return df


  
  
def filter_by_visual_inspection(
  df: pd.DataFrame, 
  ra: str = None, 
  dec: str = None,
  info: ClusterInfo = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  print('\nRemoving bad objects classified by visual inspection')
  l = len(df)
  filter_df = read_table(
    configs.ROOT / 'tables' / 'objects_to_exclude.csv', comment='#'
  )
  df_rem = crossmatch(
    df, 
    filter_df,
    radius=1*u.arcsec, 
    join='1and2', 
    find='best',
    ra1=ra,
    dec1=dec,
    ra2='ra',
    dec2='dec',
    suffix1='', 
    suffix2='_rem'
  )
  
  if df_rem is not None and len(df_rem) > 0:
    if 'ra_rem' in df_rem.columns:
      del df_rem['ra_rem']
    if 'dec_rem' in df_rem.columns:
      del df_rem['dec_rem']
  
  df_match = crossmatch(
    df, 
    filter_df, 
    radius=1*u.arcsec, 
    join='1not2', 
    find='best',
    ra1=ra,
    dec1=dec,
    ra2='ra',
    dec2='dec',
  )
  
  print('Number of objects before visual inspection filter:', l)
  print('Number of objects after visual inspection filter:', len(df))
  return df_match, df_rem





def compute_angular_distance(
  df: pd.DataFrame, 
  info: ClusterInfo,
) -> pd.DataFrame:
  print('\n>> Computing angular distances')
  
  df = df.copy()
  
  coords = SkyCoord(ra=df['ra'].values, dec=df['dec'].values, unit=u.deg)
  
  df['radius_deg_computed'] = coords.separation(info.coord).deg
  if not 'radius_deg' in df.columns: df['radius_deg'] = np.nan
  df['radius_deg'] = df['radius_deg'].fillna(df['radius_deg_computed'])
  del df['radius_deg_computed']
  
  return df




def compute_cleanup_flags(
  df: pd.DataFrame,
  info: ClusterInfo,
  out_flags_path: Path,
  out_removed_path: Path,
) -> pd.DataFrame:
  print('\n>> Computing cleanup flags')
  df = df.copy()
  
  # Flag: redshift
  df['remove_z'] = 0
  mask = (
    df.original_class_spec.isin(['GClstr', 'GGroup', 'GPair', 'GTrpl', 'PofG']) |
    ((df.f_z == 'KEEP(    )') & (df.e_z == 3.33e-4)) | 
    ((info.z > 0.02) & (df.z < 0.002)) | (df.z <= 0)
  )
  df.loc[mask, 'remove_z'] = 1
  df['remove_z'] = df['remove_z'].astype('int32')
  
  
  # Flag: remove_star
  df['remove_star'] = 0
  if 'Field' in df.columns and 'r_auto' in df.columns and 'PROB_GAL_GAIA' in df.columns:
    prob_thresh = {
      'stripe82': [0.98, 0.98, 0.92, 0.52, 0.32, 0.16],
      'splus-s': [0.80, 0.50, 0.90, 0.70, 0.64, 0.42],
      'splus-n': [0.90, 0.64, 0.92, 0.72, 0.58, 0.30],
      'hydra': [0.90, 0.64, 0.92, 0.72, 0.58, 0.30],
    }
    r_range = [(0, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 99)]
    
    for tile, probs in prob_thresh.items():
      for r_auto, prob in zip(r_range, probs):
        mask = (
          df.Field.str.lower().str.startswith(tile) & 
          df.r_auto.between(*r_auto) &
          (df.PROB_GAL_GAIA < prob) &
          (df.class_spec != 'GALAXY')
        )
        df.loc[mask, 'remove_star'] = 1
  df['remove_star'] = df['remove_star'].astype('int32')
  
  
  # Flag: remove_radius
  df['remove_radius'] = 0
  if ('PETRO_RADIUS' in df.columns and 'mag_r' in df.columns and 
      'A' in df.columns and 'B' in df.columns):
    mask = (
      (df.PETRO_RADIUS == 0) & df.mag_r.isna() & df.z.isna()
    ) | (
      (df.PETRO_RADIUS == df.PETRO_RADIUS.max()) & df.mag_r.isna() & df.z.isna()
    ) | (
      ((df.A < 1.5e-4) | (df.B < 1.5e-4)) & df.mag_r.isna() & df.z.isna()
    )
    df.loc[mask, 'remove_radius'] = 1
  df['remove_radius'] = df['remove_radius'].astype('int32')
  
  
  # Flag: remove_neighbours
  df['remove_neighbours'] = 0
  if 'mag_r' in df.columns and 'source' in df.columns and 'z' in df.columns:
    df_no_flags = df[
      (df['remove_z'] != 1) & (df['remove_star'] != 1) &
      (df['remove_radius'] != 1)
    ]
    df = selfmatch(df_no_flags, 10*u.arcsec, 'identify')
    
    if 'GroupID' in df.columns:
      gids = df['GroupID'].unique()
      gids = gids[gids > 0]
      for group_id in gids:
        group_df = df[(df['GroupID'] == group_id)]
        
        if len(group_df[group_df.flag_member.isin([0, 1])]) > 0:
          z_mask = ~group_df.flag_member.isin([0, 1])
        
        elif len(group_df[~group_df.z.isna()]) == 1:
          z_mask = group_df.z.isna()
            
        elif len(group_df[~group_df.z.isna()]) > 1:
          sdss_mask = (
            group_df.source.str.upper().str.contains('SDSSDR18_SDSS').astype(np.bool) |
            group_df.source.str.upper().str.contains('2016SDSS').astype(np.bool) |
            group_df.source.str.upper().str.contains('DESI').astype(np.bool)
          )
          if len(group_df[sdss_mask]) == 1:
            z_mask = ~sdss_mask
          elif len(group_df[sdss_mask]) > 1:
            if len(group_df[~group_df.mag_r.isna()]) > 0:
              z_mask = (
                group_df.mag_r != group_df[sdss_mask].mag_r.min()
              )
            else:
              if len(~group_df.e_z.isna()) > 0:
                z_mask = (
                  group_df.e_z != group_df[sdss_mask].e_z.min()
                )
              else:
                z_mask = np.zeros(shape=(len(group_df),), dtype=np.bool)
          else:
            if len(group_df[~group_df.mag_r.isna()]) > 0:
              z_mask = (
                group_df.mag_r.isna() | (group_df.mag_r != group_df.mag_r.min())
              )
            else:
              if len(~group_df.e_z.isna()) > 0:
                z_mask = (group_df.e_z != group_df.e_z.min())
              else:
                z_mask = np.zeros(shape=(len(group_df),), dtype=np.bool)
        
        else:
          if len(group_df[~group_df.mag_r.isna()]) > 0:
            z_mask = group_df.mag_r != group_df.mag_r.min()
          else:
            z_mask = np.ones(shape=(len(group_df),), dtype=np.bool)
            # if len(group_df[(~group_df.r_auto.isna()) & (group_df.r_auto < 99)]) > 0:
            #   z_mask = group_df.r_auto != group_df.r_auto.min()
            # else:
            #   z_mask = np.ones(shape=(len(group_df),), dtype=np.bool)
            #   z_mask[0] = False

        df.loc[group_df[z_mask].index, 'remove_neighbours'] = 1
  df['remove_neighbours'] = df['remove_neighbours'].astype('int32')
  
  
  print('\nFinal columns: ', end='')
  print(*df.columns, sep=', ', end='')
  print(f' (objects: {len(df)})')
  
  write_table(df, out_flags_path)
  
  flag_mask = (
    (df.remove_star == 1) | (df.remove_z == 1) | 
    (df.remove_neighbours == 1) | (df.remove_radius == 1)
  )
  write_table(df[flag_mask], out_removed_path)
  
  df = df[~flag_mask]
  
  df['f_z'] = df['f_z'].fillna('')
  if df is not None and 'f_z' in df.columns: 
    print('\n\n>>>> KEEP 9:', len(df[df.f_z.str.contains('KEEP')]), '\n\n')
    
  del df['remove_star']
  del df['remove_z']
  del df['remove_neighbours']
  del df['remove_radius']
  if 'GroupID' in df.columns: del df['GroupID']  
  if 'GroupSize' in df.columns: del df['GroupSize']
  return df

  

def match_all(
  df_r: pd.DataFrame,
  df_spec: pd.DataFrame,
  df_photo: pd.DataFrame,
  df_legacy: pd.DataFrame,
  df_specz_outrange_radial: pd.DataFrame,
  cls_name: str,
  out_lost: Path,
) -> pd.DataFrame:
  if df_spec is not None and len(df_spec) > 0:
    if 'ra_spec_all' in df_spec.columns and 'dec_spec_all' in df_spec.columns:
      ra, dec = 'ra_spec_all', 'dec_spec_all'
    else:
      ra, dec = guess_coords_columns(df_spec)
    df_spec = df_spec.rename(columns={ra: 'ra_spec', dec: 'dec_spec'})
    print('Spec-z objects (in-range):', len(df_spec))
    
  if df_specz_outrange_radial is not None and len(df_specz_outrange_radial) > 0:
    ra, dec = guess_coords_columns(df_specz_outrange_radial)
    df_specz_outrange_radial = df_specz_outrange_radial.rename(columns={
      ra: 'ra_spec_outrange', dec: 'dec_spec_outrange'
    })
    print('Spec-z objects (outrange):', len(df_spec))
  
  if df_photo is not None and len(df_photo) > 0:
    ra, dec = guess_coords_columns(df_photo)
    df_photo = df_photo.rename(columns={ra: 'ra_photo', dec: 'dec_photo'})
    print('Photo-z objects:', len(df_photo))
  
  if df_legacy is not None and len(df_legacy) > 0:
    ra, dec = guess_coords_columns(df_legacy)
    df_legacy = df_legacy.rename(columns={ra: 'ra_legacy', dec: 'dec_legacy'})
    print('Legacy objects:', len(df_legacy))
  
  if df_r is not None and len(df_r) > 0:
    ra, dec = guess_coords_columns(df_r)
    df_r = df_r[[ra, dec, 'z', 'z_err', 'v', 'v_err', 'radius_deg', 'radius_Mpc', 'v_offset', 'flag_member']]
    df_r = df_r.rename(columns={ra: 'ra_r', dec: 'dec_r'})
    print('Return objects:', len(df_r))
  
  
  df_spec['f_z'] = df_spec['f_z'].astype('str')
  df_spec['original_class_spec'] = df_spec['original_class_spec'].astype('str')
  
  print()
  print('>> Step 1: combine previous spec and members determinations and current spec table')
  # df_r, df_r_rem = _filter_by_visual_inspection(df_r, 'ra_r', 'dec_r')
  # df_spec, df_spec_rem = _filter_by_visual_inspection(df_spec, 'ra_spec', 'dec_spec')
  if df_r is None and df_spec is not None:
    df = _fix_coordinates(df_spec)
    df['flag_member'] = np.nan
    df['radius_deg'] = np.nan
    print('   - Previous members determinations not found: using only current spec table')
    print(f'   - Spec objects: {len(df)}')
  elif df_spec is None and df_r is not None:
    df = _fix_coordinates(df_r)
    print('   - Current spec table not found: using only previous members determinations')
    print(f'   - Determinations objects: {len(df)}')
  else:
    print('   - Previous determinations AND spec table found: combining objects')
    print(f'   - Determinations objects: {len(df_r)}, spec objects: {len(df_spec)}')
    df_missing = crossmatch(
      table1=df_r,
      table2=df_spec,
      ra1='ra_r',
      dec1='dec_r',
      ra2='ra_spec',
      dec2='dec_spec',
      radius=1*u.arcsec,
      join='1not2',
      find='best',
    )
    if df_missing is not None:
      df_missing['f_z'] = 'KEEP'
      df = concat_tables([df_spec, df_missing])
      df = _fix_coordinates(df)
      print(f'   - Determinations objects without spec-z table correspondence: {len(df_missing)}')
      print(f'   - Number of objects after spec-z and missing determinations merge: {len(df)}')
    
    print('   - Including determination information for all ojects')
    df_result = crossmatch(
      table1=df,
      table2=df_r,
      ra1='ra',
      dec1='dec',
      ra2='ra_r',
      dec2='dec_r',
      radius=1*u.arcsec,
      join='all1',
      find='best',
    ) 
    
    if df_result is not None:
      df = df_result
      for col in df_r.columns:
        col1, col2 = f'{col}_1', f'{col}_2'
        if col1 in df.columns and col2 in df.columns:
          df[col] = df[col1].fillna(df[col2])
          del df[col1]
          del df[col2]
      print(f'   - Determination object combination done successfully, objects: {len(df)}')
  _log_columns(df, 3)
  
  
  print('\n')
  print('>> Step 2: add photo-z objects')
  if df_photo is not None and len(df_photo) > 0:
    print(f'   - S-PLUS catalog found, objects: {len(df_photo)}')
    if df is not None and len(df) > 0:
      df_result = crossmatch(
        table1=df_photo,
        table2=df,
        ra1='ra_photo',
        dec1='dec_photo',
        ra2='ra',
        dec2='dec',
        radius=1*u.arcsec,
        join='1or2',
        find='best',
      )
      if df_result is not None:
        print(f'   - Crossmatch against S-PLUS done successfully, objects: {len(df_result)}')
        df = _fix_coordinates(df_result)
    else:
      print('   - Crossmatch against S-PLUS returned a None object!')
      df = _fix_coordinates(df_photo)
  else:
    print('   - S-PLUS catalog not found: skiping')
  _log_columns(df, 3)
  
  
  print('\n')
  print('>> Step 3: add Legacy catalog information')
  if df_legacy is not None and len(df_legacy) > 0:
    print(f'   - Legacy catalog found, objects: {len(df_legacy)}')
    if df is not None and len(df) > 0:
      df_result = crossmatch(
        table1=df,
        table2=df_legacy,
        ra1='ra',
        dec1='dec',
        ra2='ra_legacy',
        dec2='dec_legacy',
        radius=1*u.arcsec,
        join='all1',
        find='best',
      )
      if df_result is not None:
        print(f'   - Crossmatch against Legacy done successfully, objects: {len(df_result)}')
        df = _fix_coordinates(df_result)
  else:
    print('   - Legacy catalog not found')
    df['mag_r'] = np.nan
  _log_columns(df, 3)
  
  
  print('\n')
  print('>> Step 4: add redshift information for objects outside z-spec range')
  if df_specz_outrange_radial is not None and len(df_specz_outrange_radial) > 0:
    if df is not None and len(df) > 0:
      df_result = crossmatch(
        table1=df,
        table2=df_specz_outrange_radial,
        ra1='ra',
        dec1='dec',
        ra2='ra_spec_outrange',
        dec2='dec_spec_outrange',
        radius=1*u.arcsec,
        join='all1',
        find='best',
        suffix1='_final',
        suffix2='_spec_outrange'
      )
      if df_result is not None:
        print(f'   - Crossmatch against outrange spec-z catalog done successfully, objects: {len(df_result)}')
        df = _fix_coordinates(df_result)
        cols = [
          'z', 'e_z', 'f_z', 'class_spec', 'original_class_spec', 'source'
        ]
  
        for col in cols:
          if f'{col}_final' in df.columns:
            df[f'{col}_final'] = df[f'{col}_final'].replace(r'^\s*$', np.nan, regex=True)
            df[f'{col}_final'] = df[f'{col}_final'].fillna(df[f'{col}_spec_outrange'])
            df = df.rename(columns={f'{col}_final': col})
          if f'{col}_spec_outrange' in df.columns:
            del df[f'{col}_spec_outrange']
            
        df['f_z'] = df['f_z'].astype('str').fillna('')
        df['original_class_spec'] = df['original_class_spec'].astype('str')
        
        if 'z_err' in df.columns:
          df['e_z'] = df.e_z.fillna(df.z_err)
        
        del df['ra_spec_outrange']
        del df['dec_spec_outrange']
        
  _log_columns(df, 3)
  
  print('\n')
  print('>> Step 5: Check for lost objects')
  if df_r is not None and len(df_r) > 0:
    print(f'   - Determinations catalog found, objects: {len(df_r)}')
    df_r_filt, _ = filter_by_visual_inspection(df_r, 'ra_r', 'dec_r')
    print(f'   - Number of objects of determinations catalog after VI filter: {len(df_r_filt)}')
    if df_r_filt is not None:
      df_lost = crossmatch(
        table1=df_r_filt, 
        table2=df, 
        ra1='ra_r', 
        dec1='dec_r', 
        ra2='ra', 
        dec2='dec', 
        join='1not2',
        find='best',
      )
      if df_lost is not None:
        df_lost = df_lost.rename(columns={'ra_r': 'ra', 'dec_r': 'dec'})
        df_lost['cluster'] = cls_name
        write_table(df_lost, out_lost)
        print('   - Number of lost objects: ', len(df_lost))
      else:
        if out_lost.exists():
          out_lost.unlink()
        print('   - Number of lost objects: None')
  else:
    print('   - Determinations catalog not found: skiping')
  _log_columns(df, 3)
  
  df = _remove_separation_columns(df)
  return df



def make_cluster_catalog(
  info: ClusterInfo,
  df_specz_radial: pd.DataFrame | None,
  df_photoz_radial: pd.DataFrame | None, 
  df_legacy_radial: pd.DataFrame | None,
  df_ret: pd.DataFrame | None,
  df_specz_outrange_radial: pd.DataFrame | None,
  overwrite: bool = False,
):
  out_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{info.name}.parquet'
  out_flags_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{info.name}+flags.parquet'
  out_removed_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{info.name}+removed.parquet'
  out_removed_vi_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{info.name}+removed_vi.parquet'
  out_lost = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{info.name}+lost.csv'
  
  with cond_overwrite(out_path, overwrite) as cm:
    df_spec = df_specz_radial.copy()
    df_photo = df_photoz_radial.copy()
    df_legacy = df_legacy_radial.copy()
    df_r = df_ret.copy() if df_ret is not None else None

    print('\n\n>>>> KEEP 1:', len(df_spec[df_spec.f_z.str.contains('KEEP')]), '\n\n')
    if df_r is not None and 'f_z' in df_r.columns: 
      print('\n\n>>>> KEEP 2:', len(df_r[df_r.f_z.str.contains('KEEP')]), '\n\n')
    
    # match all catalogs
    df = match_all(df_r, df_spec, df_photo, df_legacy, df_specz_outrange_radial, info.name, out_lost)

    # compute radius_deg for all objects
    df = compute_angular_distance(df, info)
    
    # Filter bad objects after visual inspection
    df, _ = filter_by_visual_inspection(df, ra='ra', dec='dec', info=info)
    
    # compute cleanup flags
    df = compute_cleanup_flags(df, info, out_flags_path, out_removed_path)
    
    df = _sanitize_columns(df)
    
    cm.write_table(df)
  return return_table_if_exists(out_path, df)