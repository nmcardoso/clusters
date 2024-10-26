from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np
import pandas as pd
from astromodule.io import merge_pdf, read_table, write_table
from astromodule.pipeline import Pipeline, PipelineStage, PipelineStorage
from astromodule.table import (concat_tables, crossmatch, fast_crossmatch,
                               guess_coords_columns, radial_search, selfmatch)
from astropy import units as u
from astropy.coordinates import SkyCoord

from splusclusters.configs import configs
from splusclusters.loaders import remove_bad_objects
from splusclusters.utils import Timming


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
        df_search.class_spec.str.upper().str.startswith('GALAXY') &
        df_search.f_z.str.upper().str.startswith('KEEP')
      ]
    elif self.kind == 'photo':
      if 'r_auto' in df_search.columns:
        df_search = df_search[
          # df_search.zml.between(*z_photo_range) &
          df_search.r_auto.between(*configs.MAG_RANGE)
        ]
    
    print(f'Radial search finished. Elapsed time: {t.end()}')
    
    if self.save_folder:
      table_name = f'{cls_name}.parquet'
      write_table(df_search, self.save_folder / table_name)
      print(f'Table "{table_name}" saved')



class SpecZRadialSearchStage(RadialSearchStage):
  def __init__(
    self, 
    save_folder: str | Path = None, 
    radius_key: str = 'cls_search_radius_deg', 
    overwrite: bool = False,
  ):
    if save_folder is None:
      save_folder = configs.SPECZ_FOLDER
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
    save_folder: str | Path = configs.PHOTOZ_FOLDER, 
    radius_key: str = 'cls_search_radius_deg', 
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



class LegacyRadialSearchStage(RadialSearchStage):
  def __init__(
    self, 
    save_folder: str | Path = configs.LEG_PHOTO_FOLDER, 
    radius_key: str = 'cls_search_radius_deg', 
    overwrite: bool = False,
  ):
    super().__init__(
      df_name='df_legacy',
      radius_key=radius_key, 
      save_folder=save_folder, 
      kind='photo', 
      overwrite=overwrite, 
      skycoord_name='legacy_skycoord',
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
  
  

class PhotozSpeczLegacyMatchStage(PipelineStage):
  def __init__(self, overwrite: bool = False):
    self.overwrite = overwrite
    
  
  def run(
    self, 
    cls_name: str, 
    df_specz_radial: pd.DataFrame,
    df_photoz_radial: pd.DataFrame, 
    df_legacy_radial: pd.DataFrame,
    df_ret: pd.DataFrame | None,
  ):
    out_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}.parquet'
    out_flags_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}+flags.parquet'
    if out_path.exists() and not self.overwrite:
      return
    
    df_spec = df_specz_radial.copy()
    df_photo = df_photoz_radial.copy()
    df_legacy = df_legacy_radial.copy()
    df_r = df_ret.copy()
    
    if len(df_spec) > 0:
      ra, dec = guess_coords_columns(df_spec)
      df_spec = df_spec.rename(columns={ra: 'ra_spec', dec: 'dec_spec'})
    if len(df_photo) > 0:
      ra, dec = guess_coords_columns(df_photo)
      df_photo = df_photo.rename(columns={ra: 'ra_photo', dec: 'dec_photo'})
    if len(df_legacy) > 0:
      ra, dec = guess_coords_columns(df_legacy)
      df_legacy = df_legacy.rename(columns={ra: 'ra_legacy', dec: 'dec_legacy'})
    if len(df_r) > 0:
      ra, dec = guess_coords_columns(df_r)
      df_r = df_r.rename(columns={ra: 'ra_r', dec: 'dec_r'})
    
    df_spec['f_z'] = df_spec['f_z'].astype('str')
    df_spec['original_class_spec'] = df_spec['original_class_spec'].astype('str')
    
    print('Photo-z objects:', len(df_photo))
    print('Spec-z objects:', len(df_spec))
    print('Legacy objects:', len(df_legacy))
    
    
    df = None
    t = Timming()
    if df_ret is not None and len(df_ret) > 0 and len(df_photo) > 0:
      print('Crossmatch 1: photo-z UNION spec-members')
      print('spec-members columns:')
      print(*df_r.columns, sep=', ')
      print('photo-z columns:')
      print(*df_photo, sep=', ')
      df = concat_tables([df_r, df_photo])
      # df = crossmatch(
      #   table1=df_r,
      #   table2=df_photo,
      #   join='1or2',
      #   ra1='ra_r',
      #   dec1='dec_r',
      #   ra2='ra_photo',
      #   dec2='dec_photo',
      # )
      df = df[[*df_photo.columns, *df_r.columns]]
      df.insert(0, 'ra_final', np.nan)
      df.insert(1, 'dec_final', np.nan)
      df['ra_final'] = df['ra_final'].fillna(df['ra_r'])
      df['ra_final'] = df['ra_final'].fillna(df['ra_photo'])
      df['dec_final'] = df['dec_final'].fillna(df['dec_r'])
      df['dec_final'] = df['dec_final'].fillna(df['dec_photo'])
      # df = selfmatch(df, 1*u.arcsec, 'keep0', 'ra_final', 'dec_final', fmt='csv')
      
      for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'int64[pyarrow]':
          df[col].replace(r'^\s*$', np.nan, regex=True, inplace=True)
          df[col] = df[col].astype('int32')
        if df[col].dtype == 'float64' or df[col].dtype == 'float64[pyarrow]' or df[col].dtype == 'float[pyarrow]' or df[col].dtype == 'double[pyarrow]':
          df[col].replace(r'^\s*$', np.nan, regex=True, inplace=True)
          df[col] = df[col].astype('float64')
          
      # del df['ra_r']
      # del df['ra_photo']
      # del df['dec_r']
      # del df['dec_photo']
    else:
      pass
    
    print('\ncolumns after match:')
    print(*df, sep=', ')
    print(f'\nCrossmatch 1 finished. Duration: {t.end()}')
    print('Objects with photo-z only:', len(df[~df.zml.isna() & df.z.isna()]))
    print('Objects with spec-z only:', len(df[df.zml.isna() & ~df.z.isna()]))
    print('Objects with photo-z and spec-z:', len(df[~df.zml.isna() & ~df.z.isna()]))
    print('Total of objects after first match:', len(df))
    
    
    t = Timming()
    print('Crossmatch 2: match LEFT OUTER JOIN legacy')
    if df is not None and len(df_legacy) > 0:
      print('legacy columns:')
      print(*df_legacy.columns, sep=', ')
      df = crossmatch(
        table1=df,
        table2=df_legacy,
        join='all1',
        ra1='ra_final',
        dec1='dec_final',
        ra2='ra_legacy',
        dec2='dec_legacy',
        fmt='csv',
      )
      # df['ra_final'] = df['ra_final'].fillna(df['ra_legacy'])
      # df['dec_final'] = df['dec_final'].fillna(df['dec_legacy'])
      # del df['ra_legacy']
      # del df['dec_legacy']
    else:
      df['type'] = np.nan
      df['mag_r'] = np.nan
      
    print('\ncolumns after match:')
    print(*df, sep=', ')
    print(f'\nCrossmatch 2 finished. Duration: {t.end()}')
    print('Objects with legacy morpho:', len(df[~df.type.isna() | (df.type != '')]))
    print('Objects without legacy morpho:', len(df[df.type.isna() | (df.type == '')]))
    print(
      'Galaxies:', len(df[(df.type != 'PSF') & (df.type != '')]), 
      ', Stars:', len(df[df.type == 'PSF']), 
      ', Unknown:', len(df[df.type.isna() | (df.type == '')])
    )
    print('Number of objects after second match:', len(df))
    print(df)
    df = df[df.type != 'PSF']
    print('Number of objects after PSF removal:', len(df))
    
    
    
    t = Timming()
    print('\nCrossmatch 3: match LEFT OUTER JOIN spec-z-all')
    df_spec_all = self.get_data('df_spec')
    if df is not None and df_spec_all is not None:
      print('spec all columns:')
      print(*df_spec_all.columns, sep=', ')
      n_redshift = len(df[~df.z.isna()])
      df_spec_all['f_z'] = df_spec_all['f_z'].astype('str')
      df_spec_all['original_class_spec'] = df_spec_all['original_class_spec'].astype('str')
      # spec_all_ra, spec_all_dec = guess_coords_columns(df_spec_all)
      df = crossmatch(
        df,
        df_spec_all,
        ra1='ra_final',
        dec1='dec_final',
        suffix1='_final',
        ra2='ra_spec_all',
        dec2='dec_spec_all',
        suffix2='_spec_all',
        join='all1',
        fmt='csv',
      )
      cols = [
        'z', 'e_z', 'f_z', 'class_spec',
        'original_class_spec', 'source'
      ]
      for col in cols:
        if f'{col}_final' in df.columns:
          df[f'{col}_final'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
          df[f'{col}_final'].fillna(df[f'{col}_spec_all'], inplace=True)
          df.rename(columns={f'{col}_final': col}, inplace=True)
        if f'{col}_spec_all' in df.columns:
          del df[f'{col}_spec_all']
      df['f_z'] = df['f_z'].astype('str')
      df['original_class_spec'] = df['original_class_spec'].astype('str')
      # del df[spec_all_ra]
      # del df[spec_all_dec]
    
    print('\ncolumns after match:')
    print(*df, sep=', ')
    print(f'\nCrossmatch 3 finished. Duration: {t.end()}')
    print('Inserted redshifts:', len(df[~df.z.isna()]) - n_redshift)
    print('Number of objects:', len(df))
    
    
    if 'flag_member' in df.columns:
      df.loc[~df.flag_member.isin([0, 1]), 'flag_member'] = -1

    df = df.rename(columns={'ra_final': 'ra', 'dec_final': 'dec'})
    # photoz_cols = ['ra_photo', 'dec_photo', 'zml', 'odds']
    # if 'r_auto' in df.columns:
    #   photoz_cols.append('r_auto')
    # if 'field' in df.columns:
    #   photoz_cols.append('field')
    # specz_cols = ['z', 'e_z', 'f_z', 'class_spec']
    # legacy_cols = ['mag_r', 'type']
    # cols = photoz_cols + specz_cols + legacy_cols
    # df = df[cols]

    # df = df[~df.original_class_spec.isin(['GClstr', 'GGroup', 'GPair', 'GTrpl', 'PofG'])]
    # print('Number of objects after original_class_spec filter:', len(df))

    # df = df[(df['f_z'] != 'KEEP(    )') & (df['e_z'] != 3.33E-4)]
    # print('Number of objects after flag z filter:', len(df))
    
    if 'xmatch_sep' in df.columns:
      del df['xmatch_sep']
    if 'xmatch_sep_1' in df.columns:
      del df['xmatch_sep_1']
    if 'xmatch_sep_2' in df.columns:
      del df['xmatch_sep_2']
    if 'xmatch_sep_final' in df.columns:
      del df['xmatch_sep_final']
      
    
    # Filter bad objects after visual inspection
    print('\nRemoving bad objects classified by visual inspection')
    l = len(df)
    df = remove_bad_objects(df)
    print('Number of objects before filter:', l)
    print('Number of objects after filter:', len(df))
    
    
    # Flag: remove_z
    df['remove_z'] = 0
    mask = (
      df.original_class_spec.isin(['GClstr', 'GGroup', 'GPair', 'GTrpl', 'PofG']) |
      ((df['f_z'] == 'KEEP(    )') & (df['e_z'] == 3.33E-4))
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
            (df.PROB_GAL_GAIA < prob)
          )
          df.loc[mask, 'remove_star'] = 1
    df['remove_star'] = df['remove_star'].astype('int32')
    
    
    # Flag: remove_neighbours
    df['remove_neighbours'] = 0
    if 'mag_r' in df.columns and 'source' in df.columns and 'z' in df.columns:
      df = selfmatch(df, 10*u.arcsec, 'identify')
      
      if 'GroupID' in df.columns:
        gids = df['GroupID'].unique()
        gids = gids[gids > 0]
        for group_id in gids:
          group_df = df[(df['GroupID'] == group_id) & (df['remove_z'] != 1)]
          
          if len(group_df[group_df.flag_member.isin([0, 1])]) > 0:
            z_mask = ~group_df.flag_member.isin([0, 1])
          
          elif len(group_df[~group_df.z.isna()]) == 1:
            if len(group_df[~group_df.mag_r.isna()]) > 0:
              z_mask = group_df.z.isna() | (group_df.mag_r != group_df.mag_r.min())
            else:
              z_mask = group_df.z.isna()
              
          elif len(group_df[~group_df.z.isna()]) > 1:
            if len(group_df[group_df.source.str.upper().str.contains('SDSSDR18_SDSS')]) == 1:
              z_mask = ~group_df.source.str.upper().str.contains('SDSSDR18_SDSS')
            elif len(group_df[group_df.source.str.upper().str.contains('SDSSDR18_SDSS')]) > 1:
              if len(group_df[~group_df.mag_r.isna()]) > 0:
                z_mask = (
                  group_df.mag_r != group_df[group_df.source.str.upper().str.contains('SDSSDR18_SDSS')].mag_r.min()
                )
              else:
                if len(~group_df.e_z.isna()) > 0:
                  z_mask = (
                    group_df.e_z != group_df[group_df.source.str.upper().str.contains('SDSSDR18_SDSS')].e_z.min()
                  )
                else:
                  z_mask = np.zeros(shape=(len(group_df),), dtype=np.bool)
            else:
              if len(group_df[~group_df.mag_r.isna()]) > 0:
                z_mask = group_df.mag_r.isna() | (group_df.mag_r != group_df.mag_r.min())
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

          df.loc[group_df[z_mask].index, 'remove_neighbours'] = 1
    df['remove_neighbours'] = df['remove_neighbours'].astype('int32')
    
    
    # Flag: remove_radius
    df['remove_radius'] = 0
    if 'PETRO_RADIUS' in df.columns and 'mag_r' in df.columns and 'A' in df.columns and 'B' in df.columns:
      df.loc[(df.PETRO_RADIUS == 0) & df.mag_r.isna() & df.z.isna(), 'remove_radius'] = 1
      df.loc[(df.PETRO_RADIUS == df.PETRO_RADIUS.max()) & df.mag_r.isna() & df.z.isna(), 'remove_radius'] = 1
      df.loc[((df.A < 1.5e-4) | (df.B < 1.5e-4)) & df.mag_r.isna() & df.z.isna(), 'remove_radius'] = 1
    df['remove_radius'] = df['remove_radius'].astype('int32')
    
    if 'GroupSize' in df.columns:
      del df['GroupSize']
    
    
    print('\nFinal columns:')
    print(*df.columns, sep=', ')
    
    write_table(df, out_flags_path)
    
    df = df[(df.remove_star != 1) & (df.remove_z != 1) & (df.remove_neighbours != 1) & (df.remove_radius != 1)]
    del df['remove_star']
    del df['remove_z']
    del df['remove_neighbours']
    del df['remove_radius']
    if 'GroupID' in df.columns:
      del df['GroupID']
      
    write_table(df, out_path)
  
  
    
  def run_old(
    self, 
    cls_name: str, 
    df_specz_radial: pd.DataFrame,
    df_photoz_radial: pd.DataFrame, 
    df_legacy_radial: pd.DataFrame,
    df_ret: pd.DataFrame | None,
  ):
    out_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}.parquet'
    out_flags_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}+flags.parquet'
    if out_path.exists() and not self.overwrite:
      return
    
    df_spec = df_specz_radial.copy()
    df_photo = df_photoz_radial.copy()
    df_legacy = df_legacy_radial.copy()
    df_r = df_ret.copy()
    
    if len(df_spec) > 0:
      ra, dec = guess_coords_columns(df_spec)
      df_spec = df_spec.rename(columns={ra: 'ra_spec', dec: 'dec_spec'})
    if len(df_photo) > 0:
      ra, dec = guess_coords_columns(df_photo)
      df_photo = df_photo.rename(columns={ra: 'ra_photo', dec: 'dec_photo'})
    if len(df_legacy) > 0:
      ra, dec = guess_coords_columns(df_legacy)
      df_legacy = df_legacy.rename(columns={ra: 'ra_legacy', dec: 'dec_legacy'})
    if len(df_r) > 0:
      ra, dec = guess_coords_columns(df_r)
      df_r = df_r.rename(columns={ra: 'ra_r', dec: 'dec_r', 'z': 'z_r'})
    
    df_spec['f_z'] = df_spec['f_z'].astype('str')
    df_spec['original_class_spec'] = df_spec['original_class_spec'].astype('str')
    
    print('Photo-z objects:', len(df_photo))
    print('Spec-z objects:', len(df_spec))
    print('Legacy objects:', len(df_legacy))
    
    
    
    t = Timming()
    print('Crossmatch 1: photo-z UNION spec-z')
    if df_ret is not None and len(df_ret) > 0 and len(df_photo) > 0:
      df_spec_union = crossmatch(
        table1=df_r,
        table2=df_spec,
        join='1or2',
        ra1='ra_r',
        dec1='dec_r',
        ra2='ra_spec',
        dec2='dec_spec',
      )
      df_spec_union['z'] = df_spec_union['z'].fillna(df_spec_union['z_r'])
      del df_spec_union['z_r']
      df_spec_union['e_z'] = df_spec_union['e_z'].fillna(df_spec_union['z_err'])
      del df_spec_union['z_err']
      df_spec_union['ra_spec'] = df_spec_union['ra_spec'].fillna(df_spec_union['ra_r'])
      del df_spec_union['ra_r']
      df_spec_union['dec_spec'] = df_spec_union['dec_spec'].fillna(df_spec_union['dec_r'])
      del df_spec_union['dec_r']
      df = crossmatch(
        table1=df_photo,
        table2=df_spec_union,
        join='1or2',
        ra1='ra_photo',
        dec1='dec_photo',
        ra2='ra_spec',
        dec2='dec_spec',
      )
      df['ra_photo'] = df['ra_photo'].fillna(df['ra_spec'])
      df['dec_photo'] = df['dec_photo'].fillna(df['dec_spec'])
    elif len(df_photo) > 0 and len(df_spec) > 0:
      df = crossmatch(
        table1=df_photo,
        table2=df_spec,
        join='1or2',
        ra1='ra_photo',
        dec1='dec_photo',
        ra2='ra_spec',
        dec2='dec_spec',
      )
      df['ra_photo'] = df['ra_photo'].fillna(df['ra_spec'])
      df['dec_photo'] = df['dec_photo'].fillna(df['dec_spec'])
    elif len(df_photo) == 0 and len(df_spec) > 0:
      df = df_spec
      df['ra_photo'] = df['ra_spec']
      df['dec_photo'] = df['dec_spec']
      df['zml'] = np.nan
      df['odds'] = np.nan
    elif len(df_photo) > 0 and len(df_spec) == 0:
      df = df_photo
      df['z'] = np.nan
      df['e_z'] = np.nan
      df['f_z'] = np.nan
      df['class_spec'] = np.nan
    elif len(df_photo) == 0 and len(df_spec) == 0:
      return
    
    if 'ra_spec' in df.columns:
      del df['ra_spec']
      del df['dec_spec']
    
    print(f'Crossmatch 1 finished. Duration: {t.end()}')
    print('Objects with photo-z only:', len(df[~df.zml.isna() & df.z.isna()]))
    print('Objects with spec-z only:', len(df[df.zml.isna() & ~df.z.isna()]))
    print('Objects with photo-z and spec-z:', len(df[~df.zml.isna() & ~df.z.isna()]))
    print('Total of objects after first match:', len(df))
    print('Crossmatch 2: match-1 LEFT OUTER JOIN legacy')
    
    t = Timming()
    if len(df_legacy) > 0:
      print(df_legacy)
      df = crossmatch(
        table1=df,
        table2=df_legacy,
        join='all1',
        ra1='ra_photo',
        dec1='dec_photo',
        ra2='ra_legacy',
        dec2='dec_legacy',
      )
      df['ra_photo'] = df['ra_photo'].fillna(df['ra_legacy'])
      df['dec_photo'] = df['dec_photo'].fillna(df['dec_legacy'])
      del df['ra_legacy']
      del df['dec_legacy']
    else:
      df['type'] = np.nan
      df['mag_r'] = np.nan
    
    print(f'Crossmatch 2 finished. Duration: {t.end()}')
    print('Objects with legacy morpho:', len(df[~df.type.isna() | (df.type != '')]))
    print('Objects without legacy morpho:', len(df[df.type.isna() | (df.type == '')]))
    print(
      'Galaxies:', len(df[(df.type != 'PSF') & (df.type != '')]), 
      ', Stars:', len(df[df.type == 'PSF']), 
      ', Unknown:', len(df[df.type.isna() | (df.type == '')])
    )
    print('Number of objects after second match:', len(df))
    print(df)
    
    df = df[df.type != 'PSF']
    
    print('Number of objects after PSF removal:', len(df))
    
      
    t = Timming()
    n_redshift = len(df[~df.z.isna()])
    print('\nCrossmatch 3: final LEFT OUTER JOIN spec-z')
    df_spec_all = self.get_data('df_spec')
    df_spec_all['f_z'] = df_spec_all['f_z'].astype('str')
    df_spec_all['original_class_spec'] = df_spec_all['original_class_spec'].astype('str')
    spec_all_ra, spec_all_dec = guess_coords_columns(df_spec_all)
    df = crossmatch(
      df,
      df_spec_all,
      ra1='ra_photo',
      dec1='dec_photo',
      suffix1='_final',
      ra2=spec_all_ra,
      dec2=spec_all_dec,
      suffix2='_spec_all',
      join='all1',
    )
    cols = [
      'z', 'e_z', 'f_z', 'class_spec',
      'original_class_spec', 'source'
    ]
    for col in cols:
      if f'{col}_final' in df.columns:
        df[f'{col}_final'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df[f'{col}_final'].fillna(df[f'{col}_spec_all'], inplace=True)
        df.rename(columns={f'{col}_final': col}, inplace=True)
      if f'{col}_spec_all' in df.columns:
        del df[f'{col}_spec_all']
    df['f_z'] = df['f_z'].astype('str')
    df['original_class_spec'] = df['original_class_spec'].astype('str')
    del df[spec_all_ra]
    del df[spec_all_dec]
    print(f'Crossmatch 3 finished. Duration: {t.end()}')
    print('Inserted redshifts:', len(df[~df.z.isna()]) - n_redshift)
    print('Number of objects:', len(df))
    
    
    if 'flag_member' in df.columns:
      df.loc[~df.flag_member.isin([0, 1]), 'flag_member'] = -1

    df = df.rename(columns={'ra_photo': 'ra', 'dec_photo': 'dec'})
    # photoz_cols = ['ra_photo', 'dec_photo', 'zml', 'odds']
    # if 'r_auto' in df.columns:
    #   photoz_cols.append('r_auto')
    # if 'field' in df.columns:
    #   photoz_cols.append('field')
    # specz_cols = ['z', 'e_z', 'f_z', 'class_spec']
    # legacy_cols = ['mag_r', 'type']
    # cols = photoz_cols + specz_cols + legacy_cols
    # df = df[cols]

    # df = df[~df.original_class_spec.isin(['GClstr', 'GGroup', 'GPair', 'GTrpl', 'PofG'])]
    # print('Number of objects after original_class_spec filter:', len(df))

    # df = df[(df['f_z'] != 'KEEP(    )') & (df['e_z'] != 3.33E-4)]
    # print('Number of objects after flag z filter:', len(df))
    
    if 'xmatch_sep' in df.columns:
      del df['xmatch_sep']
    if 'xmatch_sep_1' in df.columns:
      del df['xmatch_sep_1']
    if 'xmatch_sep_2' in df.columns:
      del df['xmatch_sep_2']
    if 'xmatch_sep_final' in df.columns:
      del df['xmatch_sep_final']
      
    
    # Filter bad objects after visual inspection
    print('\nRemoving bad objects classified by visual inspection')
    l = len(df)
    df = remove_bad_objects(df)
    print('Number of objects before filter:', l)
    print('Number of objects after filter:', len(df)) 
    
    
    # Flag: remove_z
    df['remove_z'] = 0
    mask = (
      df.original_class_spec.isin(['GClstr', 'GGroup', 'GPair', 'GTrpl', 'PofG']) |
      ((df['f_z'] == 'KEEP(    )') & (df['e_z'] == 3.33E-4))
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
            (df.PROB_GAL_GAIA < prob)
          )
          df.loc[mask, 'remove_star'] = 1
    df['remove_star'] = df['remove_star'].astype('int32')
    
    
    # Flag: remove_neighbours
    df['remove_neighbours'] = 0
    if 'mag_r' in df.columns and 'source' in df.columns and 'z' in df.columns:
      df = selfmatch(df, 10*u.arcsec, 'identify')
      
      if 'GroupID' in df.columns:
        gids = df['GroupID'].unique()
        gids = gids[gids > 0]
        for group_id in gids:
          group_df = df[(df['GroupID'] == group_id) & (df['remove_z'] != 1)]
            
          if len(group_df[~group_df.z.isna()]) == 1:
            if len(group_df[~group_df.mag_r.isna()]) > 0:
              z_mask = group_df.z.isna() | (group_df.mag_r != group_df.mag_r.min())
            else:
              z_mask = group_df.z.isna()
              
          elif len(group_df[~group_df.z.isna()]) > 1:
            if len(group_df[group_df.source.str.upper().str.contains('SDSSDR18_SDSS')]) == 1:
              z_mask = ~group_df.source.str.upper().str.contains('SDSSDR18_SDSS')
            elif len(group_df[group_df.source.str.upper().str.contains('SDSSDR18_SDSS')]) > 1:
              if len(group_df[~group_df.mag_r.isna()]) > 0:
                z_mask = (
                  group_df.mag_r != group_df[group_df.source.str.upper().str.contains('SDSSDR18_SDSS')].mag_r.min()
                )
              else:
                if len(~group_df.e_z.isna()) > 0:
                  z_mask = (
                    group_df.e_z != group_df[group_df.source.str.upper().str.contains('SDSSDR18_SDSS')].e_z.min()
                  )
                else:
                  z_mask = np.zeros(shape=(len(group_df),), dtype=np.bool)
            else:
              if len(group_df[~group_df.mag_r.isna()]) > 0:
                z_mask = group_df.mag_r.isna() | (group_df.mag_r != group_df.mag_r.min())
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

          df.loc[group_df[z_mask].index, 'remove_neighbours'] = 1
    df['remove_neighbours'] = df['remove_neighbours'].astype('int32')
    
    
    # Flag: remove_radius
    df['remove_radius'] = 0
    if 'PETRO_RADIUS' in df.columns and 'mag_r' in df.columns and 'A' in df.columns and 'B' in df.columns:
      df.loc[(df.PETRO_RADIUS == 0) & df.mag_r.isna() & df.z.isna(), 'remove_radius'] = 1
      df.loc[(df.PETRO_RADIUS == df.PETRO_RADIUS.max()) & df.mag_r.isna() & df.z.isna(), 'remove_radius'] = 1
      df.loc[((df.A < 1.5e-4) | (df.B < 1.5e-4)) & df.mag_r.isna() & df.z.isna(), 'remove_radius'] = 1
    df['remove_radius'] = df['remove_radius'].astype('int32')
    
    if 'GroupSize' in df.columns:
      del df['GroupSize']
    
    
    
    
    print('\nFinal columns:', *df.columns)
    
    write_table(df, out_flags_path)
    
    df = df[(df.remove_star != 1) & (df.remove_z != 1) & (df.remove_neighbours != 1) & (df.remove_radius != 1)]
    del df['remove_star']
    del df['remove_z']
    del df['remove_neighbours']
    del df['remove_radius']
    if 'GroupID' in df.columns:
      del df['GroupID']
      
    write_table(df, out_path)