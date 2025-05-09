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
    df = self.get_data(self.df_name)
    if 'ra_spec_all' in df.columns:
      _ra = 'ra_spec_all'
      _dec = 'dec_spec_all'
    else:
      _ra = None
      _dec = None
    df_search = radial_search(
      position=pos, 
      table=df, 
      radius=radius*u.deg,
      cached_catalog=self.get_data(self.skycoord_name),
      ra=_ra,
      dec=_dec,
    )
    if _ra is not None:
      df_search = df_search.rename(columns={'ra_spec_all': 'ra', 'dec_spec_all': 'dec'})
    
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
    self.photo_columns = [
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
    self.returned_columns = [
      'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
      'radius_Mpc', 'v_offset', 'flag_member'
    ]
    
  
  def run(
    self, 
    cls_name: str, 
    cls_ra: float,
    cls_dec: float,
    df_specz_radial: pd.DataFrame,
    df_photoz_radial: pd.DataFrame, 
    df_legacy_radial: pd.DataFrame,
    df_ret: pd.DataFrame | None,
  ):
    out_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}.parquet'
    out_flags_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}+flags.parquet'
    out_recovered_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}+recovered.parquet'
    out_removed_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}+removed.parquet'
    out_removed_vi_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}+removed_vi.parquet'
    
    if out_path.exists() and not self.overwrite:
      return
    
    # if out_path.exists():
    #   df = read_table(out_path)
    #   if 'g_aper_3-r_aper_3' in df.columns:
    #     print('>> "g_aper_3-r_aper_3" columns found: skiping download')
    #     return
    #   else:
    #     del df
    
    center = SkyCoord(ra=cls_ra, dec=cls_dec, unit=u.deg)
    df_spec = df_specz_radial.copy()
    df_photo = df_photoz_radial.copy()
    df_legacy = df_legacy_radial.copy()
    df_r = df_ret.copy() if df_ret is not None else None
    
    if len(df_spec) > 0:
      if 'ra_spec_all' in df_spec.columns and 'dec_spec_all' in df_spec.columns:
        ra, dec = 'ra_spec_all', 'dec_spec_all'
      else:
        ra, dec = guess_coords_columns(df_spec)
      df_spec = df_spec.rename(columns={ra: 'ra_spec', dec: 'dec_spec'})
    if len(df_photo) > 0:
      ra, dec = guess_coords_columns(df_photo)
      df_photo = df_photo.rename(columns={ra: 'ra_photo', dec: 'dec_photo'})
    if len(df_legacy) > 0:
      ra, dec = guess_coords_columns(df_legacy)
      df_legacy = df_legacy.rename(columns={ra: 'ra_legacy', dec: 'dec_legacy'})
    if df_r is not None and len(df_r) > 0:
      ra, dec = guess_coords_columns(df_r)
      df_r = df_r.rename(columns={ra: 'ra_r', dec: 'dec_r'})
    
    df_spec['f_z'] = df_spec['f_z'].astype('str')
    df_spec['original_class_spec'] = df_spec['original_class_spec'].astype('str')
    
    print('Photo-z objects:', len(df_photo))
    print('Spec-z objects:', len(df_spec))
    print('Legacy objects:', len(df_legacy))
    
    
    df = None
    t = Timming()
    if df_r is not None and len(df_r) > 0 and df_photo is not None and len(df_photo) > 0:
      print('Crossmatch 1: photo-z UNION spec-members')
      print('spec-members columns:')
      print(*df_r.columns, sep=', ')
      print('photo-z columns:')
      print(*df_photo, sep=', ')
      df = crossmatch(
        df_r, 
        df_photo,
        ra1='ra_r',
        dec1='dec_r',
        ra2='ra_photo',
        dec2='dec_photo',
        join='all1'
      )
      
      df = concat_tables([df, df_photo])
      if df_r is not None:
        df = df[[*df_photo.columns, *df_r.columns]]
      else:
        df = df[[*df_photo.columns]]
      df.insert(0, 'ra_final', np.nan)
      df.insert(1, 'dec_final', np.nan)
      df['ra_final'] = df['ra_final'].fillna(df['ra_r'])
      df['ra_final'] = df['ra_final'].fillna(df['ra_photo'])
      df['dec_final'] = df['dec_final'].fillna(df['dec_r'])
      df['dec_final'] = df['dec_final'].fillna(df['dec_photo'])
      
      print(*df.columns, sep=', ')
      
      for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'int64[pyarrow]':
          df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).astype('int32')
        if df[col].dtype == 'float64' or df[col].dtype == 'float64[pyarrow]' or df[col].dtype == 'float[pyarrow]' or df[col].dtype == 'double[pyarrow]':
          df[col] = df[col].replace(r'^\s*$', np.nan, regex=True).astype('float64')
          
      del df['ra_r']
      del df['dec_r']
      del df['ra_photo']
      del df['dec_photo']
    elif df_r is not None and len(df_r) > 0:
      df = df_r
      df['ra_final'] = df_r['ra_r']
      df['dec_final'] = df_r['dec_r']
      for c in self.photo_columns:
        if c != 'RA' and c != 'DEC':
          df[c] = np.nan
    elif df_photo is not None and len(df_photo) > 0:
      df = df_photo
      df['ra_final'] = df_photo['ra_photo']
      df['dec_final'] = df_photo['dec_photo']
      for c in self.returned_columns:
        if c != 'ra' and c != 'dec':
          df[c] = np.nan
    elif df_spec is not None and len(df_spec) > 0:
      df = df_spec
      df['ra_final'] = df_spec['ra_spec']
      df['dec_final'] = df_spec['dec_spec']
      for c in self.returned_columns:
        if c != 'ra' and c != 'dec':
          df[c] = np.nan
    
    if df is not None:
      print('\ncolumns after match:')
      print(*df.columns, sep=', ')
      print(f'\nCrossmatch 1 finished. Duration: {t.end()}')
    
    if len(df_photo) > 0:
      print('Objects with photo-z only:', len(df[~df.zml.isna() & df.z.isna()]))
      print('Objects with spec-z only:', len(df[df.zml.isna() & ~df.z.isna()]))
      print('Objects with photo-z and spec-z:', len(df[~df.zml.isna() & ~df.z.isna()]))
    
    if df is not None:
      print('Total of objects after first match:', len(df))
    
    
    t = Timming()
    print('Crossmatch 2: match LEFT OUTER JOIN legacy')
    if df is not None and len(df_legacy) > 0:
      print('legacy columns:')
      print(*df_legacy.columns, sep=', ')
      df_result = crossmatch(
        table1=df,
        table2=df_legacy,
        join='all1',
        ra1='ra_final',
        dec1='dec_final',
        ra2='ra_legacy',
        dec2='dec_legacy',
      )
      if df_result is not None:
        df = df_result
        # df['ra_final'] = df['ra_final'].fillna(df['ra_legacy'])
        # df['dec_final'] = df['dec_final'].fillna(df['dec_legacy'])
        del df['ra_legacy']
        del df['dec_legacy']
      else:
        df['type'] = np.nan
        df['mag_r'] = np.nan
    else:
      df['type'] = np.nan
      df['mag_r'] = np.nan
      
    print('\ncolumns after match:')
    print(*df.columns, sep=', ')
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
      df_result = crossmatch(
        df,
        df_spec_all,
        ra1='ra_final',
        dec1='dec_final',
        suffix1='_final',
        ra2='ra_spec_all',
        dec2='dec_spec_all',
        suffix2='_spec_all',
        join='all1',
      )
      if df_result is not None:
        df = df_result
        cols = [
          'z', 'e_z', 'f_z', 'class_spec',
          'original_class_spec', 'source'
        ]
        for col in cols:
          if f'{col}_final' in df.columns:
            df[f'{col}_final'] = df[f'{col}_final'].replace(r'^\s*$', np.nan, regex=True)
            df[f'{col}_final'] = df[f'{col}_final'].fillna(df[f'{col}_spec_all'])
            df = df.rename(columns={f'{col}_final': col})
          if f'{col}_spec_all' in df.columns:
            del df[f'{col}_spec_all']
        df['f_z'] = df['f_z'].astype('str')
        df['original_class_spec'] = df['original_class_spec'].astype('str')
        del df['ra_spec_all']
        del df['dec_spec_all']
      
        df['f_z'] = df['f_z'].astype('str')
        df['original_class_spec'] = df['original_class_spec'].astype('str')
    
        print('\ncolumns after match:')
        print(*df.columns, sep=', ')
        print(f'\nCrossmatch 3 finished. Duration: {t.end()}')
        print('Inserted redshifts:', len(df[~df.z.isna()]) - n_redshift)
        print('Number of objects:', len(df))
    
    
    # fill flag_member
    if 'flag_member' in df.columns:
      df.loc[~df.flag_member.isin([0, 1]), 'flag_member'] = -1
    
    
    if df_r is not None:
      df_lost = crossmatch(
        table1=df_r, 
        table2=df, 
        ra1='ra_r', 
        dec1='dec_r', 
        ra2='ra_final', 
        dec2='dec_final', 
        join='1not2',
        find='all',
      )
      
      print('Lost Objects:')
      print(df_lost)
      
      if df_lost is not None and len(df_lost) > 0:
        if len(df_photo) > 0:
          df_lost = crossmatch(
            table1=df_lost, 
            table2=df_photo, 
            ra1='ra_r', 
            dec1='dec_r', 
            ra2='ra_photo', 
            dec2='dec_photo', 
            join='all1',
            suffix1='_final',
            suffix2='_photo'
          )
        
        if len(df_legacy) > 0:
          df_lost = crossmatch(
            table1=df_lost, 
            table2=df_legacy, 
            ra1='ra_r', 
            dec1='dec_r', 
            ra2='ra_legacy', 
            dec2='dec_legacy', 
            join='all1',
            suffix1='_final',
            suffix2='_legacy'
          )
        
        if len(df_spec_all) > 0:
          df_lost = crossmatch(
            table1=df_lost, 
            table2=df_spec_all, 
            ra1='ra_r', 
            dec1='dec_r', 
            ra2='ra_spec_all', 
            dec2='dec_spec_all', 
            join='all1',
            suffix1='_final',
            suffix2='_spec_all',
          )
        
        if not 'ra_final' in df_lost.columns:
          df_lost.insert(0, 'ra_final', np.nan)
        if not 'dec_final' in df_lost.columns:
          df_lost.insert(1, 'dec_final', np.nan)
        
        df_lost['ra_final'] = df_lost['ra_final'].fillna(df_lost['ra_r'])
        df_lost['dec_final'] = df_lost['dec_final'].fillna(df_lost['dec_r'])
        
        if 'ra_photo' in df_lost.columns:
          df_lost['ra_final'] = df_lost['ra_final'].fillna(df_lost['ra_photo'])
          df_lost['dec_final'] = df_lost['dec_final'].fillna(df_lost['dec_photo'])
          del df_lost['ra_photo']
          del df_lost['dec_photo']
        
        if 'z_spec_all' in df_lost.columns:
          df_lost['z_final'] = df_lost['z_final'].fillna(df_lost['z_spec_all'])
          df_lost = df_lost.rename(columns={'z_final': 'z'})
        
        if 'ra_legacy' in df_lost.columns:
          del df_lost['ra_legacy']
          del df_lost['dec_legacy']
        
        del df_lost['ra_r']
        del df_lost['dec_r']
        
        print(*df_lost.columns, sep=', ')
        
        write_table(df_lost, out_recovered_path)
        
        df = concat_tables([df, df_lost])
        
        if 'z_err' in df.columns:
          df['e_z'] = df['e_z'].fillna(df['z_err'])
          del df['z_err']
        if 'z_spec_all' in df.columns:
          del df['z_spec_all']
          del df['ra_spec_all']
          del df['dec_spec_all']
        
        df_lost = crossmatch(
          table1=df_r, 
          table2=df, 
          ra1='ra_r', 
          dec1='dec_r', 
          ra2='ra_final', 
          dec2='dec_final', 
          join='1not2',
          find='all',
        )
        print('\nLost objects (check):', len(df_lost))
      
      
    # compute radius_deg for all objects
    coords = SkyCoord(ra=df['ra_final'].values, dec=df['dec_final'].values, unit=u.deg)
    df['radius_deg_computed'] = coords.separation(center).deg
    df['radius_deg'] = df['radius_deg'].fillna(df['radius_deg_computed'])
    del df['radius_deg_computed']

    if 'ra' in df.columns:
      del df['ra']
    if 'dec' in df.columns:
      del df['dec']
    df.insert(0, 'ra', df['ra_final'].values)
    df.insert(1, 'dec', df['dec_final'].values)
    del df['ra_final']
    del df['dec_final']
    
    if 'xmatch_sep' in df.columns:
      del df['xmatch_sep']
    if 'xmatch_sep_1' in df.columns:
      del df['xmatch_sep_1']
    if 'xmatch_sep_2' in df.columns:
      del df['xmatch_sep_2']
    if 'xmatch_sep_final' in df.columns:
      del df['xmatch_sep_final']
    if 'xmatch_sep_finala' in df.columns:
      del df['xmatch_sep_finala']

    # colors
    # idx = df.columns.get_loc('Field')
    # if 'u_aper_6' in df.columns and 'r_aper_6' in df.columns:
    #   df.insert(idx + 1, 'u_aper_6-r_aper_6', df['u_aper_6'] - df['r_aper_6'])
    # if 'u_aper_6' in df.columns and 'r_aper_6' in df.columns:
    #   df.insert(idx + 2, 'u_aper_6-r_aper_6', df['u_aper_6'] - df['r_aper_6'])
    # if 'g_aper_6' in df.columns and 'r_aper_6' in df.columns:
    #   df.insert(idx + 1, 'g_aper_6-r_aper_6', df['g_aper_6'] - df['r_aper_6'])
    # if 'g_auto' in df.columns and 'r_auto' in df.columns:
    #   df.insert(idx + 2, 'g_auto-r_auto', df['g_auto'] - df['r_auto'])
    # if 'g_petro' in df.columns and 'r_petro' in df.columns:
    #   df.insert(idx + 3, 'g_petro-r_petro', df['g_petro'] - df['r_petro'])
    # if 'g_aper_3' in df.columns and 'r_aper_3' in df.columns:
    #   df.insert(idx + 4, 'g_aper_3-r_aper_3', df['g_aper_3'] - df['r_aper_3'])
    # if 'g_auto' in df.columns and 'r_auto' in df.columns and 'mag_g' in df.columns and 'mag_r' in df.columns:
    #   df.insert(idx + 5, 'g-r_auto-legacy', (df['g_auto'] - df['r_auto']) - (df['mag_g'] - df['mag_r']))
    # if 'g_aper_6' in df.columns and 'r_aper_6' in df.columns and 'mag_g' in df.columns and 'mag_r' in df.columns:
    #   df.insert(idx + 6, 'g-r_aper6-legacy', (df['g_aper_6'] - df['r_aper_6']) - (df['mag_g'] - df['mag_r']))
  
  
    # Filter bad objects after visual inspection
    print('\nRemoving bad objects classified by visual inspection')
    l = len(df)
    filter_df = read_table(configs.ROOT / 'tables' / 'objects_to_exclude.csv', comment='#')
    df_rem = crossmatch(df, filter_df, radius=1*u.arcsec, join='1and2', suffix1='', suffix2='_rem')
    if df_rem is not None and len(df_rem) > 0:
      del df_rem['ra_rem']
      del df_rem['dec_rem']
    write_table(df_rem, out_removed_vi_path)
    df = crossmatch(df, filter_df, radius=1*u.arcsec, join='1not2', find='all')
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
            (df.PROB_GAL_GAIA < prob) &
            (df.class_spec != 'GALAXY')
          )
          df.loc[mask, 'remove_star'] = 1
    df['remove_star'] = df['remove_star'].astype('int32')
    
    
    # Flag: remove_radius
    df['remove_radius'] = 0
    if 'PETRO_RADIUS' in df.columns and 'mag_r' in df.columns and 'A' in df.columns and 'B' in df.columns:
      df.loc[(df.PETRO_RADIUS == 0) & df.mag_r.isna() & df.z.isna(), 'remove_radius'] = 1
      df.loc[(df.PETRO_RADIUS == df.PETRO_RADIUS.max()) & df.mag_r.isna() & df.z.isna(), 'remove_radius'] = 1
      df.loc[((df.A < 1.5e-4) | (df.B < 1.5e-4)) & df.mag_r.isna() & df.z.isna(), 'remove_radius'] = 1
    df['remove_radius'] = df['remove_radius'].astype('int32')
    
    
    # Flag: remove_neighbours
    df['remove_neighbours'] = 0
    if 'mag_r' in df.columns and 'source' in df.columns and 'z' in df.columns:
      df_no_flags = df[(df['remove_z'] != 1) & (df['remove_star'] != 1) & (df['remove_radius'] != 1)]
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
              # if len(group_df[(~group_df.r_auto.isna()) & (group_df.r_auto < 99)]) > 0:
              #   z_mask = group_df.r_auto != group_df.r_auto.min()
              # else:
              #   z_mask = np.ones(shape=(len(group_df),), dtype=np.bool)
              #   z_mask[0] = False

          df.loc[group_df[z_mask].index, 'remove_neighbours'] = 1
    df['remove_neighbours'] = df['remove_neighbours'].astype('int32')
    
    
    print('\nFinal columns:')
    print(*df.columns, sep=', ')
    
    write_table(df, out_flags_path)
    
    
    df_rem = df[(df.remove_star == 1) | (df.remove_z == 1) | (df.remove_neighbours == 1) | (df.remove_radius == 1)]
    write_table(df_rem, out_removed_path)
    
    
    df = df[(df.remove_star != 1) & (df.remove_z != 1) & (df.remove_neighbours != 1) & (df.remove_radius != 1)]
    del df['remove_star']
    del df['remove_z']
    del df['remove_neighbours']
    del df['remove_radius']
    if 'GroupID' in df.columns:
      del df['GroupID']  
    if 'GroupSize' in df.columns:
      del df['GroupSize']
    write_table(df, out_path)




class FilterR200(PipelineStage):
  def __init__(self, overwrite: bool = False):
    self.overwrite = overwrite
  
  def run(
    self, 
    cls_name: str, 
    cls_r200_deg: float,
    df_all_radial: pd.DataFrame,
  ):
    out_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}+5r200.parquet'
    if out_path.exists() and not self.overwrite: return
    df = df_all_radial[df_all_radial.radius_deg <= 5*cls_r200_deg]
    write_table(df, out_path)