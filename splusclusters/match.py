from pathlib import Path
from typing import Literal, Sequence, Tuple

import pandas as pd
from astromodule.io import merge_pdf, read_table, write_table
from astromodule.pipeline import Pipeline, PipelineStage, PipelineStorage
from astromodule.table import (concat_tables, crossmatch, fast_crossmatch,
                               guess_coords_columns, radial_search, selfmatch)
from astropy import units as u
from astropy.coordinates import SkyCoord

from splusclusters.configs import configs
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
    df_legacy_radial: pd.DataFrame
  ):
    out_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}.parquet'
    if out_path.exists() and not self.overwrite:
      return
    
    df_specz_radial['f_z'] = df_specz_radial['f_z'].astype('str')
    df_specz_radial['original_class_spec'] = df_specz_radial['original_class_spec'].astype('str')
    
    print('Photo-z objects:', len(df_photoz_radial))
    print('Spec-z objects:', len(df_specz_radial))
    print('Legacy objects:', len(df_legacy_radial))
    print('Starting first crossmatch: photo-z UNION spec-z')
    
    t = Timming()
    if len(df_photoz_radial) > 0 and len(df_specz_radial) > 0:
      df = crossmatch(
        table1=df_photoz_radial,
        table2=df_specz_radial,
        join='1or2',
      )
      if 'RA_1' in df.columns: df = df.rename(columns={'RA_1': 'ra_1'})
      if 'DEC_1' in df.columns: df = df.rename(columns={'DEC_1': 'dec_1'})
      df['ra_1'] = df['ra_1'].fillna(df['RA_2'])
      df['dec_1'] = df['dec_1'].fillna(df['DEC_2'])
    elif len(df_photoz_radial) == 0 and len(df_specz_radial) > 0:
      df = df_specz_radial
      if 'RA_1' in df.columns: df = df.rename(columns={'RA_1': 'ra_1'})
      if 'DEC_1' in df.columns: df = df.rename(columns={'DEC_1': 'dec_1'})
    elif len(df_photoz_radial) > 0 and len(df_specz_radial) == 0:
      df = df_photoz_radial
      if 'RA_1' in df.columns: df = df.rename(columns={'RA_1': 'ra_1'})
      if 'DEC_1' in df.columns: df = df.rename(columns={'DEC_1': 'dec_1'})
    elif len(df_photoz_radial) == 0 and len(df_specz_radial) == 0:
      return
    
    print(f'First crossmatch finished. Duration: {t.end()}')
    print('Objects with photo-z only:', len(df[~df.zml.isna() & df.z.isna()]))
    print('Objects with spec-z only:', len(df[df.zml.isna() & ~df.z.isna()]))
    print('Objects with photo-z and spec-z:', len(df[~df.zml.isna() & ~df.z.isna()]))
    print('Total of objects after first match:', len(df))
    print('starting second crossmatch: match-1 LEFT OUTER JOIN legacy')
    
    t = Timming()
    if len(df_legacy_radial) > 0:
      df = crossmatch(
        table1=df,
        table2=df_legacy_radial,
        join='all1',
        ra1='ra_1',
        dec1='dec_1',
      )
    print(df)
    
    print(f'Second crossmatch finished. Duration: {t.end()}')
    print('Objects with legacy:', len(df[~df.type.isna()]))
    print('Objects without legacy:', len(df[df.type.isna()]))
    print('Galaxies:', len(df[df.type != 'PSF']), ', Stars:', len(df[df.type == 'PSF']))
    print('Total of objects after second match:', len(df))
    
    del df['ra'] # legacy ra
    del df['dec'] # legacy dec
    df = df.rename(columns={'ra_1': 'ra', 'dec_1': 'dec'}) # use photoz ra/dec
    
    photoz_cols = ['ra', 'dec', 'zml', 'odds']
    if 'r_auto' in df.columns:
      photoz_cols.append('r_auto')
    if 'field' in df.columns:
      photoz_cols.append('field')
    specz_cols = ['z', 'e_z', 'f_z', 'class_spec']
    legacy_cols = ['mag_r', 'type']
    cols = photoz_cols + specz_cols + legacy_cols
    df = df[cols]
    
    write_table(df, out_path)