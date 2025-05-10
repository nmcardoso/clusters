from pathlib import Path
from typing import Any

import requests

from splusclusters.utils import SingletonMeta

__all__ = ['configs']
  
  
class Configs(metaclass=SingletonMeta):
  ROOT = Path(__file__).parent.parent
  Z_OFFSET_TABLE_PATH = ROOT / 'tables' / 'z_offset.csv'
  PHOTOZ_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/idr5_photoz_clean.parquet')
  PHOTOZ2_TABLE_PATH = ROOT / Path('tables/idr5_v3.parquet')
  # SPEC_TABLE_PATH = ROOT / Path('tables/SpecZ_Catalogue_20240124.parquet')
  SPEC_TABLE_PATH = ROOT / Path('tables/specz_compilation_20250327.parquet')
  ERASS_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/liana_erass.csv')
  FULL_ERASS_TABLE_PATH = Path('/mnt/hd/natanael/astrodata/eRASS1_min.parquet')
  ERASS2_TABLE_PATH = ROOT / 'tables/Kluge_Bulbul_joint_selected_clusters_zlt0.2.csv'
  LEGACY_TABLE_PATH = '/home/natanaelmc/repos/ls/ls10_psf.parquet'
  HEASARC_TABLE_PATH = ROOT / 'public/heasarc_all.parquet'
  CATALOG_V6_TABLE_PATH = ROOT / 'tables' / 'catalog_v6.csv'
  CATALOG_V6_HYDRA_TABLE_PATH = ROOT / 'tables' / 'catalog_v6_hydra.csv'
  CATALOG_V6_OLD_TABLE_PATH = ROOT / 'tables' / 'catalog_v6_old+ordered+pos.csv'
  XRAY_TABLE_PATH = ROOT / 'tables' / 'catalog_chinese_xray.tsv'
  MEMBERS_V5_PATH = ROOT / 'tables' / 'members_v5'
  MEMBERS_V5_FOLDER = MEMBERS_V5_PATH / 'clusters'
  MEMBERS_V6_PATH = ROOT / 'tables' / 'members_v6'
  MEMBERS_V6_FOLDER = MEMBERS_V6_PATH / 'hold_cls_files'
  OUT_PATH = ROOT / 'outputs_v6'
  STUDY_FOLDER = OUT_PATH / 'studies'
  WEBSITE_PATH = ROOT / 'docs'
  PLOTS_FOLDER = OUT_PATH / 'plots'
  VELOCITY_PLOTS_FOLDER = OUT_PATH / 'velocity_plots'
  MAGDIFF_PLOTS_FOLDER = OUT_PATH / 'magdiff_plots'
  XRAY_PLOTS_FOLDER = OUT_PATH / 'xray_plots'
  MAGDIFF_OUTLIERS_FOLDER = OUT_PATH / 'magdiff_outliers'
  LEG_PHOTO_FOLDER = OUT_PATH / 'legacy'
  LEG_BRICKS_FOLDER = OUT_PATH / 'legacy_bricks'
  PHOTOZ_FOLDER = OUT_PATH / 'photoz'
  SPECZ_FOLDER = OUT_PATH / 'specz'
  PHOTOZ_SPECZ_LEG_FOLDER = OUT_PATH / 'photoz+specz+legacy'
  MAG_COMP_FOLDER = OUT_PATH / 'mag_comp'
  SUBMIT_FOLDER = OUT_PATH / 'submit'
  Z_PHOTO_DELTA = 0.015
  Z_SPEC_DELTA = 0.007
  Z_SPEC_DELTA_PAULO = 0.02
  MAG_RANGE = (13, 22)
  
  def __init__(self):
    self.setup_paths()
  
  def __getattr__(self, name):
    return self.__class__.__dict__.get(name)

  def __setattr__(self, name: str, value: Any) -> None:
    if name in self.__class__.__dict__.keys():
      setattr(self.__class__, name, value)
    else:
      raise ValueError(f'{name} is not a valid config key')
    
  def __repr__(self) -> str:
    r = 'Global Settings:\n'
    for k, v in self.__class__.__dict__.items():
      if not k.startswith('_'):
        r += f'{k}: {v}\n'
    return r
  
  def setup_paths(self):
    for k, v in self.__class__.__dict__.items():
      if k.endswith('FOLDER'):
        v.mkdir(exist_ok=True, parents=True)
  
  
configs = Configs()