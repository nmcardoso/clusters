from pathlib import Path

import pandas as pd
from astromodule.io import read_table, write_table
from astromodule.pipeline import Pipeline, PipelineStage
from astromodule.table import crossmatch


class LoadStage(PipelineStage):
  name = 'Load'
  products = ['df_source']
  
  def __init__(self, table_name: str):
    self.table_name = table_name
  
  def run(self):
    df = read_table(Path('astromorphlib') / self.table_name)
    return {'df_source': df}
  
  
  
class InteractingCutStage(PipelineStage):
  name = 'InteractingCutStage'
  products = ['df_interacting']
  
  def run(self, df_source: pd.DataFrame):
    C_max = 4
    n_max = 2
    F_max = 0.5
    df_source['C_lim'] = -9.5 * df_source.A + 4.85
    df_source['n_lim'] = -4.75 * df_source.A + 1.73
    df_source['F_lim'] = -4.75 * df_source.A + 0.73
    df = df_source[
      (df_source.C > df_source.C_lim) & (df_source.C < C_max) &
      (df_source.n > df_source.n_lim) & (df_source.n < n_max) &
      (df_source.F > df_source.F_lim) & (df_source.F < F_max)
    ]
    return 