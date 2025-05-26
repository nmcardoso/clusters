import __init__
from astromodule.table import concat_tables, crossmatch, fast_crossmatch
from astropy.table import Table
from pylegs.io import *
from tqdm import tqdm

from splusclusters.configs import configs
from splusclusters.loaders import (load_catalog_v6, load_catalog_v6_hydra,
                                   load_catalog_v6_old, load_clusters,
                                   load_spec)


def m(df_clusters, df, m1):
  for _, cluster in tqdm(df_clusters.iterrows(), total=len(df_clusters)):
    cls_name = cluster['name']
    df_cluster = read_table(configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}.parquet')
    
    m2 = crossmatch(
      table1=m1,
      table2=df_cluster,
      ra1='ra',
      dec1='dec',
      ra2='ra',
      dec2='dec',
    )
    if m2 is not None:
      m2 = m2.rename(columns={'ra_1': 'ra', 'dec_1': 'dec'})
      del m2['ra_2']
      del m2['dec_2']
      m2['cluster'] = cls_name
      
      if df is None:
        df = m2
      else:
        df = concat_tables([df, m2])
  return df



def main():
  df_spec = load_spec(coords=False)
  df_lost = read_table(configs.OUT_PATH / 'lost.parquet')
  
  configs.Z_SPEC_DELTA = 0.02
  configs.SUBMIT_FOLDER = configs.OUT_PATH / 'submit' / 'antigos'
  configs.SUBMIT_FOLDER.mkdir(exist_ok=True, parents=True)
  
  df = None
  
  m1 = crossmatch(
    table1=df_lost,
    table2=df_spec,
    ra1='ra',
    dec1='dec',
    ra2='RA',
    dec2='DEC',
    join='1not2',
    find='best',
  )
  
  df = m(load_catalog_v6_old(), df, m1)
  df = m(load_catalog_v6_hydra(), df, m1)
  df = m(load_catalog_v6(), df, m1)
  
  write_table(df, configs.OUT_PATH / 'recuperadas.csv')
      

if __name__ == '__main__':
  main()