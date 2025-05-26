import __init__
from astromodule.table import concat_tables, crossmatch, fast_crossmatch
from astropy.table import Table
from pylegs.io import *
from tqdm import tqdm

from splusclusters.configs import configs
from splusclusters.loaders import load_catalog_v6_old, load_clusters, load_spec


def main():
  df_clusters = load_catalog_v6_old()
  df_spec = load_spec(coords=False)
  
  configs.Z_SPEC_DELTA = 0.02
  configs.SUBMIT_FOLDER = configs.OUT_PATH / 'submit' / 'antigos'
  configs.SUBMIT_FOLDER.mkdir(exist_ok=True, parents=True)
  
  df = None
  
  m1 = crossmatch(
    table1=configs.OUT_PATH / 'lost.parquet',
    table2=df_spec,
    ra1='ra',
    dec1='dec',
    ra2='RA',
    dec2='DEC',
    join='1not2',
    find='best',
  )
  
  for _, cluster in tqdm(df_clusters.iterrows(), total=len(df_clusters)):
    cls_name = cluster['name']
    df_class = load_clusters()
    cls_id = df_class[df_class.name == cls_name].clsid
    ret_path = configs.MEMBERS_V5_FOLDER / f'cluster.gals.sel.shiftgap.iter.{str(cls_id).zfill(5)}'

    if ret_path.exists():
      col_names = [
        'ra', 'dec', 'z', 'z_err', 'v', 'v_err', 'radius_deg', 
        'radius_Mpc', 'v_offset', 'flag_member'
      ] # 0 - member; 1 - interloper
      df_ret = read_table(ret_path, fmt='dat', col_names=col_names)
      df_cluster = read_table(configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{cls_name}.parquet')
      
      m2 = crossmatch(
        table1=m1,
        table2=df_cluster,
        ra1='ra',
        dec1='dec',
        ra2='ra',
        dec2='dec',
      )
      m2 = m2.rename(columns={'ra_1': 'ra', 'dec_1': 'dec'})
      del m2['ra_2']
      del m2['dec_2']
      
      if df is None:
        df = m2
      else:
        df = concat_tables([df, m2])
  
  write_table(df, configs.OUT_PATH / 'recuperadas.csv')
      

if __name__ == '__main__':
  main()