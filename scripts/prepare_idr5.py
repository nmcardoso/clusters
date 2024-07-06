import re
from pathlib import Path

from astromodule.io import read_table, write_table
from astromodule.table import concat_tables
from tqdm import tqdm


def prepare_idr5():
  # fields_path = Path('/mnt/hd/natanael/astrodata/idr5_fields/')
  # output_path = Path('/mnt/hd/natanael/astrodata/idr5_photoz_clean.parquet')
  fields_path = Path('Predicted')
  output_path = Path('tables/idr5_v3.parquet')
  cols = ['RA', 'DEC', 'zml', 'odds'] # r_auto
  splus_filter = re.compile(r'^(?:HYDRA|SPLUS|STRIPE|MC).*$')
  tables_paths = [
    p for p in fields_path.glob('*.csv') 
    if splus_filter.match(p.name.upper())
  ]
  for path in (pbar := tqdm(tables_paths)):
    pbar.set_description(path.stem)
    df = read_table(path, columns=cols)
    if 'remove_flag' in df.columns:
      df = df[df['remove_flag'] == False]
      del df['remove_flag']
    df = df.rename(columns={'RA': 'ra', 'DEC': 'dec'})
    df['field'] = path.stem
    write_table(df, str(path.absolute()).replace('.csv', '.parquet'))
  write_table(concat_tables(list(fields_path.glob('*.parquet'))), output_path)
  

if __name__ == '__main__':
  prepare_idr5()