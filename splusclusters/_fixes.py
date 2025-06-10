
from prefect import task
from pylegs.io import read_table, write_table

from splusclusters._info import ClusterInfo
from splusclusters.configs import configs


@task(task_run_name='fix-z-range-{info.name}', version='1.0', persist_result=False)
def fix_z_range(info: ClusterInfo):
  out_path = configs.PHOTOZ_FOLDER / f'{info.name}.parquet'
  if out_path.exists():
    df = read_table(out_path)
    if len(df) > 0:
      df = df[df.zml.between(*info.z_photo_range)]
      write_table(df, out_path)
