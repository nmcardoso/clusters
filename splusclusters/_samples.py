import pandas as pd
from prefect import task
from pylegs.io import write_table

from splusclusters._info import ClusterInfo
from splusclusters.configs import configs
from splusclusters.utils import cond_overwrite


@task(task_run_name='filter-r200-{info.name}', version='1.0', persist_result=False)
def filter_r200(
  info: ClusterInfo,
  df_all_radial: pd.DataFrame,
  overwrite: bool = False,
):
  out_path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{info.name}+5r200.parquet'
  with cond_overwrite(out_path, overwrite, mkdir=True) as cm:
    df = df_all_radial[df_all_radial.radius_deg <= 5 * info.r200_deg]
    cm.write_table(df)