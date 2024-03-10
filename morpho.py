from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astromodule.datamodel import DESI
from astromodule.io import read_table, write_table
from astromodule.pipeline import Pipeline, PipelineStage
from astromodule.table import crossmatch, fast_crossmatch

MEMBERS_FOLDER = Path('hold_data_to_andre') / 'hold_cls_files'
MATCH_FOLDER = Path('outputs_morpho')
GZ_DESI_PATH = Path('astromorphlib') / 'gz_desi_deep_learning_catalog_friendly.parquet'
CLUSTER_INDEX_PATH = Path('outputs_v5') / 'paulo' / 'all' / 'index.dat'


class ClusterMembersLoadStage(PipelineStage):
  name = 'ClusterMembersLoad'
  products = ['members_df', 'cluster_index', 'cluster_name']
  def __init__(self, members_path: str | Path = None, index_df: pd.DataFrame = None):
    self.members_path = members_path
    self.index_df = index_df

  def run(self, members_path: str | Path = None):
    members_path = members_path or self.members_path
    cols = [
      'ra', 'dec', 'zspec', 'zspec-err', 'velocity', 'velocity-err', 
      'radius_deg', 'radius_mpc', 'velocity_offset', 'flagm'
    ]
    df = read_table(members_path, fmt='dat', header=None, col_names=cols)
    df = df[df.flagm == 0]
    
    cluster_index = int(members_path.suffix[1:])
    cluster_name = None
    if self.index_df:
      cluster_name = self.index_df.iloc[cluster_index]['name']
      
    return {
      'members_df': df,
      'cluster_index': cluster_index,
      'cluster_name': cluster_name,
    }
    

class GZDesiMatchStage(PipelineStage):
  name = 'GZDesiMatch'
  products = ['match_df']
  def __init__(
    self, 
    desi_df: pd.DataFrame, 
    fast: bool = True, 
    save: bool = True,
    load_if_exists: bool = False,
  ):
    self.desi_df = desi_df
    self.fast = fast
    self.save = save
    self.load_if_exists = load_if_exists
  
  def run(self, members_df: pd.DataFrame, cluster_index: int):
    table_name = f'cluster_morpho_{str(cluster_index).zfill(4)}.csv'
    table_path = MATCH_FOLDER / table_name
    
    if self.load_if_exists and table_path.exists():
      match_df = read_table(table_path)
    else:
      if self.fast:
        match_df = fast_crossmatch(
          members_df, 
          self.desi_df,
        )
      else:
        match_df = crossmatch(
          members_df,
          self.desi_df
        )
    
      final_columns = [
        'ra',
        'dec',
        'zspec',
        DESI.MERGING.MERGER.fr,
        DESI.MERGING.MAJOR_DISTURBANCE.fr,
        DESI.MERGING.MINOR_DISTURBANCE.fr,
        DESI.MERGING.NONE.fr,
        DESI.SPIRAL_ARM_COUNT.ONE.fr,
        DESI.SPIRAL_WINDING.LOOSE.fr,
      ]
      match_df = match_df[final_columns]
      
      if self.save:
        write_table(match_df, table_path)
      
    return {
      'match_df': match_df
    }
    

class MatchFilterStage(PipelineStage):
  name = 'MatchFilter'
  products = ['match_filter_df']
  def __init__(self, save: bool = True):
    self.save = save
    
  def run(self, match_df: pd.DataFrame, cluster_index: int):
    mask = (
      (match_df[DESI.MERGING.NONE.fr] < 0.75) |
      (match_df[DESI.SPIRAL_ARM_COUNT.ONE.fr] > 0.2) |
      (match_df[DESI.SPIRAL_WINDING.LOOSE.fr] > 0.2)
    )
    match_filter_df = match_df[mask]
    
    if self.save:
      table_name = f'cluster_morpho+filter_{str(cluster_index).zfill(4)}.csv'
      write_table(match_filter_df, MATCH_FOLDER / table_name)
    
    return {
      'match_filter_df': match_filter_df
    }


class AnalysisStage(PipelineStage):
  name = 'Analysis'
  
  def run(self):
    cluster_ids = [int(path.suffix[1:]) for path in MEMBERS_FOLDER.glob('*')]
    len_members_dist = []
    len_match_dist = []
    len_filter_dist = []
    match_percent_dist = []
    filter_percent_dist = []
    for i, clsid in enumerate(cluster_ids):
      member_path = MEMBERS_FOLDER / f'cluster.gals.sel.shiftgap.iter.{str(clsid).zfill(5)}'
      match_path = MATCH_FOLDER / f'cluster_morpho_{str(clsid).zfill(4)}.csv'
      match_filter_path = MATCH_FOLDER / f'cluster_morpho+filter_{str(clsid).zfill(4)}.csv'
      member_df = read_table(member_path, fmt='dat', header=None)
      match_df = read_table(match_path)
      index_df = read_table(CLUSTER_INDEX_PATH)
      match_filter_df = read_table(match_filter_path)
      cluster_index = int(member_path.suffix[1:])
      cluster_name = index_df[index_df.clsid == cluster_index]['name'].values[0]
      len_members = len(member_df)
      len_match = len(match_df)
      len_filter = len(match_filter_df)
      if len_members > 0:
        len_match_percent = (len_match / len_members) * 100
        len_filter_percent = (len_filter / len_members) * 100
      else:
        len_match_percent = 0
        len_filter_percent = 0
      len_members_dist.append(len_members)
      len_match_dist.append(len_match)
      len_filter_dist.append(len_filter)
      match_percent_dist.append(len_match_percent)
      filter_percent_dist.append(len_filter_percent)
      print(f'[{i+1} / {len(cluster_ids)}] Cluster {cluster_name}')
      print(f'    - # of members: {len_members}')
      print(f'    - # of members with GZ classifications: {len_match} ({len_match_percent:.2f} %)')
      print(f'    - # of classifications after filter: {len_filter} ({len_filter_percent:.2f} %)')
    
    plt.figure()
    # plt.hist(match_percent_dist, histtype='step', bins=20, label='All Classifications', linewidth=2)
    # plt.hist(filter_percent_dist, histtype='step', bins=8, label='Filtered', linewidth=2)
    bins = np.linspace(np.min(len_filter_dist), np.max(len_members_dist), 35)
    plt.hist(len_members_dist, histtype='step', bins=bins, label='Members', linewidth=2)
    plt.hist(len_match_dist, histtype='step', bins=bins, label='Members + Classif.', linewidth=2)
    plt.hist(len_filter_dist, histtype='step', bins=bins, label='Members + Classif. + Filter', linewidth=2)
    plt.ylabel('Number of clusters')
    plt.xlabel('Number of objects')
    plt.legend()
    plt.savefig(MATCH_FOLDER / 'hist.png', bbox_inches='tight', pad_inches=0.05)
    plt.show()
    
  
def main():
  tables = list(MEMBERS_FOLDER.glob('*'))
  desi_df = read_table(GZ_DESI_PATH)
  p = Pipeline(
    ClusterMembersLoadStage(members_path=MEMBERS_FOLDER / 'cluster.gals.sel.shiftgap.iter.00002'),
    GZDesiMatchStage(desi_df=desi_df, fast=True, load_if_exists=True),
    MatchFilterStage(save=True),
  )
  p.map_run(key='members_path', array=tables, workers=3)
  # p.run(validate=False)
  
  analysis_pipe = Pipeline(
    AnalysisStage()
  )
  analysis_pipe.run()
  
  
if __name__ == '__main__':
  main()