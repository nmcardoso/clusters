
import sys
from pathlib import Path

from sklearn.neighbors import KernelDensity

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astromodule.pipeline import Pipeline, PipelineStorage
from astromodule.table import concat_tables
from pylegs.io import read_table, write_table
from scipy.stats import gaussian_kde
from tqdm import tqdm

from splusclusters.configs import configs
from splusclusters.loaders import (LoadClusterInfoStage, LoadLegacyRadialStage,
                                   LoadPhotozRadialStage, LoadSpeczRadialStage,
                                   load_clusters, load_members_index_v6,
                                   load_photoz, load_spec)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage
from splusclusters.utils import compute_pdf_peak, relative_err, rmse
from splusclusters.website import WebsitePagesStage


def create_zoffset_table(df_clusters: pd.DataFrame, z_delta: float = 0.02, overwrite: bool = False):
  if configs.Z_OFFSET_TABLE_PATH.exists() and not overwrite:
    return
  
  data = []
  skiped_clusters = 0
  
  for i, row in df_clusters.iterrows():
    z_cluster = row['z_spec']
    r200_Mpc = row['R200_Mpc']
    cls_id = row['clsid']
    name = row['name']
    n_memb = row['Nmemb']
    
    path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{name}.parquet'
    df = read_table(path)
    mask = (
      (df.flag_member.isin([0, 1])) & 
      (df.z.between(z_cluster-z_delta, z_cluster+z_delta)) &
      (df.zml.between(z_cluster-0.03, z_cluster+0.03)) &
      (df.radius_Mpc < 5*r200_Mpc)
    )
    df = df[mask]
    df_members = df[df.flag_member == 0]
    
    print(f'[{i+1} / {len(df_clusters)}] Cluster: {name}')
    print(f'Members: {len(df_members)}')
    print(f'Members + Interlopers: {len(df)}')
    
    if len(df_members) < 10: 
      print('>> skiped\n')
      skiped_clusters += 1
      continue
    
    row = dict()
    row['name'] = name
    row['clsid'] = cls_id
    row['z_cluster'] = z_cluster
    row['R200_Mpc'] = r200_Mpc
    
    # members offset
    row['peak_spec_m'], row['peak_spec_m_density'] = compute_pdf_peak(df_members.z.values, binwidth=0.001)
    row['peak_photo_m'], row['peak_photo_m_density'] = compute_pdf_peak(df_members.zml.values, binwidth=0.001)
    row['z_offset_m'] = row['peak_photo_m'] - row['peak_spec_m']
    
    # members + interlopers offset
    row['peak_spec_mi'], row['peak_spec_mi_density'] = compute_pdf_peak(df.z.values, binwidth=0.001)
    row['peak_photo_mi'], row['peak_photo_mi_density'] = compute_pdf_peak(df.zml.values, binwidth=0.001)
    row['z_offset_mi'] = row['peak_photo_mi'] - row['peak_spec_mi']
    
    # baseline offset errors
    row['rmse_base_m'] = rmse(df_members.z.values, df_members.zml.values)
    row['rmse_base_mi'] = rmse(df.z.values, df.zml.values)
    
    # members offset errors
    row['rmse_om_m'] = rmse(df_members.z.values, df_members.zml.values - row['z_offset_m'])
    row['rmse_om_mi'] = rmse(df.z.values, df.zml.values - row['z_offset_m'])
    row['rel_om_m'] = relative_err(row['rmse_om_m'], row['rmse_base_m'])
    row['rel_om_mi'] = relative_err(row['rmse_om_mi'], row['rmse_base_mi'])
    
    # members + intelopers offset errors
    row['rmse_omi_m'] = rmse(df_members.z.values, df_members.zml.values - row['z_offset_mi'])
    row['rmse_omi_mi'] = rmse(df.z.values, df.zml.values - row['z_offset_mi'])
    row['rel_omi_m'] = relative_err(row['rmse_omi_m'], row['rmse_base_m'])
    row['rel_omi_mi'] = relative_err(row['rmse_omi_mi'], row['rmse_base_mi'])
    
    data.append(row)
    print()
  
  print('Skiped clusters:', skiped_clusters)
  
  write_table(pd.DataFrame(data), configs.Z_OFFSET_TABLE_PATH)




def hist_plot(z, zml, z_label, zml_label, title, filename, binrange):
  plt.figure(figsize=(8,6.5))
  sns.histplot(x=z, binrange=binrange, binwidth=0.001, kde=True, label=z_label)
  sns.histplot(x=zml, binrange=binrange, binwidth=0.001, kde=True, label=zml_label)
  plt.legend()
  plt.title(title)
  plt.ylabel(None)
  plt.savefig(filename, pad_inches=0.02, bbox_inches='tight')
  plt.close()



def create_zoffset_plots(df_clusters: pd.DataFrame, z_delta: float = 0.02, overwrite: bool = False):
  for i, row in df_clusters.iterrows():
    z_cluster = row['z_spec']
    r200_Mpc = row['R200_Mpc']
    cls_id = row['clsid']
    name = row['name']
    n_memb = row['Nmemb']
    
    print(f'[{i+1} / {len(df_clusters)}] Cluster: {name}')
    
    zoffset = read_table(configs.Z_OFFSET_TABLE_PATH)
    zoffset = zoffset[zoffset['name'] == name]
    if len(zoffset) == 0: 
      print('>> Skiped\n')
      continue
    zoffset = zoffset.iloc[0]
    
    path = configs.PHOTOZ_SPECZ_LEG_FOLDER / f'{name}.parquet'
    df = read_table(path)
    mask = (
      (df.flag_member.isin([0, 1])) & 
      (df.z.between(z_cluster-z_delta, z_cluster+z_delta)) &
      (df.zml.between(z_cluster-0.03, z_cluster+0.03)) &
      (df.radius_Mpc < 5*r200_Mpc)
    )
    df = df[mask]
    df_members = df[df.flag_member == 0]
    
    x_min = min(df.zml.min(), df.z.min())
    x_max = max(df.zml.max(), df.z.max())
    binrange = (x_min, x_max)
    
    hist_plot(
      z=df_members.z.values, 
      zml=df_members.zml.values, 
      z_label=f'spec-z M ({len(df_members)})', 
      zml_label=f'photo-z M ({len(df_members)})', 
      title=f"{name}: MEM (no correction) $z_{{offset}} = {zoffset['z_offset_m']:.4f}$; $RMSE = {zoffset['rmse_base_m']:.4f}$", 
      filename=configs.WEBSITE_PATH / 'clusters_v6' / name / 'zoffset_baseline_m.jpg', 
      binrange=binrange
    )
    hist_plot(
      z=df.z.values, 
      zml=df.zml.values, 
      z_label=f'spec-z M+I ({len(df)})', 
      zml_label=f'photo-z M+I ({len(df)})', 
      title=f"{name}: M+I (no correction) $z_{{offset}} = {zoffset['z_offset_mi']:.4f}$; $RMSE = {zoffset['rmse_base_mi']:.4f}$", 
      filename=configs.WEBSITE_PATH / 'clusters_v6' / name / 'zoffset_baseline_mi.jpg', 
      binrange=binrange
    )
    hist_plot(
      z=df_members.z.values, 
      zml=df_members.zml.values - zoffset['z_offset_m'], 
      z_label=f'spec-z M ({len(df_members)})', 
      zml_label=f'photo-z M ({len(df_members)})', 
      title=f"{name}: M (M shift) $z_{{offset}} = {0.0:.4f}$; $RMSE = {zoffset['rmse_om_m']:.4f}$", 
      filename=configs.WEBSITE_PATH / 'clusters_v6' / name / 'zoffset_m-shift_m.jpg', 
      binrange=binrange
    )
    hist_plot(
      z=df.z.values, 
      zml=df.zml.values - zoffset['z_offset_mi'], 
      z_label=f'spec-z M+I ({len(df)})', 
      zml_label=f'photo-z M+I ({len(df)})', 
      title=f"{name}: M+I (M+I shift) $z_{{offset}} = {0.0:.4f}$; $RMSE = {zoffset['rmse_om_mi']:.4f}$", 
      filename=configs.WEBSITE_PATH / 'clusters_v6' / name / 'zoffset_mi-shift_mi.jpg', 
      binrange=binrange
    )
    print()
    


def main(args):
  configs.Z_SPEC_DELTA = configs.Z_SPEC_DELTA_PAULO
  if args.delta is not None:
    configs.Z_PHOTO_DELTA = args.delta
  else:
    configs.Z_PHOTO_DELTA = configs.Z_SPEC_DELTA_PAULO
  
  df_clusters = load_members_index_v6()
  
  if args.two:
    df_clusters = df_clusters[df_clusters.name.isin(['MKW4', 'A168'])]
  else:
    allowed_names = read_table(configs.ROOT / 'tables' / 'clusters_59.csv').name
    df_clusters = df_clusters[df_clusters.name.isin(allowed_names)]
    
  df_clusters = df_clusters[(df_clusters.z_spec >= 0.02) & (df_clusters.z_spec <= 0.1)]
  df_clusters = df_clusters.sort_values(by='z_spec', ascending=True).reset_index()
  
  if args.table:
    create_zoffset_table(df_clusters, z_delta=args.delta, overwrite=args.overwrite)
  
  if args.plot:
    create_zoffset_plots(df_clusters, z_delta=args.delta, overwrite=args.overwrite)
  
  WebsitePagesStage(df_clusters=df_clusters, version=6).make_zoffset_page()
  




if __name__ == "__main__":
  parser = ArgumentParser(description="Website")
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--delta', action='store', default=0.02, type=float)
  parser.add_argument('--two', action='store_true')
  parser.add_argument('--table', action='store_true')
  parser.add_argument('--plot', action='store_true')
  args = parser.parse_args()
  main(args)