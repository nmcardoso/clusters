import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

TABLE_URL = 'https://github.com/nmcardoso/clusters/releases/download/clusters_v3/concat_59_streamlit.parquet'


def rmse(a1, a2):
  return np.linalg.norm(a1 - a2) / np.sqrt(len(a1))


def relative_err(actual, expected):
  return (actual - expected) / expected


def compute_pdf_peak(a, binrange = None, binwidth = None):
  fig, ax = plt.subplots()
  sns.histplot(x=a, binrange=binrange, binwidth=binwidth, kde=True, ax=ax)
  x, y = ax.get_lines()[0].get_data()
  x_max, y_max = x[np.argmax(y)], np.max(y)
  plt.close(fig)
  return x_max, y_max




def hist_plot(z, zml, z_label, zml_label, title, binrange):
  fig, ax = plt.subplots(figsize=(6,4.5))
  sns.histplot(x=z, binrange=binrange, binwidth=0.001, kde=True, label=z_label, ax=ax)
  sns.histplot(x=zml, binrange=binrange, binwidth=0.001, kde=True, label=zml_label, ax=ax)
  ax.legend()
  ax.set_title(title)
  ax.set_ylabel(None)
  st.pyplot(fig, clear_figure=True)
  

@st.cache_data
def load_table():
  return pd.read_parquet(TABLE_URL)


@st.cache_data
def load_members_index_v6():
  cols = [
    'clsid', 'ra', 'dec', 'z_spec', 'sigma_cl_kms', 'sigma_cl_lower', 
    'sigma_cl_upper', 'r200', 'r200_lower', 'r200_upper', 'm200',
    'm200_lower', 'm200_upper', 'nwcls', 'n_memb', 'n_memb_wR200', 'name'
  ]
  path = 'https://github.com/nmcardoso/clusters/raw/refs/heads/main/tables/members_v6/info_cls_shiftgap_iter_10.0hmpcf_nrb.dat'
  names_df = pd.read_csv(path, names=cols, sep=r'\s+', comment='#')
  path = 'https://github.com/nmcardoso/clusters/raw/refs/heads/main/tables/members_v6/info_cls_shiftgap_iter_10.0hmpcf.dat_nrb'
  cols = [
    'clsid', 'ra', 'dec', 'z_spec', 'veli', 'velf', 'Nwcls', 'Nmemb', 'sigma_p',
    'sigma_p_lower', 'sigma_p_upper', 'R500_Mpc', 'R500_lower', 'R500_upper',
    'M500_solar', 'M500_lower', 'M500_upper', 'R200_Mpc', 'R200_lower',
    'R200_upper', 'M200_solar', 'M200_lower', 'M200_upper', 'znew', 'znew_err',
    'Rap', 'Nmemb_wR200', 'col1', 'col2', 'col3', 'col4'
  ]
  info_df = pd.read_csv(path, names=cols, sep=r'\s+', comment='#')
  info_df['name'] = names_df['name'].values
  return info_df


@st.cache_data
def get_clusters_list():
  df = load_table()
  return df['cluster_name'].unique()


def fmt_func(c):
  df_info = load_members_index_v6()
  z_cluster = df_info[df_info['name'] == c].iloc[0].z_spec
  return f'{c} ({z_cluster:.3f})'






def main():
  st.set_page_config(page_title='Odds', layout='wide', page_icon='🌌')
  
  col1, col2 = st.columns([1, 1])
  with col1:
    cluster = st.selectbox('Cluster', get_clusters_list(), format_func=fmt_func)
  with col2:
    odds = st.number_input('Minimum odds', min_value=0.0, max_value=1.0, step=0.1, format='%.2f')
  
  if cluster:
    df_info = load_members_index_v6()
    df_info = df_info[df_info['name'] == cluster].iloc[0]
    z_cluster = df_info['z_spec']
    r200_Mpc = df_info['R200_Mpc']
    
    df = load_table()
    z_delta = 0.02
    mask = (
      (df.cluster_name == cluster) &
      (df.flag_member == 0) &
      (df.z.between(z_cluster-z_delta, z_cluster+z_delta)) &
      (df.zml.between(z_cluster-0.03, z_cluster+0.03)) &
      (df.radius_Mpc < 5*r200_Mpc) &
      (df.odds >= odds)
    )
    df_sample = df[mask]
    
    if len(df_sample) > 0:
      x_min = min(df_sample.zml.min(), df_sample.z.min())
      x_max = max(df_sample.zml.max(), df_sample.z.max())
      binrange = (x_min, x_max)
      
      z = df_sample.z.values
      zml = df_sample.zml.values
      
      spec_peak, _ = compute_pdf_peak(z, binwidth=0.001)
      photo_peak, _ = compute_pdf_peak(zml, binwidth=0.001)
      offset = photo_peak - spec_peak
      err = rmse(z, zml)
      
      col1, col2 = st.columns([1, 1])
      with col1:
        hist_plot(
          z=z, 
          zml=zml, 
          z_label=f'spec-z ({len(z)})', 
          zml_label=f'photo-z ({len(zml)})', 
          title=f'{cluster}. Offset: {offset:.2f}. RMSE: {err:.3f}', 
          binrange=binrange,
        )
      
      err = rmse(z, zml-offset)
      x_min = min(df_sample.zml.min()-offset, df_sample.z.min())
      x_max = max(df_sample.zml.max()-offset, df_sample.z.max())
      binrange = (x_min, x_max)
      with col2:
        hist_plot(
          z=z, 
          zml=zml-offset, 
          z_label=f'spec-z ({len(z)})', 
          zml_label=f'shifted photo-z ({len(zml)})', 
          title=f'{cluster}. Offset: 0.00. RMSE: {err:.3f}', 
          binrange=binrange,
        )
    else:
      st.write('No data to plot')
  
  

if __name__ == "__main__":
  main()
