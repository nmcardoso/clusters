import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import streamlit as st

SPEC_DATA_URL = 'https://github.com/nmcardoso/clusters/raw/main/public/clusters_spec_all.csv'
AUX_DATA_URL = 'https://github.com/nmcardoso/clusters/raw/main/public/catalog_chinese_xray.csv'  

# SPEC_DATA_URL = 'public/clusters_spec_all.csv'
# AUX_DATA_URL = 'public/catalog_chinese_xray.csv'


@st.cache_data
def load_spec_data() -> pd.DataFrame:
  df = pd.read_csv(SPEC_DATA_URL)
  df = df[['RA', 'DEC', 'z', 'zml', 'cluster']]
  return df

@st.cache_data
def load_aux_data() -> pd.DataFrame:
  df = pd.read_csv(AUX_DATA_URL)
  return df


def main():
  st.set_page_config(layout='wide')
  
  if 'spec_df' not in st.session_state:
    st.session_state.spec_df = load_spec_data()
  if 'aux_df' not in st.session_state:
    st.session_state.aux_df = load_aux_data()
  
  spec_df = st.session_state.spec_df
  aux_df = st.session_state.aux_df
  
  
  sel_col1, sel_col2 = st.columns(2)
  with sel_col1:
    sel_cluster_name = st.selectbox('Cluster:', label_visibility='collapsed', options=spec_df[spec_df.cluster.isin(aux_df.name)].cluster.unique())
    sel_cluster_z = aux_df[aux_df['name'] == sel_cluster_name]['z'].values[0]
  
  with sel_col2:
    st.write(f'{sel_cluster_name.upper()} Redshift: {sel_cluster_z}')


  with st.form('z_range'):
    col1, col2 = st.columns(2)
    with col1:
      z_range = st.slider(
        label='Spec Z Range', 
        min_value=0.0, 
        max_value=0.05, 
        value=0.01, 
        format='%.3f', 
        step=0.001, 
        label_visibility='visible'
      )

    with col2:
      photoz_range = st.slider(
        label='Photo Z Range', 
        min_value=0.0, 
        max_value=0.05,
        value=0.015, 
        format='%.3f', 
        step=0.001, 
        label_visibility='visible'
      )

    st.form_submit_button('Update Plots')
    
  

  cluster_df_z = spec_df[
    (spec_df.cluster == sel_cluster_name) &
    (spec_df.z.between(sel_cluster_z - z_range, sel_cluster_z + z_range))
  ]
  cluster_df_photoz = spec_df[
    (spec_df.cluster == sel_cluster_name) & 
    (spec_df.zml.between(sel_cluster_z - photoz_range, sel_cluster_z + photoz_range))
  ]


  plt_col1, plt_col2 = st.columns(2)
  
  with plt_col1:
    st.header('Spec Z')
    
    hist_z = px.histogram(
      data_frame=cluster_df_z,
      x='z',
      title='Spec Z Histogram',
    )
    st.plotly_chart(hist_z, use_container_width=True)
    
    scatter_z = px.scatter(
      data_frame=cluster_df_z, 
      x='RA', 
      y='DEC', 
      color='z', 
      title='Spec Z Distribution', 
      color_continuous_scale='Plasma',
    )
    st.plotly_chart(scatter_z, use_container_width=True)
    
    density_plot_z = ff.create_2d_density(
      x=cluster_df_z.RA.values, 
      y=cluster_df_z.DEC.values, 
      point_size=6,
      # hist_color='#ADD8E6',
      title='Spec Z Density'
    )
    st.plotly_chart(density_plot_z, use_container_width=True)
  
  
  with plt_col2:
    st.header('Photo Z')
    
    hist_photoz = px.histogram(
      data_frame=cluster_df_photoz,
      x='zml',
      title='Photo Z Histogram',
    )
    st.plotly_chart(hist_photoz, use_container_width=True)
    
    scatter_photoz = px.scatter(
      data_frame=cluster_df_photoz, 
      x='RA', 
      y='DEC', 
      color='zml', 
      title='Photo Z Distribution', 
      color_continuous_scale='Plasma'
    )
    st.plotly_chart(scatter_photoz, use_container_width=True)
    
    density_plot_photoz = ff.create_2d_density(
      x=cluster_df_photoz.RA.values, 
      y=cluster_df_photoz.DEC.values, 
      point_size=6,
      # hist_color='#ADD8E6',
      title='Photo Z Density'
    )
    st.plotly_chart(density_plot_photoz, use_container_width=True)
  
  
main()