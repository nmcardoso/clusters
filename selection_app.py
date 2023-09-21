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
  if 'spec_df' not in st.session_state:
    st.session_state.spec_df = load_spec_data()
  if 'aux_df' not in st.session_state:
    st.session_state.aux_df = load_aux_data()
  
  spec_df = st.session_state.spec_df
  aux_df = st.session_state.aux_df
  
  
  sel_cluster_name = st.selectbox('Cluster:', options=spec_df[spec_df.cluster.isin(aux_df.name)].cluster.unique())
  sel_cluster_z = aux_df[aux_df['name'] == sel_cluster_name]['z'].values[0]
  
  
  with st.form('z_range'):
    photoz_range = st.slider('Photo Z Range', 0.0, 0.04, 0.015, format='%.3f', step=0.001, label_visibility='visible')
    z_range = st.slider('Spec Z Range', 0.0, 0.04, 0.015, format='%.3f', step=0.001, label_visibility='visible')
    
    st.form_submit_button('Update Plots')
    
  

  cluster_df_photoz = spec_df[
    (spec_df.cluster == sel_cluster_name) & 
    (spec_df.zml.between(sel_cluster_z - photoz_range, sel_cluster_z + photoz_range))
  ]
  cluster_df_z = spec_df[
    (spec_df.cluster == sel_cluster_name) &
    (spec_df.z.between(sel_cluster_z - z_range, sel_cluster_z + z_range))
  ]


  scatter_z = px.scatter(
    data_frame=cluster_df_z, 
    x='RA', 
    y='DEC', 
    color='z', 
    title='Spec Z', 
    color_continuous_scale='Plasma'
  )
  st.plotly_chart(scatter_z, use_container_width=True)
  
  density_plot_z = ff.create_2d_density(
    x=cluster_df_z.RA.values, 
    y=cluster_df_z.DEC.values, 
    point_size=6,
    hist_color='#ADD8E6',
    title='Spec Z Density'
  )
  st.plotly_chart(density_plot_z, use_container_width=True)
  
  
  scatter_photoz = px.scatter(
    data_frame=cluster_df_photoz, 
    x='RA', 
    y='DEC', 
    color='zml', 
    title='Photo Z', 
    color_continuous_scale='Plasma'
  )
  st.plotly_chart(scatter_photoz, use_container_width=True)
  
  density_plot_photoz = ff.create_2d_density(
    x=cluster_df_photoz.RA.values, 
    y=cluster_df_photoz.DEC.values, 
    point_size=6,
    hist_color='#ADD8E6',
    title='Photo Z Density'
  )
  st.plotly_chart(density_plot_photoz, use_container_width=True)
  
  
main()