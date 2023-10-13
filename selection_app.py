import json
import os
import urllib
from copy import deepcopy

import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import streamlit as st

SPEC_DATA_V1_URL = 'public/clusters_v1.csv'
SPEC_DATA_V2_URL = 'public/clusters_v2.csv'
SPEC_DATA_V3_URL = 'https://github.com/nmcardoso/clusters/releases/download/clusters_v3/clusters_v3.parquet' if os.environ.get('ENV', '') == 'streamlit' else 'outputs_v3/clusters_v3.csv'
AUX_DATA_URL = 'public/catalog_chinese_xray.csv'
URL_MAP = {
  'clusters_v1': SPEC_DATA_V1_URL,
  'clusters_v2': SPEC_DATA_V2_URL,
  'clusters_v3': SPEC_DATA_V3_URL,
}
TABLE_DESCRIPTIONS = {
  'clusters_v1': '''
Lista de clusters

- Restrições na seleção dos clusters:
  - Todos os clusters da lista
  
- Restrições na seleção dos objetos de cada cluster:
  - Todos os objetos presentes nos campos do S-PLUS especificados
  '''.strip(),
  
  'clusters_v2': '''
Todos os clusters da tabela chinesa presentes no S-PLUS

- Restrições na seleção dos clusters:
  - Clusters com redshift espectroscópico nominal no intervalo: $z_{{nominal}} < 0.1$
  
- Restrições na seleção dos objetos de cada cluster:
  - Raio de busca (S-PLUS iDR4): 1 grau do centro nominal do cluster
  - Posição (S-PLUS iDR4): objeto com distância máxima de 1 arcsec de algum objeto do S-PLUS
  - PhotoZ (S-PLUS iDR4): $z_{{nominal}} - \\epsilon_2 < z_{{photo}} < z_{{nominal}} + \\epsilon_2$, com $\\epsilon_2 = 0.04$
  - Redshift (Tabela Erick): $z_{{nominal}} - \\epsilon_1 < z_{{spec}} < z_{{nominal}} + \\epsilon_1$, com $\\epsilon_1 = 0.02$
  '''.strip(),
  
  'clusters_v3': '''
Todos os clusters da tabela chinesa presentes no S-PLUS.

- Restrições na seleção dos clusters:
  - Clusters com redshift espectroscópico nominal no intervalo: $0.02 < z_{{nominal}} < 0.1$

- Restrições na seleção dos objetos de cada cluster:
  - Raio de busca (S-PLUS iDR4): 15 Mpc do centro nominal do cluster
  - Posição (S-PLUS iDR4): objeto com distância máxima de 1 arcsec de algum objeto do S-PLUS
  - PhotoZ (S-PLUS iDR4): $z_{{nominal}} - \\epsilon_2 < z_{{photo}} < z_{{nominal}} + \\epsilon_2$, com $\\epsilon_2 = 0.04$
  - Redshift (Tabela Erick): $z_{{nominal}} - \\epsilon_1 < z_{{spec}} < z_{{nominal}} + \\epsilon_1$, com $\\epsilon_1 = 0.02$
  '''.strip(),
}

print(os.environ)


@st.cache_data
def load_spec_data(url: str) -> pd.DataFrame:
  suffix = url.split('.')[-1]
  
  if suffix == 'csv':
    df = pd.read_csv(url)
  elif suffix == 'parquet':
    df = pd.read_parquet(url)
  
  df = df[['RA', 'DEC', 'z', 'zml', 'cluster']]
  return df

@st.cache_data
def load_aux_data() -> pd.DataFrame:
  df = pd.read_csv(AUX_DATA_URL)
  return df

@st.cache_data
def get_xray_image(cluster_name: str) -> np.ndarray:
  try:
    prepared_cluster_name = cluster_name
    if prepared_cluster_name[0] == 'A':
      while prepared_cluster_name[1] == '0':
        prepared_cluster_name = 'A' + prepared_cluster_name[2:]
    
    url = f'http://zmtt.bao.ac.cn/galaxy_clusters/dyXimages/image_all/{prepared_cluster_name}_image.eps'
    img = PIL.Image.open(urllib.request.urlopen(url))
    img.load(transparency=True)
    img = img.rotate(-90)
    return np.array(img)
  except:
    return None


plot_layout = {
  'yaxis': {
    'tickfont': {
      'size': 15,
      'color': 'black'
    },
    'titlefont': {
      'color': 'black',
      'size': 15
    },
  },
  'xaxis': {
    'tickfont': {
      'size': 15,
      'color': 'black'
    },
    'titlefont': {
      'color': 'black',
      'size': 15
    },
  },
  'xaxis2': {
    'tickfont': {
      'size': 15,
      'color': 'black'
    },
  },
  'yaxis2': {
    'tickfont': {
      'size': 15,
      'color': 'black'
    },
  },
  'coloraxis': {
    'colorbar': {
      'tickfont': {
        'size': 15,
        'color': 'black',
      },
      'titlefont': {
        'size': 15,
        'color': 'black',
      },
    },
  },
}

plot_layout_reversed = deepcopy(plot_layout)
plot_layout_reversed['xaxis']['autorange'] = 'reversed'


def main():
  st.set_page_config(page_title='Cluster Analysis', layout='wide', page_icon='🌌')
  
  sel_col1, sel_col2 = st.columns(2)
  with sel_col1:
    option = st.selectbox(label='Table', options=['clusters_v1', 'clusters_v2', 'clusters_v3'])
    selected_table = URL_MAP[option]
    print(selected_table)
    
    if 'spec_df' not in st.session_state or st.session_state.get('selected_table') != option:
      st.session_state.spec_df = load_spec_data(selected_table)
      st.session_state.selected_table = option
    if 'aux_df' not in st.session_state:
      st.session_state.aux_df = load_aux_data()
    
    spec_df = st.session_state.spec_df
    aux_df = st.session_state.aux_df
    
  with sel_col2:
    sel_cluster_name = st.selectbox(
      label='Cluster',
      # label_visibility='collapsed', 
      options=spec_df[spec_df.cluster.isin(aux_df.name)].cluster.unique()
    )
    sel_cluster_z = aux_df[aux_df['name'] == sel_cluster_name]['z'].values[0]


  with st.form('z_range'):
    col1, col2 = st.columns(2)
    with col1:
      z_range = st.slider(
        label='Spec Z Range ($\\epsilon_1$)', 
        min_value=0.0, 
        max_value=0.02, 
        value=0.01, 
        format='%.4f', 
        step=0.0001, 
        label_visibility='visible'
      )
      cluster_df_z = spec_df[
        (spec_df.cluster == sel_cluster_name) &
        (spec_df.z.between(sel_cluster_z - z_range, sel_cluster_z + z_range))
      ]
      st.write(f'Filter: ${(sel_cluster_z - z_range):.4f} < z_{{spec}} < {(sel_cluster_z + z_range):.4f}$. Number of objects: {len(cluster_df_z)}')

    with col2:
      photoz_range = st.slider(
        label='Photo Z Range ($\\epsilon_2$)', 
        min_value=0.0, 
        max_value=0.02,
        value=0.0005 if option == 'clusters_v3' else 0.015, 
        format='%.4f', 
        step=0.0001, 
        label_visibility='visible'
      )
      cluster_df_photoz = spec_df[
        (spec_df.cluster == sel_cluster_name) & 
        (spec_df.zml.between(sel_cluster_z - photoz_range, sel_cluster_z + photoz_range))
      ]
      st.write(f'Filter: ${(sel_cluster_z - photoz_range):.4f} < z_{{photo}} < {(sel_cluster_z + photoz_range):.4f}$. Number of objects: {len(cluster_df_photoz)}')
    
    st.form_submit_button('Update Plots')



  col1, col2 = st.columns(2)    
  with col1:
    st.markdown(f'''
**Table Description:**

{TABLE_DESCRIPTIONS.get(option, "*empty*")}
    '''.strip())
    
  with col2:
    st.write('**Cluster Attributes:**')
    st.write(f'**Redshift:** {sel_cluster_z}')

    cluster_ra = aux_df[aux_df['name'] == sel_cluster_name]['ra'].values[0]
    cluster_dec = aux_df[aux_df['name'] == sel_cluster_name]['dec'].values[0]
    
    if np.isnan(cluster_ra) or np.isnan(cluster_dec):
      cluster_ra = np.median(cluster_df_z.RA.values)
      cluster_dec = np.median(cluster_df_z.DEC.values)
      st.markdown(f'**Nominal RA:** _not found_, using median: {cluster_ra:.4f}')
      st.markdown(f'**Nominal DEC:** _not found_, using median: {cluster_dec:.4f}')
    else:
      st.write(f'**Nominal RA:** {cluster_ra:.4f} (X-ray table)')
      st.write(f'**Nominal DEC:** {cluster_dec:.4f} (X-ray table)')



  plt_col1, plt_col2 = st.columns(2)
  with plt_col1:
    st.header('Spec Z')
    
    hist_z = px.histogram(
      data_frame=cluster_df_z,
      x='z',
      title='Spec Z Histogram',
      nbins=22,
    )
    hist_z.update_layout(plot_layout)
    st.plotly_chart(hist_z, use_container_width=True, config={'staticPlot': True})
    
    
    ra_delta_z = np.maximum(np.abs(cluster_ra - cluster_df_z.RA.values.min()), np.abs(cluster_df_z.RA.values.max() - cluster_ra))
    dec_delta_z = np.maximum(np.abs(cluster_dec - cluster_df_z.DEC.values.min()), np.abs(cluster_df_z.DEC.values.max() - cluster_dec))
    
    density_z = ff.create_2d_density(
      x=cluster_df_z.RA.values, 
      y=cluster_df_z.DEC.values, 
      point_size=6,
      # hist_color='#ADD8E6',
      title='Spec Z Density',
      ncontours=22,
    )
    density_z.add_scatter(
      x=[cluster_ra], 
      y=[cluster_dec], 
      marker_symbol='cross-thin', 
      marker_line_color='red', 
      marker_color='red', 
      marker_line_width=4, 
      marker_size=15
    )
    density_z.update_layout(xaxis={'range': [cluster_ra - ra_delta_z, cluster_ra + ra_delta_z]})
    density_z.update_layout(yaxis={'range': [cluster_dec - dec_delta_z, cluster_dec + dec_delta_z]})
    density_z.update_layout(plot_layout_reversed)
    st.plotly_chart(density_z, use_container_width=True, config={'staticPlot': True})
    
    scatter_z = px.scatter(
      data_frame=cluster_df_z, 
      x='RA', 
      y='DEC', 
      color='z', 
      title='Spec Z Distribution', 
      color_continuous_scale='Plasma',
    )
    scatter_z.update_layout(plot_layout_reversed)
    st.plotly_chart(scatter_z, use_container_width=True, config={'staticPlot': True})
  
  
  
  with plt_col2:
    st.header('Photo Z')
    
    hist_photoz = px.histogram(
      data_frame=cluster_df_photoz,
      x='zml',
      title='Photo Z Histogram',
      nbins=22,
    )
    hist_photoz.update_layout(plot_layout)
    st.plotly_chart(hist_photoz, use_container_width=True, config={'staticPlot': True})
    
    
    ra_delta_photoz = np.maximum(np.abs(cluster_ra - cluster_df_photoz.RA.values.min()), np.abs(cluster_df_photoz.RA.values.max() - cluster_ra))
    dec_delta_photoz = np.maximum(np.abs(cluster_dec - cluster_df_photoz.DEC.values.min()), np.abs(cluster_df_photoz.DEC.values.max() - cluster_dec))
    
    # https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Histogram2dContour.html
    density_photoz = ff.create_2d_density(
      x=cluster_df_photoz.RA.values, 
      y=cluster_df_photoz.DEC.values, 
      point_size=6,
      # hist_color='#ADD8E6',
      title='Photo Z Density',
      ncontours=22,
    )
    density_photoz.add_scatter(
      x=[cluster_ra], 
      y=[cluster_dec], 
      marker_symbol='cross-thin', 
      marker_line_color='red', 
      marker_color='red', 
      marker_line_width=4, 
      marker_size=15
    )
    density_photoz.update_layout(xaxis={'range': [cluster_ra - ra_delta_photoz, cluster_ra + ra_delta_photoz]})
    density_photoz.update_layout(yaxis={'range': [cluster_dec - dec_delta_photoz, cluster_dec + dec_delta_photoz]})
    
    density_photoz.update_layout(plot_layout_reversed)
    st.plotly_chart(density_photoz, use_container_width=True, config={'staticPlot': True})
    
    scatter_photoz = px.scatter(
      data_frame=cluster_df_photoz, 
      x='RA', 
      y='DEC', 
      color='zml', 
      title='Photo Z Distribution', 
      color_continuous_scale='Plasma',
    )
    scatter_photoz.update_layout(plot_layout_reversed)
    st.plotly_chart(scatter_photoz, use_container_width=True, config={'staticPlot': True})
    
  
  
  img_col1, img_col2, img_col3 = st.columns([1, 1, 1])
  with img_col1:
    with st.form('img_form'):
      st.markdown('##### Legacy DR10')
      stamp_ra = st.number_input(label='RA', value=cluster_ra, format='%.6f')
      stamp_dec = st.number_input(label='DEC', value=cluster_dec, format='%.6f')
      stamp_pixscale = st.number_input(
        label='Pixel Scale [arcsec/pixel] (Legacy Only)', 
        min_value=0.1, 
        max_value=30.0, 
        value=1.1, 
        step=0.1
      )
      fov_leg = f"""$
      \\displaystyle
      \\begin{{aligned}}
        FOV_{{legacy}} &= \\left({stamp_pixscale:.2f}\\frac{{arcsec}}{{pixel}}\\right) \\left(500\\ pixel \\right) =\\\\
        &= {(stamp_pixscale * 500):.0f}\\ arcsec = {(stamp_pixscale * 500 / 60):.3f}\\ arcmin = {(stamp_pixscale * 500 / 60 / 60):.3f}^{{\\circ}}
      \\end{{aligned}}
      $"""
      fov_splus = f"""$
      \\displaystyle
      \\begin{{aligned}}
        FOV_{{splus}} &= \\left(0.55\\frac{{arcsec}}{{pixel}}\\right) \\left(1000\\ pixel \\right) =\\\\
        &= 550\\ arcsec = 9.167\\ arcmin = 0.153^{{\\circ}}
      \\end{{aligned}}
      $"""
      st.write(fov_leg)
      st.write(fov_splus)
      st.form_submit_button('Reload Stamp')
      
  with img_col2:
    legacy_url = (
      f'https://www.legacysurvey.org/viewer/jpeg-cutout?layer=ls-dr10&'
      f'ra={stamp_ra}&dec={stamp_dec}&pixscale={stamp_pixscale}&size=500'
    )
    st.image(image=legacy_url, use_column_width=True, caption=f'{sel_cluster_name} Legacy Stamp')
    
  with img_col3:
    xray_img = get_xray_image(sel_cluster_name)
    if xray_img is not None:
      st.image(image=xray_img, use_column_width=True, caption=f'{sel_cluster_name} X-Ray')
    else:
      splus_url=(
        f'https://checker-melted-forsythia.glitch.me/img?'
        f'ra={stamp_ra}&dec={stamp_dec}&size=1000'
      )
      st.image(image=splus_url, use_column_width=True, caption=f'{sel_cluster_name} S-PLUS Stamp')
  
main()