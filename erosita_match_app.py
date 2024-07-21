import pandas as pd
import streamlit as st
from astromodule.io import read_table
from astromodule.table import guess_coords_columns, radial_search
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky


@st.cache_data
def load_erosita():
  df = read_table('public/eRASS1_min.parquet')
  coords = SkyCoord(ra=df.ra.values, dec=df.dec.values, unit='deg', frame='icrs')
  return df, coords

def load_heasarc():
  df = read_table('public/heasarc_all.parquet')
  coords = SkyCoord(ra=df.ra.values, dec=df.dec.values, unit='deg', frame='icrs')
  return df, coords

def load_chandra():
  df = read_table('public/catalog_chinese_xray.csv')
  coords = SkyCoord(ra=df.ra.values, dec=df.dec.values, unit='deg', frame='icrs')
  return df, coords

def load_clusters_v5():
  df = read_table('public/clusters_v5.dat')
  if 'clsid' in df.columns:
    del df['clsid']
  coords = SkyCoord(ra=df.RA.values, dec=df.DEC.values, unit='deg', frame='icrs')
  return df, coords


def search_calalog(df: pd.DataFrame, position: SkyCoord, radius: float, coords: SkyCoord):
  df_match = radial_search(
    position, 
    df, 
    radius=radius*u.arcmin, 
    cached_catalog=coords, 
    include_sep=True
  )
  df_match['separation'] = (df_match['separation'].values * u.deg).to(u.arcmin).value
  df_match = df_match.sort_values('separation')
  df_match = df_match.rename(columns={'separation': 'sep. (arcmin)'})
  return df_match


def main():
  st.set_page_config(layout='wide', page_title='Clusters Radial Search', page_icon='🔭')
  tab1, tab2 = st.tabs(['Single Object', 'List of Objects'])
  with tab1:
    with st.form('position'):
      col1, col2, col3, col4 = st.columns(4, gap='medium')
      with col1:
        ra = st.number_input('Cluster Center - RA (decimal)', format='%.6f')
      with col2:
        dec = st.number_input('Cluster Center - DEC (decimal)', format='%.6f')
      with col3:
        radius = st.number_input('Search Radius (arcmin)', format='%.3f')
      with col4:
        st.text('')
        st.text('')
        submit = st.form_submit_button('Submit', use_container_width=True)
    
    if submit:
      pos = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
      
      df, coords = load_erosita()
      df_erosita = search_calalog(df, pos, radius, coords)
      
      df, coords = load_heasarc()
      df_heasarc = search_calalog(df, pos, radius, coords)
      
      df, coords = load_chandra()
      df_chandra = search_calalog(df, pos, radius, coords)
      
      df, coords = load_clusters_v5()
      df_clusters_v5 = search_calalog(df, pos, radius, coords)
      
      col1, col2, col3, col4 = st.columns(4)
      with col1:
        st.markdown('##### eROSITA Results:')
        st.dataframe(df_erosita, hide_index=True, use_container_width=True)
      
      with col2:
        st.markdown('##### Heasarc Results:')
        st.dataframe(df_heasarc, hide_index=True, use_container_width=True)
        
      with col3:
        st.markdown('##### Chandra Results:')
        st.dataframe(df_chandra, hide_index=True, use_container_width=True)
        
      with col4:
        st.markdown('##### Clusters_v5 Results:')
        st.dataframe(df_clusters_v5, hide_index=True, use_container_width=True)
  with tab2:
    with st.form('upload'):
      uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
      submit_upload = st.form_submit_button('Submit', use_container_width=True)
    
    if submit_upload:
      df_upload = read_table(uploaded_file, fmt='csv')
      ra_col, dec_col = guess_coords_columns(df_upload)
      coords_upload = SkyCoord(ra=df_upload[ra_col].values, dec=df_upload[dec_col].values, unit='deg')
      
      df, coords = load_erosita()
      _, sep, _ = match_coordinates_sky(coords_upload, coords)
      df_upload['erosita_sep'] = sep.to(u.arcmin).value
      
      df, coords = load_heasarc()
      _, sep, _ = match_coordinates_sky(coords_upload, coords)
      df_upload['heasarc_sep'] = sep.to(u.arcmin).value
      
      df, coords = load_chandra()
      _, sep, _ = match_coordinates_sky(coords_upload, coords)
      df_upload['chandra_sep'] = sep.to(u.arcmin).value
      
      df, coords = load_clusters_v5()
      _, sep, _ = match_coordinates_sky(coords_upload, coords)
      df_upload['v5_sep'] = sep.to(u.arcmin).value
      
      st.markdown('##### RESULT:')
      st.write('Separation in arcmin')
      st.dataframe(df_upload, hide_index=True, use_container_width=True)



if __name__ == '__main__':
  main()