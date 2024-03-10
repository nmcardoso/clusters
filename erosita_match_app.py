import pandas as pd
import streamlit as st
from astromodule.io import read_table
from astromodule.table import radial_search
from astropy import units as u
from astropy.coordinates import SkyCoord


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
  with st.form('position'):
    col1, col2, col3 = st.columns(3)
    with col1:
      ra = st.number_input('RA (decimal)', format='%.6f')
    with col2:
      dec = st.number_input('DEC (decimal)', format='%.6f')
    with col3:
      radius = st.number_input('Search Radius (arcmin)', format='%.3f')
    submit = st.form_submit_button('Submit')
  
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
  



if __name__ == '__main__':
  main()