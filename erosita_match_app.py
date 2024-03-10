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
  df_match = df_match.rename(columns={'separation': 'separation (arcmin)'})
  return df_match


def main():
  with st.form('position'):
    ra = st.number_input('RA (decimal)', format='%.6f')
    dec = st.number_input('DEC (decimal)', format='%.6f')
    radius = st.number_input('Search Radius (arcmin)', format='%.3f')
    submit = st.form_submit_button('Submit')
  
  if submit:
    df, coords = load_erosita()
    pos = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
    df_erosita = search_calalog(df, pos, radius, coords)
    st.text('eROSITA Results:')
    st.dataframe(df_erosita)
    
    df, coords = load_heasarc()
    df_heasarc = search_calalog(df, pos, radius, coords)
    st.text('Heasarc Results:')
    st.dataframe(df_heasarc)
  



if __name__ == '__main__':
  main()