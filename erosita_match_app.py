import streamlit as st
from astromodule.io import read_table
from astromodule.table import radial_search
from astropy import units as u
from astropy.coordinates import SkyCoord


@st.cache_data
def load_table():
  df = read_table('public/eRASS1_min.parquet')
  coords = SkyCoord(ra=df.ra.values, dec=df.dec.values, unit='deg', frame='icrs')
  return df, coords


def main():
  with st.form('position'):
    ra = st.number_input('RA (decimal)', format='%.6f')
    dec = st.number_input('DEC (decimal)', format='%.6f')
    radius = st.number_input('Search Radius (arcmin)', format='%.3f')
    submit = st.form_submit_button('Submit')
  
  if submit:
    df, coords = load_table()
    pos = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
    df_match = radial_search(
      pos, 
      df, 
      radius=radius*u.arcmin, 
      cached_catalog=coords, 
      include_sep=True
    )
    df_match['separation'] = (df_match['separation'].values * u.deg).to(u.arcmin).value
    df_match = df_match.sort_values('separation')
    df_match = df_match.rename(columns={'separation': 'separation (arcmin)'})
    
    st.dataframe(df_match)
  



if __name__ == '__main__':
  main()