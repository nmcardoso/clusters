from typing import Sequence, Tuple

import pandas as pd
from astromodule.io import write_table
from astromodule.pipeline import PipelineStage
from astromodule.table import radial_search
from astropy import units as u
from astropy.coordinates import SkyCoord

from splusclusters.constants import *


class WebsitePagesStage(PipelineStage):
  def __init__(self, clusters: Sequence[str]):
    self.clusters = sorted(clusters)
  
  def make_splus_fields_tables(
    self, 
    cls_name: str,
    cls_ra: float,
    cls_dec: float,
    cls_r200_deg: float,
    cls_r500_deg: float,
    df: pd.DataFrame,
  ):
    cls_center = SkyCoord(cls_ra, cls_dec, unit='deg')
    r200_path = WEBSITE_PATH / 'clusters' / cls_name / 'splus_fields_5r200.csv'
    r500_path = WEBSITE_PATH / 'clusters' / cls_name / 'splus_fields_5r500.csv'
    r15Mpc_path = WEBSITE_PATH / 'clusters' / cls_name / 'splus_fields_15Mpc.csv'
    if not r200_path.exists() or not r500_path.exists() or not r15Mpc_path.exists():
    # if True:
      coords = SkyCoord(df.ra, df.dec, unit='deg')
      df_r200 = radial_search(cls_center, df, 5*cls_r200_deg * u.deg, cached_catalog=coords)
      df_r500 = radial_search(cls_center, df, 5*cls_r500_deg * u.deg, cached_catalog=coords)
      fields_r200 = df_r200.groupby('field').size().reset_index(name='n_objects')
      fields_r500 = df_r500.groupby('field').size().reset_index(name='n_objects')
      fields_15Mpc = df.groupby('field').size().reset_index(name='n_objects')
      write_table(fields_r200, r200_path)
      write_table(fields_r500, r500_path)
      write_table(fields_15Mpc, r15Mpc_path)
    
  
  def get_paginator(self, back: bool = True):
    if back:
      links = [f'<a href="../{name}/index.html">{name}</a>' for name in self.clusters]
    else:
      links = [f'<a href="clusters/{name}/index.html">{name}</a>' for name in self.clusters]
      
    return ' &nbsp;&bullet;&nbsp; '.join(links)
  
  def make_index(self):
    page = f'''<!DOCTYPE html>
    <html>
    <head>
      <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌌</text></svg>">
      <title>S-PLUS Clusters Catalog</title>
      <link rel="stylesheet" href="lightbox.min.css" />
    </head>
    <body>
      <h2>Clusters Index</h2>
      {self.get_paginator(back=False)}
      <br /><br /><br />
      <img src="all_sky.png" width="80%" style="display: block; margin: 0 auto;" />
      <script src="lightbox-plus-jquery.min.js"></script>
    </body>
    </html>
    '''
    index_path = WEBSITE_PATH / 'index.html'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(page)
  
  def run(
    self, 
    cls_name: str, 
    cls_ra: float, 
    cls_dec: float, 
    cls_z: float,
    cls_15Mpc_deg: float,
    cls_r200_deg: float,
    cls_r200_Mpc: float,
    cls_r500_deg: float,
    cls_r500_Mpc: float,
    z_spec_range: Tuple[float, float],
    z_photo_range: Tuple[float, float],
    df_photoz_radial: pd.DataFrame,
  ):
    width = 400
    height = 400
    folder_path = WEBSITE_PATH / 'clusters' / cls_name
    attachments = [
      'splus_fields_5r200.csv', 'splus_fields_5r500.csv', 
      'splus_fields_15Mpc.csv'
    ]
    attachments_html = [f'<a href="{a}">{a}</a>' for a in attachments]
    images = [
      'specz', 'photoz', 'photoz_specz', 
      'spec_velocity_position', 'spec_velocity_rel_position', 
      'spec_velocity', 'specz_distance', 'photoz_distance', 
      'mag_diff', 'mag_diff_hist', 'xray',
    ]
    img_paths = []
    for i in images:
      candidates = list(folder_path.glob(f'{i}.*'))
      if len(candidates) > 0:
        img_paths.append(str(candidates[0].name))
        
    gallery = [
      f'<a href="{img}" class="gallery" data-lightbox="images"><img src="{img}" width="{width}" height="{height}" /></a>'
      for img in img_paths
    ]
    page = f'''<!DOCTYPE html>
    <html>
    <head>
      <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌌</text></svg>">
      <title>{cls_name}</title>
      <link rel="stylesheet" href="../../lightbox.min.css" />
      <style type="text/css">
      a.gallery:hover {{
        cursor: -moz-zoom-in; 
        cursor: -webkit-zoom-in;
        cursor: zoom-in;
      }}
      </style>
    </head>
    <body>
      <b>Clusters Index</b><br />
      {self.get_paginator()}
      <hr />
      <h2>Cluster: {cls_name}</h2>
      <i>
        Measures: 
        <b>RA:</b> {cls_ra:.4f}&deg; &nbsp;&nbsp;&nbsp;  
        <b>DEC:</b> {cls_dec:.4f}&deg; &nbsp;&nbsp;&nbsp;  
        <b>z<sub>cluster</sub>:</b> {cls_z:.4f} &nbsp;&nbsp;&nbsp; 
        <b>search radius:</b> 15Mpc ({cls_15Mpc_deg:.3f}&deg;) &nbsp;&nbsp;&nbsp;  
        <b>5&times;R200:</b> {5*cls_r200_Mpc:.3f}Mpc ({5*cls_r200_deg:.3f}&deg;) &nbsp;&nbsp;&nbsp; 
        <b>5&times;R500:</b> {5*cls_r500_Mpc:.3f}Mpc ({5*cls_r500_deg:.3f}&deg;)
      </i>
      <br />
      <i>
        Constraints: 
        <b>z<sub>spec</sub>:</b> z<sub>cluster</sub> &plusmn; 0.007 = [{z_spec_range[0]:.4f}, {z_spec_range[1]:.4f}] &nbsp;&nbsp;&nbsp; 
        <b>z<sub>photo</sub>:</b> z<sub>cluster</sub> &plusmn; 0.015 = [{z_photo_range[0]:.4f}, {z_photo_range[1]:.4f}] &nbsp;&nbsp;&nbsp; 
        <b>mag<sub>r</sub>:</b> [13, 22] &nbsp;&nbsp;&nbsp; <b>class<sub>spec</sub>:</b> GALAXY*
      </i>
      <br />
      <i>
        Cosmology: 
        <b>H<sub>0</sub>:</b> 70 km Mpc<sup>-1</sup> s<sup>-1</sup> &nbsp;&nbsp;&nbsp; 
        <b>&Omega;<sub>m</sub>:</b> 0.3 &nbsp;&nbsp;&nbsp; 
        <b>&Omega;<sub>&Lambda;</sub>:</b> 0.7
      </i>
      <br /><br />
      <b>Attachments:</b> {' &nbsp;&bullet;&nbsp; '.join(attachments_html)}
      <br />
      <p><b>Gallery</b></p>
      {' '.join(gallery)}
      <p><b>Legacy DR10</b></p>
      <div id="aladin-lite-div" style="width: 850px; height: 700px; margin:0 auto;"></div>
      
      <script src="../../lightbox-plus-jquery.min.js"></script>
      <script>
      lightbox.option({{
        'resizeDuration': 0,
        'fadeDuration': 0,
        'imageFadeDuration': 0,
        'wrapAround': true,
        'fitImagesInViewport': true,
      }})
      </script>
      
      <script src='../../aladin.js' charset='utf-8'></script>
      <script>
      const catFilter = (source) => {{
        return !isNaN(parseFloat(source.data['redshift'])) && (source.data['redshift'] > {z_spec_range[0]}) && (source.data['redshift'] < {z_spec_range[1]})
      }}
      
      var aladin;
      window.addEventListener("load", () => {{
        A.init.then(() => {{
          // Init Aladin
          aladin = A.aladin('#aladin-lite-div', {{
            source: 'CDS/P/DESI-Legacy-Surveys/DR10/color',
            target: '{cls_ra:.6f} {cls_dec:.6f}', 
            fov: {5*cls_r200_deg + 0.3},
            cooFrame: 'ICRSd',
          }});
          aladin.setImageSurvey('CDS/P/DESI-Legacy-Surveys/DR10/color');
          
          // Add 5R200 Circle
          var overlay = A.graphicOverlay({{color: '#ee2345', lineWidth: 2}});
          aladin.addOverlay(overlay);
          overlay.add(A.ellipse({cls_ra:.6f}, {cls_dec:.6f}, {2.5*cls_r200_deg}, {2.5*cls_r200_deg}, 0));
          
          // Add redshift catalog
          const cat_url = 'http://cdsxmatch.u-strasbg.fr/QueryCat/QueryCat?catName=SIMBAD&mode=cone&pos={cls_ra:.6f}%20{cls_dec:.6f}&r={5*cls_r200_deg:.3f}deg&format=votable&limit=6000'
          const cat = A.catalogFromURL(cat_url, {{
            name: 'redshift',
            sourceSize:12, 
            color: '#f72525', 
            displayLabel: true, 
            labelColumn: 'redshift', 
            labelColor: '#31c3f7', 
            labelFont: '14px sans-serif', 
            onClick: 'showPopup', 
            shape: 'circle',
            filter: catFilter,
          }})
          aladin.addCatalog(cat)
        }});
      }})
      </script>
    </body>
    </html>
    '''
    index_path = folder_path / 'index.html'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(page)
    
    self.make_splus_fields_tables(
      cls_name=cls_name, 
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      cls_r200_deg=cls_r200_deg, 
      cls_r500_deg=cls_r500_deg, 
      df=df_photoz_radial
    )