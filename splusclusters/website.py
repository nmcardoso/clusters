from typing import Sequence, Tuple

import pandas as pd
from astromodule.io import write_table
from astromodule.pipeline import PipelineStage
from astromodule.table import crossmatch, radial_search
from astropy import units as u
from astropy.coordinates import SkyCoord

from splusclusters.configs import configs
from splusclusters.loaders import (load_clusters, load_members_v5,
                                   load_members_v6)


class WebsitePagesStage(PipelineStage):
  def __init__(self, df_clusters: pd.DataFrame, version: int = 5):
    self.df_clusters = df_clusters
    self.clusters = sorted(df_clusters.name.values)
    self.version = version
  
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
    base_path = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name
    r200_path = base_path / 'splus_fields_5r200.csv'
    r500_path = base_path / 'splus_fields_5r500.csv'
    search_radius_path = base_path / 'splus_fields_search_radius.csv'
    if not r200_path.exists() or not r500_path.exists() or not search_radius_path.exists():
    # if True:
      coords = SkyCoord(df.ra, df.dec, unit='deg')
      df_r200 = radial_search(cls_center, df, 5*cls_r200_deg * u.deg, cached_catalog=coords)
      df_r500 = radial_search(cls_center, df, 5*cls_r500_deg * u.deg, cached_catalog=coords)
      fields_r200 = df_r200.groupby('field').size().reset_index(name='n_objects')
      fields_r500 = df_r500.groupby('field').size().reset_index(name='n_objects')
      fields_search_radius = df.groupby('field').size().reset_index(name='n_objects')
      write_table(fields_r200, r200_path)
      write_table(fields_r500, r500_path)
      write_table(fields_search_radius, search_radius_path)
    
  def get_head(self):
    return '''<head>
      <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌌</text></svg>">
      <title>S-PLUS Clusters Catalog</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
      <link rel="stylesheet" href="/lightbox.min.css" />
      <style type="text/css">
      a.gallery:hover {
        cursor: -moz-zoom-in; 
        cursor: -webkit-zoom-in;
        cursor: zoom-in;
      }
      </style>
    </head>'''
  
  def get_scripts(self):
    return '''
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
      
      <script src="/lightbox-plus-jquery.min.js"></script>
      <script>
      lightbox.option({
        'resizeDuration': 0,
        'fadeDuration': 0,
        'imageFadeDuration': 0,
        'wrapAround': true,
        'fitImagesInViewport': true,
      })
      </script>
      
      <script src='/aladin.js' charset='utf-8'></script>
    '''
  
  def get_nav(self, curr_name: str = None):
    links = ''
    for name in self.clusters:
      active = 'active' if name == curr_name else ''
      links += f'<li class="nav-item"><a class="nav-link {active}" href="/clusters_v{self.version}/{name}">{name}</a></li>'
    html = f'''
      <nav class="navbar navbar-expand-lg bg-body-tertiary">
        <div class="container-fluid">
          <a class="navbar-brand" href="#">🌌 S-PLUS Clusters</a>
          <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
          </button>
          <div class="collapse navbar-collapse" id="navbarSupportedContent">
            <ul class="navbar-nav me-auto mb-2 mb-lg-0">
              <li class="nav-item dropdown">
                <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown" aria-expanded="false">
                  Version {self.version}
                </a>
                <ul class="dropdown-menu">
                  <li><a class="dropdown-item" href="/clusters_v6">Version 6</a></li>
                  <li><a class="dropdown-item" href="/clusters_v5">Version 5</a></li>
                </ul>
              </li>
          </div>
        </div>
      </nav>
      <ul class="nav nav-pills">
        {links}
      </ul>
    '''
    return html
  
  def make_index(self):
    page = f'''<!DOCTYPE html>
    <html>
    {self.get_head()}
    <body>
      {self.get_nav()}
      <br /><br /><br />
      <img src="/all_sky.png" width="80%" style="display: block; margin: 0 auto;" />
      {self.get_scripts()}
    </body>
    </html>
    '''
    index_path = configs.WEBSITE_PATH / f'clusters_v{self.version}' / 'index.html'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(page)
  
  def make_landing(self):
    page = f'<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=clusters_v6"></head></html>'
    index_path = configs.WEBSITE_PATH / 'index.html'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(page)
  
  def get_nearest(self, ra: float, dec: float):
    center = SkyCoord(ra=ra, dec=dec, unit='deg')
    others = SkyCoord(self.df_clusters['ra'], self.df_clusters['dec'], unit='deg')
    sep = center.separation(others)
    nearest_idx = sep.argsort()
    html = ''
    for i in nearest_idx[1:8]:
      name = self.df_clusters.name[i]
      separation = sep[i].to('arcmin')
      html += f' <a href="/clusters_v{self.version}/{name}">{name}</a> ({separation:.2f}) &bullet;'
    return html[:-8]

  def v5_diff(self, name: str):
    if self.version != 6: return ''
    df_v5 = load_clusters()
    q = df_v5[df_v5.name == name]
    if len(q) == 0: return '<b>Differences between v5 and v6:</b> <i>not included in v5</i><br /><br />'
    id_v5 = q.clsid.values[0]
    catalog_v5 = load_members_v5(id_v5)
    catalog_v6 = load_members_v6(name)
    df = crossmatch(catalog_v6, catalog_v5, join='1not2')
    html = f'<b>Differences between v5 and v6:</b> <span class="badge text-bg-success">members included: {len(df[df.flag_member == 0])}</span> &nbsp; '
    html += f'<span class="badge text-bg-success">interlopers included: {len(df[df.flag_member == 1])}</span> &nbsp; '
    df = crossmatch(catalog_v6, catalog_v5, join='2not1')
    html += f'<span class="badge text-bg-danger">members excluded: {len(df[df.flag_member == 0])}</span> &nbsp; '
    html += f'<span class="badge text-bg-danger">interlopers excluded: {len(df[df.flag_member == 1])}</span><br /><br />'
    return html
  
  def run(
    self, 
    cls_name: str, 
    cls_ra: float, 
    cls_dec: float, 
    cls_z: float,
    cls_search_radius_deg: float,
    cls_search_radius_Mpc: float,
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
    folder_path = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name
    attachments = [
      'splus_fields_5r200.csv', 'splus_fields_5r500.csv', 
      'splus_fields_search_radius.csv'
    ]
    attachments_html = [f'<a href="{a}">{a}</a>' for a in attachments]
    images = [
      'specz', 'photoz', 'photoz_specz', 
      'spec_velocity_position', 'spec_velocity_rel_position', 
      'spec_velocity', 'specz_distance', 'photoz_distance', 
      'mag_diff', 'mag_diff_hist', 'redshift_diagonal', 
      'redshift_histogram_members', 'redshift_histogram_interlopers', 
      'redshift_histogram_all', 'xray',
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
    {self.get_head()}
    <body>
      {self.get_nav(cls_name)}
      
      <hr />
      
      <div class="container-fluid">
        <center><h2>{cls_name}</h2></center>
        <div class="row">
          <div class="col-4">
            <table class="table table-striped table-hover">
              <thead>
                <tr>
                  <th>Property</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Position</td>
                  <td>{cls_ra:.5f} {'+' if cls_dec >= 0 else ''}{cls_dec:.5f} (deg)</td>
                </tr>
                <tr>
                  <td>Search Radius</td>
                  <td>{cls_search_radius_Mpc:.2f} Mpc ({cls_search_radius_deg:.3f} deg)</td>
                </tr>
                <tr>
                  <td>5&times;R200</td>
                  <td>{5*cls_r200_Mpc:.3f} Mpc ({5*cls_r200_deg:.3f} deg)</td>
                </tr>
                <tr>
                  <td>5&times;R500</td>
                  <td>{5*cls_r500_Mpc:.3f} Mpc ({5*cls_r500_deg:.3f} deg)</td>
                </tr>
                <tr>
                  <td>z<sub>cluster</sub></td>
                  <td>{cls_z:.4f}</td>
                </tr>
                <tr>
                  <td>z<sub>spec</sub></td>
                  <td>{cls_z:.4f} &plusmn; 0.007 = [{z_spec_range[0]:.4f}, {z_spec_range[1]:.4f}]</td>
                </tr>
                <tr>
                  <td>z<sub>photo</sub></td>
                  <td>{cls_z:.4f} &plusmn; 0.015 = [{z_photo_range[0]:.4f}, {z_photo_range[1]:.4f}]</td>
                </tr>
                <tr>
                  <td>mag<sub>r</sub></td>
                  <td>[13, 22]</td>
                </tr>
                <tr>
                  <td>class<sub>spec</sub></td>
                  <td>GALAXY*</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="col-8">
            <b>Nearest clusters in this catalog (angular distance):</b> {self.get_nearest(cls_ra, cls_dec)}
            <br /><br />
            {self.v5_diff(cls_name)}
            <b>Attachments:</b> {' &nbsp;&bullet;&nbsp; '.join(attachments_html)}
            <br /><br />
            <b>Cosmology:</b>
            H<sub>0</sub>: 70 km Mpc<sup>-1</sup> s<sup>-1</sup> &nbsp;&nbsp;
            &Omega;<sub>m</sub>: 0.3 &nbsp;&nbsp;
            &Omega;<sub>&Lambda;</sub>: 0.7
          </div>
        </div>
        <br /><br />
        <center><h4>Gallery</h4></center>
        {' '.join(gallery)}
        <br /><br />
      </div>

      <div style="background-color: #201F1F;" class="w-100 text-white py-3">
        <center><h4>Legacy DR10</h4></center>
        <div id="aladin-lite-div" style="width: 90%; height: 700px; margin:0 auto;"></div>
      </div>
      
      {self.get_scripts()}
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
          var overlay = A.graphicOverlay({{name: '5R200', color: '#ee2345', lineWidth: 2}});
          aladin.addOverlay(overlay);
          overlay.add(A.circle({cls_ra:.6f}, {cls_dec:.6f}, {5*cls_r200_deg:.3f}, 0));
          
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