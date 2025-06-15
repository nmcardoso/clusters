from shutil import copy
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from astromodule.io import write_table
from astromodule.pipeline import PipelineStage
from astromodule.table import crossmatch, radial_search
from astropy import units as u
from astropy.coordinates import SkyCoord
from pylegs.io import read_table

from splusclusters._loaders import ClusterInfo
from splusclusters.configs import configs
from splusclusters.loaders import (load_clusters, load_members_v5,
                                   load_members_v6)


def _get_cluster_folder(version: int):
  return configs.WEBSITE_PATH / f'clusters_v{version}'


def _get_head_tag():
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
  
def _get_scripts_tag():
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
  
def _get_nav_tag(
  df_clusters: pd.DataFrame, 
  df_clusters_prev: pd.DataFrame, 
  version: int,
  curr_name: str = None, 
):
  links = ''
  for name in sorted(df_clusters.name.values):
    active = 'active' if name == curr_name else ''
    
    if df_clusters_prev is not None and name not in df_clusters_prev.name.values:
      new_tag = '<span class="badge text-bg-secondary">New</span>'
    else:
      new_tag = ''
    
    links += (
      '<li class="nav-item">'
      f'<a class="nav-link {active}" href="/clusters_v{version}/{name}">'
      f'{name} {new_tag}'
      '</a>'
      '</li>'
    )
  
  html = f'''
    <nav class="navbar navbar-expand-lg bg-body-tertiary">
      <div class="container-fluid">
        <a class="navbar-brand" href="#">🌌 S-PLUS Clusters</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarSupportedContent">
          <ul class="navbar-nav me-auto mb-2 mb-lg-0">
            <li class="nav-item dropdown">
              <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown">
                Version {version}
              </a>
              <ul class="dropdown-menu">
                <li><a class="dropdown-item" href="/clusters_v7">Version 7</a></li>
                <li><a class="dropdown-item" href="/clusters_v6">Version 6</a></li>
                <li><a class="dropdown-item" href="/clusters_v5">Version 5</a></li>
              </ul>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="/clusters_v{version}/zoffset.html">Redshift offset</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="https://zoffset.streamlit.app">Odds App</a>
            </li>
            <li class="nav-item">
              <a class="nav-link" href="https://erosita.streamlit.app">Crossmatch App</a>
            </li>
          </ul>
        </div>
      </div>
    </nav>
    <ul class="nav nav-pills mt-2">
      {links}
    </ul>
  '''
  return html


  
def _get_nearest(info: ClusterInfo, df_clusters: pd.DataFrame, version: int):
  others = SkyCoord(df_clusters['ra'], df_clusters['dec'], unit='deg')
  sep = info.coord.separation(others)
  nearest_idx = sep.argsort()
  html = ''
  print(f'Cluster: {info.name}, nearest_idx: {nearest_idx}')
  for i in nearest_idx[1:8]:
    name = df_clusters['name'][i]
    separation = sep[i].to('arcmin')
    html += f' <a href="/clusters_v{version}/{name}">{name}</a> ({separation:.2f}) &bullet;'
  return html[:-8]



def _get_version_diff(name: str, df_clusters_prev: pd.DataFrame, version: int):
  if version <= 5 or df_clusters_prev is None: return ''
  q = df_clusters_prev[df_clusters_prev.name == name]
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




def make_zoffset_page(
  df_clusters: pd.DataFrame, 
  df_clusters_prev: pd.DataFrame, 
  version: int
):
  df = read_table(configs.Z_OFFSET_TABLE_PATH)
  rows = ''
  for i, row in df.iterrows():
    cls_m = [''] * 3
    cls_mi = [''] * 3
    cls_offset = [''] * 2
    err_m = np.asarray([row['rmse_om_m'], row['rmse_omi_m']])
    err_mi = np.asarray([row['rmse_om_mi'], row['rmse_omi_mi']])
    cls_m[np.argmin(err_m)] = 'table-info fw-bold'
    cls_mi[np.argmin(err_mi)] = 'table-info fw-bold'
    # cls_offset[np.argmin([err_m.min(), err_mi.min()])] = 'bg-success text-light fw-bold'
    rows += f"""
    <tr>
      <th>{i+1}</th>
      <td>{row['name']}</td>
      
      <td class="">{row['z_offset_m']:.4f}</td>
      <td class="">{row['z_offset_mi']:.4f}</td>
      
      <td>{row['rmse_base_m']:.3f} (0%)</td>
      <td class="{cls_m[0]}">{row['rmse_om_m']:.3f} ({row['rel_om_m']*100:.1f}%)</td>
      <td class="{cls_mi[0]}">{row['rmse_om_mi']:.3f} ({row['rel_om_mi']*100:.1f}%)</td>
      
      <td>{row['rmse_base_mi']:.3f} (0%)</td>
      <td class="{cls_m[1]}">{row['rmse_omi_m']:.3f} ({row['rel_omi_m']*100:.1f}%)</td>
      <td class="{cls_mi[1]}">{row['rmse_omi_mi']:.3f} ({row['rel_omi_mi']*100:.1f}%)</td>
      
      <td>
        <a href="{row['name']}/zoffset_baseline_m.jpg" class="gallery" data-lightbox="{row['name']}">
          <img height="120" src="{row['name']}/zoffset_baseline_m.jpg" />
        </a>
      </td>
      <td>
        <a href="{row['name']}/zoffset_m-shift_m.jpg" class="gallery" data-lightbox="{row['name']}">
          <img height="120" src="{row['name']}/zoffset_m-shift_m.jpg" />
        </a>
      </td>
      <td>
        <a href="{row['name']}/zoffset_baseline_mi.jpg" class="gallery" data-lightbox="{row['name']}">
          <img height="120" src="{row['name']}/zoffset_baseline_mi.jpg" />
        </a>
      </td>
      <td>
        <a href="{row['name']}/zoffset_mi-shift_mi.jpg" class="gallery" data-lightbox="{row['name']}">
          <img height="120" src="{row['name']}/zoffset_mi-shift_mi.jpg" />
        </a>
      </td>
    </tr>
    """
  
  html = f'''<!DOCTYPE html>
  <html>
  {_get_head_tag()}
  <body>
    {_get_nav_tag(df_clusters=df_clusters, df_clusters_prev=df_clusters_prev, version=version)}
    
    <hr />
    
    <div class="container-fluid">
      <h6>Columns description:</h6>
      <ul>
        <li><b>off<sub>M</sub>:</b> redshift offset computed in members sample</li>
        <li><b>off<sub>M+I</sub>:</b> redshift offset computed in members+interlopers sample</li>
        
        <li><b>e<sub>base</sub> (M):</b> RMSE between reference and estimated redshift without any correction (baseline) in the members sample</li>
        <li><b>e<sub>M</sub> (M):</b> RMSE between reference and estimated redshift corrected by members-offset in the members sample</li>
        <li><b>e<sub>M</sub> (M+I):</b> RMSE between reference and estimated redshift corrected by members-offset in the members+interlopers sample</li>
        
        <li><b>e<sub>base</sub> (M+I):</b> RMSE between reference and estimated redshift without any correction (baseline) in the members+interlopers sample</li>
        <li><b>e<sub>M+I</sub> (M):</b> RMSE between reference and estimated redshift corrected by members+interlopers-offset in the members sample</li>
        <li><b>e<sub>M+I</sub> (M+I):</b> RMSE between reference and estimated redshift corrected by members+interlopers-offset in the members+interlopers sample</li>
      </ul>
    </div>
    
    <div class="container-fuild w-100">
      <table class="table table-hover align-middle w-100">
        <thead class="sticky-top">
          <th>#</th>
          <th>cluster</th>
          
          <th>off<sub>M</sub></th>
          <th>off<sub>M+I</sub></th>
          
          <th>e<sub>base</sub> (M)</th>
          <th>e<sub>M</sub> (M)</th>
          <th>e<sub>M</sub> (M+I)</th>
          
          <th>e<sub>base</sub> (M+I)</th>
          <th>e<sub>M+I</sub> (M)</th>
          <th>e<sub>M+I</sub> (M+I)</th>
          
          <th>No correction (M)</th>
          <th>M-shift (M)</th>
          <th>No correction (MI)</th>
          <th>MI-shift (MI)</th>
        </thead>
        
        <tbody class="table-group-divider">
          {rows}
        </tbody>
      </table>
    </div>
    {_get_scripts_tag()}
  </body>
  </html>
  '''
  
  zoffset_path = _get_cluster_folder(version) / 'zoffset.html'
  zoffset_path.parent.mkdir(parents=True, exist_ok=True)
  zoffset_path.write_text(html)




def copy_xray(info: ClusterInfo, overwrite: bool = False):
  src = info.plot_xray_raster_path
  dst = info.website_cluster_page / src.name
  if (not dst.exists() or overwrite) and src.exists(): 
    copy(src, dst)




def make_splus_fields_tables(info: ClusterInfo, version: int, df: pd.DataFrame):
  base_path = _get_cluster_folder(version) / info.name
  r200_path = base_path / 'splus_fields_5r200.csv'
  r500_path = base_path / 'splus_fields_5r500.csv'
  search_radius_path = base_path / 'splus_fields_search_radius.csv'
  if not r200_path.exists() or not r500_path.exists() or not search_radius_path.exists():
  # if True:
    coords = SkyCoord(df.ra, df.dec, unit='deg')
    df_r200 = radial_search(info.coord, df, 5*info.r200_deg * u.deg, cached_catalog=coords)
    df_r500 = radial_search(info.coord, df, 5*info.r500_deg * u.deg, cached_catalog=coords)
    fields_r200 = df_r200.groupby('field').size().reset_index(name='n_objects')
    fields_r500 = df_r500.groupby('field').size().reset_index(name='n_objects')
    fields_search_radius = df.groupby('field').size().reset_index(name='n_objects')
    write_table(fields_r200, r200_path)
    write_table(fields_r500, r500_path)
    write_table(fields_search_radius, search_radius_path)
    
    
    

def make_index(df_clusters: pd.DataFrame, df_clusters_prev: pd.DataFrame, version: int):
  page = f'''<!DOCTYPE html>
  <html>
  {_get_head_tag()}
  <body>
    {_get_nav_tag(df_clusters, df_clusters_prev, version)}
    <br /><br /><br />
    <img src="/all_sky.png" width="80%" style="display: block; margin: 0 auto;" />
    {_get_scripts_tag()}
  </body>
  </html>
  '''
  index_path = _get_cluster_folder(version) / 'index.html'
  index_path.parent.mkdir(parents=True, exist_ok=True)
  index_path.write_text(page)





def make_landing():
  page = f'<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=clusters_v6"></head></html>'
  index_path = configs.WEBSITE_PATH / 'index.html'
  index_path.parent.mkdir(parents=True, exist_ok=True)
  index_path.write_text(page)






def make_cluster_page(
  info: ClusterInfo, 
  df_clusters: pd.DataFrame, 
  df_clusters_prev: pd.DataFrame, 
  version: int
):
  width = 400
  height = 400
  folder_path = _get_cluster_folder(version) / info.name
  attachments = [
    'splus_fields_5r200.csv', 'splus_fields_5r500.csv', 
    'splus_fields_search_radius.csv'
  ]
  attachments_html = [f'<a href="{a}">{a}</a>' for a in attachments]
  filter_plots = [
    info.plot_redshift_hist_members_path.name, 
    info.plot_redshift_hist_interlopers_path.name,
    info.plot_redshift_hist_all_path.name,
  ]
  
  images = [
    p.name for p in folder_path.glob(f'*.{info.plot_format}') 
    if p.name not in filter_plots
  ]
      
  gallery = [
    f'<a href="{img}" class="gallery" data-lightbox="images"><img src="{img}" width="{width}" height="{height}" /></a>'
    for img in images
  ]
  
  cat_url = (
    f'http://cdsxmatch.u-strasbg.fr/QueryCat/QueryCat'
    f'?catName=SIMBAD&mode=cone&pos={info.ra:.6f}%20{info.dec:.6f}'
    f'&r={5 * info.r200_deg:.3f}deg&format=votable&limit=8000'
  )
  
  page = f'''<!DOCTYPE html>
  <html>
  {_get_head_tag()}
  <body>
    {_get_nav_tag(df_clusters=df_clusters, df_clusters_prev=df_clusters_prev, version=version, curr_name=info.name)}
    
    <hr />
    
    <div class="container-xxl">
      <center><h2>{info.name}</h2></center>
      <div class="row mt-4">
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
                <td>position</td>
                <td>{info.ra:.5f} {'+' if info.dec >= 0 else ''}{info.dec:.5f} (deg)</td>
              </tr>
              <tr>
                <td>search radius</td>
                <td>{info.search_radius_Mpc:.2f} Mpc ({info.search_radius_deg:.3f} deg)</td>
              </tr>
              <tr>
                <td>5&times;R200</td>
                <td>{5*info.r200_Mpc:.3f} Mpc ({5*info.r200_deg:.3f} deg)</td>
              </tr>
              <tr>
                <td>5&times;R500</td>
                <td>{5*info.r500_Mpc:.3f} Mpc ({5*info.r500_deg:.3f} deg)</td>
              </tr>
              <tr>
                <td>z<sub>cluster</sub></td>
                <td>{info.z:.4f}</td>
              </tr>
              <tr>
                <td>z<sub>spec</sub> range</td>
                <td>
                  {info.z:.4f} &plusmn; 0.007 = [{info.z_spec_range[0]:.4f}, 
                  {info.z_spec_range[1]:.4f}]
                </td>
              </tr>
              <tr>
                <td>z<sub>photo</sub> range</td>
                <td>
                  {info.z:.4f} &plusmn; 0.015 = [{info.z_photo_range[0]:.4f}, 
                  {info.z_photo_range[1]:.4f}]
                </td>
              </tr>
              <tr>
                <td>mag<sub>r</sub> range</td>
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
          <b>Nearest clusters in this catalog (angular distance):</b> {_get_nearest(info, df_clusters, version)}
          <br /><br />
          {_get_version_diff(info.name, df_clusters_prev, version)}
          <b>Attachments:</b> {' &nbsp;&bullet;&nbsp; '.join(attachments_html)}
          <br /><br />
          <b>Cosmology:</b>
          H<sub>0</sub> = 70 [km Mpc<sup>-1</sup> s<sup>-1</sup>] &nbsp;&nbsp;&nbsp;
          &Omega;<sub>m</sub> = 0.3 &nbsp;&nbsp;&nbsp;
          &Omega;<sub>&Lambda;</sub> = 0.7
        </div>
      </div>
      <br /><br />
      <center><h4>Gallery</h4></center>
      <div class="grid text-center">
        {' '.join(gallery)}
      </div>
      <br /><br />
    </div>

    <div style="background-color: #201F1F;" class="w-100 text-white mt-2 py-3">
      <center><h4 class="mb-2">Legacy DR10</h4></center>
      <div id="aladin-lite-div" style="width: 90%; height: 700px; margin:0 auto;"></div>
    </div>
    
    {_get_scripts_tag()}
    <script>
    const catFilter = (source) => {{
      return (
        !isNaN(parseFloat(source.data['redshift'])) && 
        (source.data['redshift'] > {info.z_spec_range[0]}) && 
        (source.data['redshift'] < {info.z_spec_range[1]})
      )
    }}
    
    var aladin;
    window.addEventListener("load", () => {{
      A.init.then(() => {{
        // Init Aladin
        aladin = A.aladin('#aladin-lite-div', {{
          source: 'CDS/P/DESI-Legacy-Surveys/DR10/color',
          target: '{info.ra:.6f} {info.dec:.6f}', 
          fov: {5 * info.r200_deg + 0.3},
          cooFrame: 'ICRSd',
        }});
        aladin.setImageSurvey('CDS/P/DESI-Legacy-Surveys/DR10/color');
        
        // Add 5R200 Circle
        var overlay = A.graphicOverlay({{name: '5R200', color: '#ee2345', lineWidth: 2}});
        aladin.addOverlay(overlay);
        overlay.add(A.circle({info.ra:.6f}, {info.dec:.6f}, {5 * info.r200_deg:.3f}, 0));
        
        // Add redshift catalog
        const cat_url = '{cat_url}'
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






def build_cluster_page(
  info: ClusterInfo,
  version: int,
  df_photoz_radial: pd.DataFrame,
  df_members: pd.DataFrame,
  df_clusters: pd.DataFrame,
  df_clusters_prev: pd.DataFrame,
):
  make_cluster_page(
    info=info, 
    df_clusters=df_clusters, 
    df_clusters_prev=df_clusters_prev,
    version=version
  )
  
  make_splus_fields_tables(info=info, df=df_photoz_radial)
  
  if df_members is not None:
    folder_path = _get_cluster_folder(version) / info.name
    write_table(df_members, folder_path / 'members.csv')