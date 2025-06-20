import os
import re
import signal
import subprocess
from shutil import copy
from typing import Tuple

import astropy.units as u
import lsdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import splusdata
from astromodule.io import read_table, write_table
from astromodule.legacy import LegacyService
from astromodule.pipeline import Pipeline, PipelineStage, PipelineStorage
from astromodule.splus import SplusService
from astropy.units import Quantity
from pylegs.archive import RadialMatcher
from pylegs.utils import Timer
from upath import UPath

from splusclusters.configs import configs
from splusclusters.utils import config_dask


class DownloadLegacyCatalogStage(PipelineStage):
  def __init__(self, radius_key: str, overwrite: bool = False, workers: int = 3):
    self.radius_key = radius_key
    self.overwrite = overwrite
    self.workers = workers
    
  def run(self, cls_ra: float, cls_dec: float, cls_name: str):
    out_path = configs.LEG_PHOTO_FOLDER / f'{cls_name}.parquet'
    if not self.overwrite and out_path.exists():
      return
    
    sql = """
      SELECT t.ra, t.dec, t.type, t.shape_r, t.mag_g, t.mag_r, t.mag_i, t.mag_z, 
      t.mag_w1, t.mag_w2, t.mag_w3, t.mag_w4
      FROM ls_dr10.tractor AS t
      WHERE (ra BETWEEN {ra_min} AND {ra_max}) AND 
      (dec BETWEEN {dec_min} AND {dec_max}) AND
      (brick_primary = 1) AND 
      (mag_r BETWEEN {r_min:.2f} AND {r_max:.2f})
    """.strip()
    
    radius = self.get_data(self.radius_key)
    queries = [
      sql.format(
        ra_min=cls_ra-radius, 
        ra_max=cls_ra+radius, 
        dec_min=cls_dec-radius,
        dec_max=cls_dec+radius,
        r_min=_r,
        r_max=_r+.05
      )
      for _r in np.arange(*configs.MAG_RANGE, .05)
    ]
    service = LegacyService(replace=True, workers=self.workers)
    service.batch_sync_query(
      queries=queries, 
      save_paths=out_path, 
      join_outputs=True, 
      workers=self.workers
    )



class ArchiveDownloadLegacyCatalogStage(PipelineStage):
  def __init__(
    self, 
    radius_key: str, 
    overwrite: bool = False, 
    overwrite_bricks: bool = False, 
    workers: int = 3
  ):
    self.radius_key = radius_key
    self.overwrite = overwrite
    self.overwrite_bricks = overwrite_bricks
    self.workers = workers
    
  def run(self, cls_ra: float, cls_dec: float, cls_name: str):
    out_path = configs.LEG_PHOTO_FOLDER / f'{cls_name}.parquet'
    if not self.overwrite and out_path.exists():
      return
    
    def f(table):
      mag_min, mag_max = configs.MAG_RANGE
      mask = (table['type'] != 'PSF') & (table['mag_r'] > mag_min) & (table['mag_r'] < mag_max)
      return table[mask]
    
    radius = self.get_data(self.radius_key)
    matcher = RadialMatcher(ra=cls_ra, dec=cls_dec, radius=radius)
    matcher.download_bricks(
      bricks_dir=configs.LEG_BRICKS_FOLDER, 
      columns=['ra', 'dec', 'type', 'mag_r'], 
      brick_primary=True, 
      compute_mag=['r'], 
      overwrite=self.overwrite_bricks, 
      workers=self.workers,
      filter_function=f,
    )
    matcher.match(
      output_path=out_path, 
      bricks_dir=configs.LEG_BRICKS_FOLDER, 
      overwrite=self.overwrite
    )
    



class DownloadXRayStage(PipelineStage):
  def __init__(self, overwrite: bool = False, fmt: str = 'png'):
    self.fmt = fmt
    self.overwrite = overwrite
    self.base_url = 'http://zmtt.bao.ac.cn/galaxy_clusters/dyXimages/image_all/'

  def run(self, cls_name: str):
    eps_path = configs.XRAY_PLOTS_FOLDER / f'{cls_name}.eps'
    raster_path = configs.XRAY_PLOTS_FOLDER / f'{cls_name}.{self.fmt}'
    if not eps_path.exists():
      url = self.base_url + cls_name + '_image.eps'
      r = requests.get(url)
      if r.ok:
        eps_path.write_bytes(r.content)
    if eps_path.exists() and (not raster_path.exists() or self.overwrite):
      # cbpfdown repos/clusters/outputs_v6/xray_plots/*.eps .
      # for i in *.eps; do convert -density 300 "$i" -trim -rotate 90 "${i%.*}.png"; done
      subprocess.run([
        'convert', '-density', '300', str(eps_path.absolute()), 
        '-trim', '-rotate', '90', str(raster_path.absolute())
      ])




class CopyXrayStage(PipelineStage):
  def __init__(self, overwrite: bool = False, fmt: str = 'png', version: int = 6):
    self.overwrite = overwrite
    self.fmt = fmt
    self.version = version
    
  def run(self, cls_name: str):
    src = configs.XRAY_PLOTS_FOLDER / f'{cls_name}.{self.fmt}'
    dst = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'xray.{self.fmt}'
    if (not dst.exists() or self.overwrite) and src.exists(): 
      copy(src, dst)
      
      
      
class SplusMembersMatchStage(PipelineStage):
  def __init__(self, overwrite: bool = False, version: int = 6):
    self.overwrite = overwrite
    self.version = version
  
  def run(self, cls_name: str, df_members: pd.DataFrame):
    out_path = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'members_ext.csv'
    if not self.overwrite and out_path.exists():
      return
    
    sql = """
      SELECT photo.RA AS RA_splus, photo.DEC AS DEC_splus, photo.g_auto, photo.i_auto, photo.J0378_auto, 
      photo.J0395_auto, photo.J0410_auto, photo.J0430_auto, photo.J0515_auto, 
      photo.J0660_auto, photo.J0861_auto, photo.r_auto, photo.u_auto, 
      photo.z_auto, photo.e_g_auto, photo.e_i_auto, photo.e_J0378_auto, 
      photo.e_J0395_auto, photo.e_J0410_auto, photo.e_J0430_auto, 
      photo.e_J0515_auto, photo.e_J0660_auto, photo.e_J0861_auto, 
      photo.e_r_auto, photo.e_u_auto, photo.e_z_auto, photo.THETA, 
      photo.ELLIPTICITY, photo.ELONGATION, photo.A, photo.PETRO_RADIUS, 
      photo.FLUX_RADIUS_50, photo.FLUX_RADIUS_90, photo.MU_MAX_g, photo.MU_MAX_r, 
      photo.BACKGROUND_g, photo.BACKGROUND_r, photo.s2n_g_auto, photo.s2n_r_auto,
      photoz.zml, photoz.odds
      FROM idr5.idr5_dual AS photo
      LEFT JOIN idr5_vacs.idr5_photoz AS photoz ON photo.ID = photoz.ID
      RIGHT JOIN TAP_UPLOAD.upload AS upl
      ON 1 = CONTAINS( 
        POINT('ICRS', photo.RA, photo.DEC), 
        CIRCLE('ICRS', upl.ra, upl.dec, 0.000277777777778) 
      )
    """
    
    if df_members is not None and len(df_members) > 0:
      u, p = os.environ['SPLUS_USER'], os.environ['SPLUS_PASS']
      service = SplusService(username=u, password=p)
      service.query(
        sql=sql,
        save_path=out_path,
        table=df_members,
        scope='private',
      )
    else:
      print('members dataframe not found, skiping')
    

class DownloadSplusPhotozStage(PipelineStage):
  def __init__(
    self, 
    radius_key: str = 'cls_search_radius_deg', 
    overwrite: bool = False, 
    workers: int = 10
  ):
    self.radius_key = radius_key
    self.overwrite = overwrite
    self.workers = workers
    
    
  def download(self, cls_name: str, cls_ra: float, cls_dec: float, z_photo_range: Tuple[float, float]):
    out_path = configs.PHOTOZ_FOLDER / f'{cls_name}.parquet'
    if not self.overwrite and out_path.exists():
      return
    
    # if out_path.exists():
    #   df = read_table(out_path)
    #   if 'g_aper_6' in df.columns:
    #     print('>> "g_aper_6" columns found: skiping download')
    #     return
    #   else:
    #     del df
        
    radius = self.get_data(self.radius_key)
    
    # config_dask()
    conn = splusdata.Core(username=os.environ['SPLUS_USER'], password=os.environ['SPLUS_PASS'])
    
    # iDR5 dual catalog
    t = Timer()
    print('Downloading Dual HIPS.', end='')
    idr5_links = splusdata.get_hipscats('idr5/dual', headers=conn.headers)[0]
    # idr5_margin = lsdb.read_hats(idr5_links[1], storage_options=dict(headers=conn.headers))
    dual = lsdb.read_hats(
      UPath(idr5_links[0], headers=conn.headers),
      # margin_cache=idr5_links[1],
      columns = [
        'RA', 'DEC', 'A', 'B', 'THETA', 'ELLIPTICITY',
        'PETRO_RADIUS', 'FLUX_RADIUS_20', 'FLUX_RADIUS_50', 'FLUX_RADIUS_90', 
        'MU_MAX_g', 'MU_MAX_r', 'BACKGROUND_g', 'BACKGROUND_r',
        's2n_g_auto', 's2n_r_auto',
        # auto mags
        'J0378_auto', 'J0395_auto', 'J0410_auto', 'J0430_auto', 'J0515_auto',
        'J0660_auto', 'J0861_auto', 'g_auto', 'i_auto', 'r_auto', 'u_auto', 
        'z_auto',
        # PStotal mags
        'J0378_PStotal', 'J0395_PStotal', 'J0410_PStotal', 'J0430_PStotal', 
        'J0515_PStotal', 'J0660_PStotal', 'J0861_PStotal', 'g_PStotal', 
        'i_PStotal', 'r_PStotal', 'u_PStotal', 'z_PStotal',
        # aper_6 mags
        'J0378_aper_6', 'J0395_aper_6', 'J0410_aper_6', 'J0430_aper_6', 'J0515_aper_6',
        'J0660_aper_6', 'J0861_aper_6', 'g_aper_6', 'i_aper_6', 'r_aper_6', 'u_aper_6', 
        'z_aper_6',
        # auto error
        'e_J0378_auto', 'e_J0395_auto', 'e_J0410_auto', 'e_J0430_auto', 
        'e_J0515_auto', 'e_J0660_auto', 'e_J0861_auto', 'e_g_auto', 'e_i_auto', 
        'e_r_auto', 'e_u_auto', 'e_z_auto',
        # PStotal error
        'e_J0378_PStotal', 'e_J0395_PStotal', 'e_J0410_PStotal', 'e_J0430_PStotal', 
        'e_J0515_PStotal', 'e_J0660_PStotal', 'e_J0861_PStotal', 'e_g_PStotal', 
        'e_i_PStotal', 'e_r_PStotal', 'e_u_PStotal', 'e_z_PStotal',
        # aper_6 mags
        'e_J0378_aper_6', 'e_J0395_aper_6', 'e_J0410_aper_6', 'e_J0430_aper_6', 'e_J0515_aper_6',
        'e_J0660_aper_6', 'e_J0861_aper_6', 'e_g_aper_6', 'e_i_aper_6', 'e_r_aper_6', 'e_u_aper_6', 
        'e_z_aper_6',
        # G mags
        'g_aper_3', 'g_res', 'g_iso', 'g_petro', 
        # R mags
        'r_aper_3', 'r_res', 'r_iso', 'r_petro', 
        'Field'
      ],
      filters=[
        ('r_auto', '<=', 22)
      ]
    )
    print(f' [OK] Duration: {t.duration_str}')

    # iDR5 photo-z
    t = Timer()
    print('Downloading Photo-z HIPS.', end='')
    idr5_pz = splusdata.get_hipscats('idr5/photoz', headers=conn.headers)[0]
    # idr5_pz_margin = lsdb.read_hats(idr5_pz[1], storage_options=dict(headers=conn.headers))
    pz = lsdb.read_hats(
      UPath(idr5_pz[0], headers=conn.headers),
      # margin_cache=idr5_pz[1],
      columns=[
        'RA', 'DEC', 'zml', 'odds', 'pdf_weights_0', 'pdf_weights_1', 
        'pdf_weights_2', 'pdf_means_0', 'pdf_means_1', 'pdf_means_2', 
        'pdf_stds_0', 'pdf_stds_1', 'pdf_stds_2',
      ],
      filters=[('zml', '>', z_photo_range[0]), ('zml', '<', z_photo_range[1])]
    )
    print(f' [OK] Duration: {t.duration_str}')

    # iDR5 SQG
    t = Timer()
    print('Downloading SQG HIPS.', end='')
    idr5_sqg = splusdata.get_hipscats('idr5/sqg', headers=conn.headers)[0]
    # idr5_sqg_margin = lsdb.read_hats(idr5_sqg[1], storage_options=dict(headers=conn.headers))
    sqg = lsdb.read_hats(
      UPath(idr5_sqg[0], headers=conn.headers),
      # margin_cache=idr5_sqg[1],
      columns=['RA', 'DEC', 'PROB_GAL_GAIA'],
      # filters=[('PROB_GAL', '>=', 0.5)]
    )
    print(f' [OK] Duration: {t.duration_str}')

    # iDR5 overlap flags
    t = Timer()
    print('Downloading Overlap HIPS.', end='')
    idr5_overlap = splusdata.get_hipscats('idr5/overlap_flags', headers=conn.headers)[0]
    # idr5_overlap_margin = lsdb.read_hats(idr5_overlap[1], storage_options=dict(headers=conn.headers))
    overlap = lsdb.read_hats(
      UPath(idr5_overlap[0], headers=conn.headers),
      # margin_cache=idr5_overlap[1],
      columns=['RA', 'DEC', 'in_overlap_region'],
      # filters=[('in_overlap_region', '==', 0)]
    )
    print(f' [OK] Duration: {t.duration_str}')
    
    t = Timer()
    print('\nCrossmatch: SQG <-> Dual.', end='')
    dual_sqg = sqg.crossmatch(
      dual, 
      radius_arcsec=1,
      suffixes=('_sqg', '_dual')
    )
    print(f' [OK] Duration: {t.duration_str}')

    t = Timer()
    print('Crossmatch: SQG, Dual <-> Photo-z.', end='')
    dual_sqg_pz = dual_sqg.crossmatch(
      pz, 
      radius_arcsec=1,
      suffixes=('_sqg', '_pz')
    )
    print(f' [OK] Duration: {t.duration_str}')

    t = Timer()
    print('Crossmatch: SQG, Dual, Photo-z <-> Overlap.', end='')
    dual_sqg_pz_overlap = dual_sqg_pz.crossmatch(
      overlap, 
      radius_arcsec=1,
      suffixes=('_pz', '_overlap')
    )
    print(f' [OK] Duration: {t.duration_str}')
    
    t = Timer()
    result = dual_sqg_pz_overlap.cone_search(
      cls_ra,
      cls_dec,
      radius * 3600 # radius in arcsecs
    )
    print('Table columns: ', end='')
    print(*result.columns, sep=', ')
    
    
    print(f'\nPerforming conesearch within {radius:.2f} deg.', end='')
    result = result.compute()
    print(f' [OK] Duration: {t.duration_str}')
      
    
    # drop repeated columns
    result = result.drop(columns=[
      'RA_dual_sqg_pz', 'DEC_dual_sqg_pz', '_dist_arcsec_sqg_pz',
      'RA_pz_pz', 'DEC_pz_pz', '_dist_arcsec_pz',
      'RA_overlap', 'DEC_overlap', '_dist_arcsec'
    ])
    
    # remove suffix
    def mapper(col: str):
      return col.replace('_sqg', '').replace('_dual', '').replace('_pz', '')\
                .replace('in_overlap_region_overlap', 'in_overlap_region')
    result = result.rename(columns=mapper)
    
    # filter overlap objects
    if 'in_overlap_region' in result.columns:
      if cls_name.upper() != 'MKW4':
        result = result[result['in_overlap_region'] == 0]
      # else:
      #   mask = (
      #     (result['in_overlap_region'] == 0) & (~result['Field'].isin([]))
      #   ) | (
      #     result['Field'].isin([])
      #   )
      #   result = result[mask]

    # result = result.rename(columns={
    #   'RA_sqg_sqg_pz': 'RA',
    #   'DEC_sqg_sqg_pz': 'DEC',
    #   'r_auto_dual_sqg_pz': 'r_auto',
    #   'r_PStotal_dual_sqg_pz': 'r_PStotal',
    #   'PROB_GAL_sqg_sqg_pz': 'PROB_GAL',
    #   'zml_pz_pz': 'zml',
    #   'odds_pz_pz': 'odds',
    #   'in_overlap_region_overlap': 'in_overlap_region',
    #   '_dist_arcsec': 'lsdb_separation'
    # })
    
    print('Final table columns:', *result.columns)
    print('\nTable rows:', len(result))
    print(result)
    write_table(result, out_path)
    
  
  def run(self, cls_name: str, cls_ra: float, cls_dec: float, z_photo_range: Tuple[float, float]):
    def timeout_handler(*kargs, **kwargs):
      raise Exception('Execution Timeout')
    
    success = False
    attempt = 1
    while attempt <= 6 and not success:
      try:
        print('\n>> Download Attempt', attempt, 'of', 6, '\n')
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(12*60) # 12 minutes
        
        self.download(
          cls_name=cls_name, 
          cls_ra=cls_ra, 
          cls_dec=cls_dec, 
          z_photo_range=z_photo_range,
        )
        
        signal.alarm(0)
        success = True
      except Exception as e:
        print('\n>> Attempt', attempt, 'failed!\n')
        attempt += 1
        print(e)




class FixZRange(PipelineStage):
  def run(self, cls_name: str, z_photo_range: Tuple[float, float]):
    out_path = configs.PHOTOZ_FOLDER / f'{cls_name}.parquet'
    if out_path.exists():
      df = read_table(out_path)
      if len(df) > 0:
        df = df[df.zml.between(*z_photo_range)]
        write_table(df, out_path)