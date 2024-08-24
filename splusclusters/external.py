import os
import subprocess
from shutil import copy

import matplotlib.pyplot as plt
import numpy as np
import requests
from astromodule.legacy import LegacyService
from astromodule.pipeline import Pipeline, PipelineStage, PipelineStorage
from astromodule.splus import SplusService
from pylegs.archive import RadialMatcher

from splusclusters.configs import configs


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
      SELECT t.ra, t.dec, t.type, t.mag_r
      FROM ls_dr10.tractor AS t
      WHERE (ra BETWEEN {ra_min} AND {ra_max}) AND 
      (dec BETWEEN {dec_min} AND {dec_max}) AND
      (brick_primary = 1) AND 
      (mag_r BETWEEN {r_min:.2f} AND {r_max:.2f}) AND
      (type != 'PSF')
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
  def __init__(self, overwrite: bool = False, workers: int = 6):
    self.overwrite = overwrite
    self.workers = workers
  
  def run(self, cls_name: str, cls_ra: float, cls_dec: float):
    out_path = configs.LEG_PHOTO_FOLDER / f'{cls_name}.parquet'
    if not self.overwrite and out_path.exists():
      return
    
    sql = """
      SELECT photo.RA, photo.DEC, photo.g_auto, photo.i_auto, photo.J0378_auto, 
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
      JOIN idr5_vacs.idr5_photoz AS photoz ON photo.ID = photoz.ID 
      WHERE 1 = CONTAINS( 
        POINT('ICRS', photo.RA, photo.DEC), 
        CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.6f}) 
      ) AND photo.r_auto BETWEEN {r_min:.3f} AND {r_max:.3f} 
    """
    
    radius = self.get_data(self.radius_key)
    queries = [
      sql.format(
        ra=cls_ra,
        dec=cls_dec,
        radius=radius,
        r_min=_r,
        r_max=_r+.05
      )
      for _r in np.arange(*configs.MAG_RANGE, .05)
    ]
    service = SplusService(username=os.environ['SPLUS_USER'], password=os.environ['SPLUS_PASS'])
    service.batch_query(
      sql=queries,
      save_path=out_path,
      join=True,
      workers=self.workers
    )
    

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
  
  def run(self, cls_name: str, cls_ra: float, cls_dec: float):
    out_path = configs.PHOTOZ_FOLDER / f'{cls_name}.parquet'
    if not self.overwrite and out_path.exists():
      return
    
    sql = """
      SELECT photo.RA AS ra, photo.DEC AS dec, photoz.r_auto, photoz.zml, 
      photoz.odds
      FROM idr5_vacs.idr5_photoz AS photoz
      WHERE 1 = CONTAINS( 
        POINT('ICRS', photo.RA, photo.DEC), 
        CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.6f}) 
      ) AND photo.r_auto BETWEEN {r_min:.3f} AND {r_max:.3f} 
    """
    
    radius = self.get_data(self.radius_key)
    queries = [
      sql.format(
        ra=cls_ra,
        dec=cls_dec,
        radius=radius,
        r_min=_r,
        r_max=_r+.05
      )
      for _r in np.arange(*configs.MAG_RANGE, .05)
    ]
    service = SplusService(username=os.environ['SPLUS_USER'], password=os.environ['SPLUS_PASS'])
    service.batch_query(
      sql=queries,
      save_path=out_path,
      join=True,
      workers=self.workers,
      scope='private'
    )