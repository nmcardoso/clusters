import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from shutil import copy
from typing import Dict, Literal, Sequence, Tuple

import astropy.units as u
import dagster as dg
import lsdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import splusdata
from astromodule.io import merge_pdf, read_table, write_table
from astromodule.legacy import LegacyService
from astromodule.pipeline import Pipeline, PipelineStage, PipelineStorage
from astromodule.splus import SplusService
from astromodule.table import (concat_tables, crossmatch, fast_crossmatch,
                               guess_coords_columns, radial_search, selfmatch)
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.misc.yaml import AstropyDumper
from astropy.units import Quantity
from pylegs.archive import RadialMatcher
from pylegs.utils import Timer
from upath import UPath

from splusclusters._loaders import ClusterInfo, load_spec
from splusclusters.configs import configs
from splusclusters.loaders import remove_bad_objects
from splusclusters.utils import (Timming, cond_overwrite, config_dask,
                                 return_table_if_exists)


class EmptyDataFrameError(Exception):
  def __init__(self, path: str | Path = None, size: int = None):
    if path is not None:
      path = Path(path)
      self.message = f'The dataframe {path} is empty, file size: {size} bytes'
    else:
      self.message = f'Empty dataframe'
    super().__init__(self.message)



def _check_empty_dataframe(path: str | Path, error: bool = True):
  path = Path(path)
  if path.exists() and path.stat().st_size <= 600:
    path.unlink()
    if error:
      raise EmptyDataFrameError(path, path.stat().st_size)


def specz_cone(
  info: ClusterInfo,
  overwrite: bool = False,
  in_range: bool = True,
) -> pd.DataFrame | None:
  path = info.specz_path if in_range else info.specz_outrange_path
  df = None
  
  with cond_overwrite(path, overwrite, mkdir=True, time=True) as cm:
    specz_df, specz_skycoord = load_spec()
    print(*specz_df.columns, sep=', ')
    if specz_skycoord is None:
      specz_skycoord = SkyCoord(
        ra=specz_df.ra_spec_all.values, 
        dec=specz_df.dec_spec_all.values, 
        unit=u.deg
      )
    df = radial_search(
      position=info.coord, 
      table=specz_df, 
      radius=info.search_radius_deg * u.deg,
      cached_catalog=specz_skycoord,
      ra='ra_spec_all',
      dec='dec_spec_all',
    )
    
    df = df.rename(columns={'ra_spec_all': 'ra', 'dec_spec_all': 'dec'})
    
    mask = (
      df.class_spec.str.upper().str.startswith('GALAXY') &
      df.f_z.str.upper().str.startswith('KEEP')
    )
    if in_range:
      mask &= df.z.between(*info.z_spec_range)
    else:
      mask &= ~df.z.between(*info.z_spec_range)
    
    df = df[mask]
    cm.write_table(df)
  return return_table_if_exists(path, df)



def legacy_cone(
  info: ClusterInfo, 
  workers: int = 3, 
  overwrite: bool = False
) -> pd.DataFrame | None:
  out_path = info.legacy_path
  _check_empty_dataframe(out_path, error=False)
  with cond_overwrite(out_path, overwrite, mkdir=True):
    sql = """
      SELECT t.ra, t.dec, t.type, t.shape_r, t.mag_g, t.mag_r, t.mag_i, t.mag_z, 
      t.mag_w1, t.mag_w2, t.mag_w3, t.mag_w4
      FROM ls_dr10.tractor AS t
      WHERE (ra BETWEEN {ra_min} AND {ra_max}) AND 
      (dec BETWEEN {dec_min} AND {dec_max}) AND
      (brick_primary = 1) AND 
      (mag_r BETWEEN {r_min:.2f} AND {r_max:.2f})
    """.strip()

    queries = [
      sql.format(
        ra_min=info.ra - info.search_radius_deg, 
        ra_max=info.ra + info.search_radius_deg, 
        dec_min=info.dec - info.search_radius_deg,
        dec_max=info.dec + info.search_radius_deg,
        r_min=_r,
        r_max=_r + .05
      )
      for _r in np.arange(*info.magnitude_range, .05)
    ]
    service = LegacyService(replace=True, workers=workers)
    service.batch_sync_query(
      queries=queries, 
      save_paths=out_path, 
      join_outputs=True, 
      workers=workers
    )
    _check_empty_dataframe(out_path)
  return return_table_if_exists(out_path)



def photoz_cone(
  info: ClusterInfo, 
  overwrite: bool = False
) -> pd.DataFrame | None:
  result = None
  out_path = info.photoz_path
  _check_empty_dataframe(out_path, error=False)
  with cond_overwrite(out_path, overwrite) as cm:
    # config_dask()
    conn = splusdata.Core(
      username=os.environ['SPLUS_USER'], 
      password=os.environ['SPLUS_PASS']
    )
    
    # iDR5 dual catalog
    t = Timer()
    print('Downloading Dual HIPS.', end='')
    idr5_links = splusdata.get_hipscats('idr5/dual', headers=conn.headers)[0]
    dual = lsdb.read_hats(
      UPath(idr5_links[0], headers=conn.headers),
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
        'J0378_aper_6', 'J0395_aper_6', 'J0410_aper_6', 'J0430_aper_6', 
        'J0515_aper_6', 'J0660_aper_6', 'J0861_aper_6', 'g_aper_6', 'i_aper_6', 
        'r_aper_6', 'u_aper_6', 'z_aper_6',
        # auto error
        'e_J0378_auto', 'e_J0395_auto', 'e_J0410_auto', 'e_J0430_auto', 
        'e_J0515_auto', 'e_J0660_auto', 'e_J0861_auto', 'e_g_auto', 'e_i_auto', 
        'e_r_auto', 'e_u_auto', 'e_z_auto',
        # PStotal error
        'e_J0378_PStotal', 'e_J0395_PStotal', 'e_J0410_PStotal', 'e_J0430_PStotal', 
        'e_J0515_PStotal', 'e_J0660_PStotal', 'e_J0861_PStotal', 'e_g_PStotal', 
        'e_i_PStotal', 'e_r_PStotal', 'e_u_PStotal', 'e_z_PStotal',
        # aper_6 mags
        'e_J0378_aper_6', 'e_J0395_aper_6', 'e_J0410_aper_6', 'e_J0430_aper_6', 
        'e_J0515_aper_6', 'e_J0660_aper_6', 'e_J0861_aper_6', 'e_g_aper_6', 
        'e_i_aper_6', 'e_r_aper_6', 'e_u_aper_6', 'e_z_aper_6',
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
    pz = lsdb.read_hats(
      UPath(idr5_pz[0], headers=conn.headers),
      columns=[
        'RA', 'DEC', 'zml', 'odds', 'pdf_weights_0', 'pdf_weights_1', 
        'pdf_weights_2', 'pdf_means_0', 'pdf_means_1', 'pdf_means_2', 
        'pdf_stds_0', 'pdf_stds_1', 'pdf_stds_2',
      ],
      filters=[('zml', '>', info.z_photo_range[0]), ('zml', '<', info.z_photo_range[1])]
    )
    print(f' [OK] Duration: {t.duration_str}')

    # iDR5 SQG
    t = Timer()
    print('Downloading SQG HIPS.', end='')
    idr5_sqg = splusdata.get_hipscats('idr5/sqg', headers=conn.headers)[0]
    sqg = lsdb.read_hats(
      UPath(idr5_sqg[0], headers=conn.headers),
      columns=['RA', 'DEC', 'PROB_GAL_GAIA'],
      # filters=[('PROB_GAL', '>=', 0.5)]
    )
    print(f' [OK] Duration: {t.duration_str}')

    # iDR5 overlap flags
    t = Timer()
    print('Downloading Overlap HIPS.', end='')
    idr5_overlap = splusdata.get_hipscats('idr5/overlap_flags', headers=conn.headers)[0]
    overlap = lsdb.read_hats(
      UPath(idr5_overlap[0], headers=conn.headers),
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
      info.ra,
      info.dec,
      info.search_radius_deg * 3600 # radius in arcsecs
    )
    print('Table columns: ', end='')
    print(*result.columns, sep=', ')
    
    
    print(f'\nPerforming conesearch within {info.search_radius_deg:.2f} deg.', end='')
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
      if info.name.upper() != 'MKW4':
        result = result[result['in_overlap_region'] == 0]

    
    print('Final table columns:', *result.columns)
    print('\nTable rows:', len(result))
    print(result)
    write_table(result, out_path)
    _check_empty_dataframe(out_path)
  return return_table_if_exists(out_path, result)



def download_xray(
  info: ClusterInfo, 
  overwrite: bool = False,
  fmt: str = 'png',
):
  eps_path = info.plot_xray_vector_path
  raster_path = info.plot_xray_raster_path
  if not eps_path.exists():
    base_url = 'http://zmtt.bao.ac.cn/galaxy_clusters/dyXimages/image_all/'
    url = base_url + info.name + '_image.eps'
    r = requests.get(url)
    if r.ok:
      eps_path.write_bytes(r.content)
  if eps_path.exists() and (not raster_path.exists() or overwrite):
    # cbpfdown repos/clusters/outputs_v6/xray_plots/*.eps .
    # for i in *.eps; do convert -density 300 "$i" -trim -rotate 90 "${i%.*}.png"; done
    print('Python interpreter:', sys.executable)
    print('Machine:', os.environ.get('MACHINE'))
    if os.environ.get('MACHINE', '').lower() == 'cbpf':
      program = str((Path(sys.executable).parent / 'convert').absolute())
    else:
      program = 'convert'
    subprocess.run([
      program, '-density', '300', str(eps_path.absolute()), 
      '-trim', '-rotate', '90', str(raster_path.absolute())
    ])



def photoz_cone_old(
  photoz_df: pd.DataFrame,
  photoz_skycoord: SkyCoord,
  info: ClusterInfo,
  overwrite: bool = False,
):
  df = None
  path = info.photoz_path
  with cond_overwrite(path, overwrite, mkdir=True, time=True) as cm:
    df = radial_search(
      position=info.coord, 
      table=photoz_df, 
      radius=info.search_radius_deg * u.deg,
      cached_catalog=photoz_skycoord,
    )
    
    if 'r_auto' in df.columns:
      df = df[
        # df.zml.between(*z_photo_range) &
        df.r_auto.between(*info.magnitude_range)
      ]
    
    cm.write_table(df)
  if path.exists() and df is None: df = read_table(path)
  return df



def legacy_cone_old(
  legacy_df: pd.DataFrame,
  legacy_skycoord: SkyCoord,
  info: ClusterInfo,
  overwrite: bool = False,
):
  df = None
  path = info.legacy_path
  with cond_overwrite(path, overwrite, mkdir=True, time=True) as cm:
    df = radial_search(
      position=info.coord, 
      table=legacy_df, 
      radius=info.search_radius_deg * u.deg,
      cached_catalog=legacy_skycoord,
      ra='ra',
      dec='dec',
    )
    
    cm.write_table(df)
  if path.exists() and df is None: df = read_table(path)
  return df




def splus_members_match_old(
  info: ClusterInfo, 
  df_members: pd.DataFrame, 
  overwrite: bool = False
):
  out_path = info.website_cluster_page / f'members_ext.csv'
  with cond_overwrite(out_path, overwrite, mkdir=True):
    sql = """
      SELECT photo.RA AS RA_splus, photo.DEC AS DEC_splus, photo.g_auto, 
      photo.i_auto, photo.J0378_auto, photo.J0395_auto, photo.J0410_auto, 
      photo.J0430_auto, photo.J0515_auto, photo.J0660_auto, photo.J0861_auto, 
      photo.r_auto, photo.u_auto, photo.z_auto, photo.e_g_auto, photo.e_i_auto, 
      photo.e_J0378_auto, photo.e_J0395_auto, photo.e_J0410_auto, 
      photo.e_J0430_auto, photo.e_J0515_auto, photo.e_J0660_auto, 
      photo.e_J0861_auto, photo.e_r_auto, photo.e_u_auto, photo.e_z_auto, 
      photo.THETA, photo.ELLIPTICITY, photo.ELONGATION, photo.A, 
      photo.PETRO_RADIUS, photo.FLUX_RADIUS_50, photo.FLUX_RADIUS_90, 
      photo.MU_MAX_g, photo.MU_MAX_r, photo.BACKGROUND_g, photo.BACKGROUND_r, 
      photo.s2n_g_auto, photo.s2n_r_auto,photoz.zml, photoz.odds
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




def make_cones(
  info: ClusterInfo,
  specz_df: pd.DataFrame,
  specz_skycoord: SkyCoord | None,
  overwrite: bool = False,
  workers: int = 5,
):
  cone_functions = [
    specz_cone,
    photoz_cone,
    legacy_cone,
  ]
  
  cone_params = [
    dict(specz_df=specz_df, specz_skycoord=specz_skycoord, info=info, overwrite=overwrite),
    dict(info=info, overwrite=overwrite),
    dict(info=info, workers=workers, overwrite=overwrite),
  ]
  
  futures = [f.submit(**p) for f, p in zip(cone_functions, cone_params)]
  # wait(futures)