from functools import partial
from typing import List

import dagster as dg
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io.misc.yaml import AstropyDumper

from splusclusters._extraction import (download_xray, legacy_cone, make_cones,
                                       photoz_cone, specz_cone)
from splusclusters._loaders import (compute_cluster_info, load_catalog,
                                    load_cones, load_legacy_cone,
                                    load_photoz_cone, load_shiftgap_cone,
                                    load_spec, load_specz_cone)
from splusclusters._match import make_cluster_catalog
from splusclusters._plots import make_plots
from splusclusters._website import (build_cluster_page, copy_xray, make_index,
                                    make_zoffset_page)
from splusclusters.utils import config_dask


def single_cluster_pipeline(
  cls_name: str,
  df_clusters: pd.DataFrame,
  df_clusters_prev: pd.DataFrame,
  specz_df: pd.DataFrame,
  specz_coords: SkyCoord | None,
  version: int, 
  skip_cones: bool = False,
  skip_plots: bool = False,
  skip_website: bool = False,
  overwrite: bool = False,
  photoz_odds: float = 0.9,
  separated: bool = True,
  splus_only: bool = False,
  fmt: str = 'png',
):
  info = compute_cluster_info(df_clusters=df_clusters, cls_name=cls_name)
  
  if not skip_cones:
    make_cones(
      info=info, 
      specz_df=specz_df, 
      specz_skycoord=specz_coords, 
      overwrite=overwrite, 
      workers=7,
    )
  
  cones = load_cones(info, version)
  
  df_all = make_cluster_catalog(
    info=info, 
    df_specz_radial=cones.specz,
    df_photoz_radial=cones.photoz, 
    df_legacy_radial=cones.legacy,
    df_ret=cones.shiftgap,
    df_spec_all=specz_df,
    overwrite=overwrite,
  )
  
  if not skip_plots:
    make_plots(
      info=info,
      df_photoz_radial=cones.photoz,
      df_specz_radial=cones.specz,
      df_all_radial=df_all,
      df_members=cones.members,
      df_interlopers=cones.interlopers,
      df_legacy_radial=cones.legacy,
      version=version,
      photoz_odds=photoz_odds,
      separated=separated,
      overwrite=overwrite,
      splus_only=splus_only,
      fmt=fmt, 
    )
  
  if not skip_website:
    download_xray(info=info, overwrite=False, fmt='png')
    copy_xray(cls_name=info.name, version=version, fmt='png', overwrite=False)
    build_cluster_page(
      info=info,
      version=version,
      df_photoz_radial=cones.photoz,
      df_members=cones.members,
      df_clusters_prev=df_clusters_prev,
    )



def all_clusters_pipeline(
  version: int,
  skip_cones: bool = False,
  skip_plots: bool = False,
  skip_website: bool = False,
  overwrite: bool = False,
  photoz_odds: float = 0.9,
  separated: bool = True,
  splus_only: bool = False,
  fmt: str = 'png',
  two: bool = False,
):
  df_clusters = load_catalog(version)
  df_clusters_prev = load_catalog(version - 1)
  
  if two:
    df_clusters = df_clusters[df_clusters.name.isin(['A168', 'MKW4'])]
    if df_clusters_prev is not None:
      df_clusters_prev = df_clusters_prev[df_clusters_prev.name.isin(['A168', 'MKW4'])]
  
  specz_df, specz_coords = None, None
  if not skip_cones:
    # config_dask()
    # specz_df, specz_coords = load_spec()
    specz_df = load_spec(False)
  
  for cluster in df_clusters.name.values:
    single_cluster_pipeline(
      cls_name=cluster,
      df_clusters=df_clusters,
      df_clusters_prev=df_clusters_prev,
      specz_df=specz_df,
      specz_coords=specz_coords,
      version=version,
      skip_cones=skip_cones,
      skip_plots=skip_plots,
      skip_website=skip_website,
      overwrite=overwrite,
      photoz_odds=photoz_odds,
      separated=separated,
      splus_only=splus_only,
      fmt=fmt,
    )
  
  if not skip_website:
    make_zoffset_page(df_clusters, df_clusters_prev, version)
    make_index(df_clusters, df_clusters_prev, version)
