from typing import List, Optional, Tuple

import dagster as dg
import pandas as pd
from astropy.coordinates import SkyCoord
from pylegs.io import read_table

from splusclusters._extraction import (create_zoffset_table, download_xray,
                                       legacy_cone, photoz_cone, specz_cone)
from splusclusters._loaders import (ClusterInfo, compute_cluster_info,
                                    load_catalog, load_shiftgap_cone,
                                    load_spec)
from splusclusters._match import make_cluster_catalog
from splusclusters._plots import make_plots
from splusclusters._website import (build_cluster_page, copy_xray, make_index,
                                    make_zoffset_page)


class ConfigResource(dg.ConfigurableResource):
  version: Optional[int] = 7
  subset: Optional[bool] = False
  overwrite: Optional[bool] = False
  workers: Optional[int] = 5
  magnitude_min: float = 13
  magnitude_max: float = 22
  
  z_spec_delta: Optional[float] = 0.02
  z_photo_delta: Optional[float] = 0.05
  
  photoz_odds: Optional[float] = 0.9
  separated_plots: Optional[bool] = True
  splus_only_plots: Optional[bool] = False
  plot_format: Optional[str] = 'png'
  
  skip_cones: Optional[bool] = False
  skip_plots: Optional[bool] = False
  skip_website: Optional[bool] = False


@dg.op(
  tags={'io': 'intensive'},
  out={
    'specz_all_df': dg.Out(pd.DataFrame), 
    'specz_all_coords': dg.Out(SkyCoord),
  }
)
def op_load_spec():
  return load_spec()


@dg.op(out=dg.Out(ClusterInfo), pool='cluster_info')
def op_compute_cluster_info(conf: ConfigResource, cls_name: str) -> ClusterInfo:
  return compute_cluster_info(
    cls_name=cls_name,
    z_spec_delta=conf.z_spec_delta,
    z_photo_delta=conf.z_photo_delta,
    version=conf.version,
    plot_format=conf.plot_format,
    magnitude_range=(conf.magnitude_min, conf.magnitude_max),
    subset=conf.subset,
  )


@dg.op(pool='cluster')
def op_specz_cone(conf: ConfigResource, info: ClusterInfo):
  specz_cone(info=info, overwrite=conf.overwrite)


@dg.op(pool='cluster')
def op_specz_cone_outrange(conf: ConfigResource, info: ClusterInfo):
  specz_cone(info=info, overwrite=conf.overwrite, in_range=False)


@dg.op(tags={'remote': 'splus'}, pool='cluster', retry_policy=dg.RetryPolicy(3))
def op_photoz_cone(conf: ConfigResource, info: ClusterInfo):
  photoz_cone(info=info, overwrite=conf.overwrite)


@dg.op(tags={'remote': 'datalab'}, pool='legacy', retry_policy=dg.RetryPolicy(3))
def op_legacy_cone(conf: ConfigResource, info: ClusterInfo):
  legacy_cone(
    info=info, 
    workers=conf.workers, 
    overwrite=conf.overwrite
  )


@dg.op(pool='cluster')
def op_shiftgap_cone(conf: ConfigResource, info: ClusterInfo):
  load_shiftgap_cone(info=info, version=conf.version)


@dg.op(
  tags={'crossmatch': 'intensive'}, 
  ins={
    'specz_cone': dg.In(dg.Nothing), 'photoz_cone': dg.In(dg.Nothing), 
    'legacy_cone': dg.In(dg.Nothing), 'specz_outrange_cone': dg.In(dg.Nothing)
  },
  pool='cluster_intensive'
)
def op_compile_cluster_catalog(conf: ConfigResource, info: ClusterInfo):
  shiftgap_df, _, _ = load_shiftgap_cone(info, conf.version)
  make_cluster_catalog(
    info=info,
    df_specz_radial=info.specz_df,
    df_photoz_radial=info.photoz_df,
    df_legacy_radial=info.legacy_df,
    df_ret=shiftgap_df,
    df_specz_outrange_radial=info.specz_outrange_df,
    overwrite=conf.overwrite,
  )


@dg.op(ins={'start_after': dg.In(dg.Nothing)}, pool='cluster')
def op_render_plots(conf: ConfigResource, info: ClusterInfo):
  _, members_df, interlopers_df = load_shiftgap_cone(info, conf.version)
  if not conf.skip_plots and not bool(conf.subset):
    make_plots(
      info=info,
      df_photoz_radial=info.photoz_df,
      df_specz_radial=info.specz_df,
      df_all_radial=info.compilation_df,
      df_members=members_df,
      df_interlopers=interlopers_df,
      df_legacy_radial=info.legacy_df,
      photoz_odds=conf.photoz_odds,
      separated=conf.separated_plots,
      overwrite=conf.overwrite,
      splus_only=conf.splus_only_plots,
    )


@dg.op
def op_create_zoffset_table(conf: ConfigResource):
  df_clusters = load_catalog(version=conf.version, subset=conf.subset)
  create_zoffset_table(
    df_clusters=df_clusters,
    z_delta=conf.z_spec_delta,
    version=conf.version,
    overwrite=conf.overwrite,
  )



@dg.op(ins={'start_after': dg.In(dg.Nothing)}, pool='cluster')
def op_build_cluster_page(conf: ConfigResource, info: ClusterInfo):
  if not conf.skip_website and not bool(conf.subset):
    df_clusters = load_catalog(version=conf.version, subset=conf.subset)
    df_clusters_prev = load_catalog(version=conf.version - 1, subset=conf.subset)
    df_comp = info.compilation_df
    _, df_members, _ = load_shiftgap_cone(info=info, version=conf.version)
    download_xray(
      info=info, 
      overwrite=conf.overwrite, 
      fmt=conf.plot_format
    )
    copy_xray(
      info=info,
      overwrite=conf.overwrite
    )
    build_cluster_page(
      info=info,
      version=conf.version,
      df_compilation_radial=df_comp,
      df_members=df_members,
      df_clusters_prev=df_clusters_prev,
      df_clusters=df_clusters,
    )
    # op_build_other_pages
    make_index(
      df_clusters=df_clusters,
      df_clusters_prev=df_clusters_prev,
      version=conf.version,
    )
    make_zoffset_page(
      df_clusters=df_clusters,
      df_clusters_prev=df_clusters_prev,
      version=conf.version,
    )


@dg.op(pool='cluster')
def op_build_other_pages(
  conf: ConfigResource,
  df_clusters: pd.DataFrame,
  df_clusters_prev: pd.DataFrame,
):
  if not conf.skip_website and not bool(conf.subset):
    make_index(
      df_clusters=df_clusters,
      df_clusters_prev=df_clusters_prev,
      version=conf.version,
    )
    make_zoffset_page(
      df_clusters=df_clusters,
      df_clusters_prev=df_clusters_prev,
      version=conf.version,
    )


@dg.graph
def cluster_pipeline(cls_name: str):
  info = op_compute_cluster_info(cls_name=cls_name)
  
  df_specz = op_specz_cone(info=info)
  df_specz_outrange = op_specz_cone_outrange(info)
  df_photoz = op_photoz_cone(info)
  df_legacy = op_legacy_cone(info)
  
  df_all = op_compile_cluster_catalog(
    info=info, 
    specz_cone=df_specz, 
    photoz_cone=df_photoz, 
    legacy_cone=df_legacy, 
    specz_outrange_cone=df_specz_outrange,
  )
  
  x = op_render_plots(info=info, start_after=df_all)
  
  op_build_cluster_page(start_after=x, info=info)



@dg.graph
def cluster_website_pipeline(cls_name: str):
  info = op_compute_cluster_info(cls_name=cls_name)
  x = op_render_plots(info=info, start_after=info)
  op_build_cluster_page(start_after=x, info=info)



@dg.op(out=dg.DynamicOut(str))
def op_get_all_cluster_names(conf: ConfigResource):
  df_clusters = load_catalog(version=conf.version, subset=conf.subset)
  rep_map = {'+': 'p', '-': 'm', '.': '_', '[': '_', ']': '_'}
  df_clusters['key'] = df_clusters['name']
  for k, v in rep_map.items():
    df_clusters['key'] = df_clusters['key'].str.replace(k, v)
      
  for i, cluster in df_clusters.iterrows():
    yield dg.DynamicOutput(
      cluster['name'], 
      mapping_key=f'{cluster["clsid"]}_{cluster["key"]}'
    )



@dg.job(resource_defs={'conf': ConfigResource()})
def scale_pipeline():
  op_create_zoffset_table()
  op_get_all_cluster_names().map(cluster_pipeline)
  # op_build_other_pages(df_clusters, df_clusters_prev)


@dg.job(resource_defs={'conf': ConfigResource()})
def scale_website_pipeline():
  op_create_zoffset_table()
  op_get_all_cluster_names().map(cluster_website_pipeline)



defs = dg.Definitions(
  jobs=[scale_pipeline, scale_website_pipeline],
  resources={'conf': ConfigResource()},
)