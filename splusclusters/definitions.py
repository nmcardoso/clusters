from typing import List, Optional

import dagster as dg
import pandas as pd
from astropy.coordinates import SkyCoord

from splusclusters._extraction import (download_xray, legacy_cone, photoz_cone,
                                       specz_cone)
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
  out=dg.Out(pd.DataFrame, io_manager_key='in_memory')
)
def op_load_clusters_catalog(conf: ConfigResource):
  return load_catalog(version=conf.version, subset=conf.subset)


@dg.op(
  out=dg.Out(pd.DataFrame, io_manager_key='in_memory')
)
def op_load_previous_clusters_catalog(conf: ConfigResource):
  if conf.version > 6:
    return load_catalog(version=conf.version - 1, subset=conf.subset)
  return None


@dg.op(
  pool='io/intensive',
  out={
    'df_spec_all': dg.Out(pd.DataFrame, io_manager_key='in_memory'), 
    'skycoord_spec_all': dg.Out(SkyCoord, io_manager_key='in_memory'),
  }
)
def op_load_spec():
  return load_spec()


@dg.op(out=dg.Out(ClusterInfo, io_manager_key='in_memory'))
def op_compute_cluster_info(
  conf: ConfigResource,
  df_clusters: pd.DataFrame,
  cls_name: str
) -> ClusterInfo:
  return compute_cluster_info(
    df_clusters=df_clusters, 
    cls_name=cls_name,
    z_spec_delta=conf.z_spec_delta,
    z_photo_delta=conf.z_photo_delta,
  )


@dg.op(out=dg.Out(pd.DataFrame, io_manager_key='in_memory'))
def op_specz_cone(
  conf: ConfigResource, 
  specz_df: pd.DataFrame, 
  specz_skycoord: pd.DataFrame, 
  info: ClusterInfo,
) -> pd.DataFrame:
  return specz_cone(
    specz_df=specz_df, 
    specz_skycoord=specz_skycoord, 
    info=info, 
    overwrite=conf.overwrite,
  )


@dg.op(out=dg.Out(pd.DataFrame, io_manager_key='in_memory'))
def op_specz_cone_outrange(
  conf: ConfigResource, 
  specz_df: pd.DataFrame, 
  specz_skycoord: pd.DataFrame, 
  info: ClusterInfo,
) -> pd.DataFrame:
  return specz_cone(
    specz_df=specz_df, 
    specz_skycoord=specz_skycoord, 
    info=info, 
    overwrite=conf.overwrite,
    in_range=False,
  )


@dg.op(pool='remote/splus', out=dg.Out(pd.DataFrame, io_manager_key='in_memory'))
def op_photoz_cone(conf: ConfigResource, info: ClusterInfo) -> pd.DataFrame:
  return photoz_cone(info=info, overwrite=conf.overwrite)


@dg.op(pool='remote/datalab', out=dg.Out(pd.DataFrame, io_manager_key='in_memory'))
def op_legacy_cone(conf: ConfigResource, info: ClusterInfo) -> pd.DataFrame:
  return legacy_cone(
    info=info, 
    workers=conf.workers, 
    overwrite=conf.overwrite
  )


@dg.op(
  out={
    'df_shiftgap': dg.Out(pd.DataFrame, io_manager_key='in_memory'), 
    'df_members': dg.Out(pd.DataFrame, io_manager_key='in_memory'), 
    'df_interlopers': dg.Out(pd.DataFrame, io_manager_key='in_memory'),
  }
)
def op_shiftgap_cone(conf: ConfigResource, info: ClusterInfo):
  return load_shiftgap_cone(info=info, version=conf.version)


@dg.op(pool='crossmatch/intensive')
def op_compile_cluster_catalog(
  conf: ConfigResource,
  info: ClusterInfo,
  specz_cone_df: pd.DataFrame,
  photoz_cone_df: pd.DataFrame,
  legacy_cone_df: pd.DataFrame,
  shiftgap_df: pd.DataFrame,
  specz_outrange_cone_df: pd.DataFrame,
) -> pd.DataFrame:
  return make_cluster_catalog(
    info=info,
    df_specz_radial=specz_cone_df,
    df_photoz_radial=photoz_cone_df,
    df_legacy_radial=legacy_cone_df,
    df_ret=shiftgap_df,
    df_specz_outrange_radial=specz_outrange_cone_df,
    overwrite=conf.overwrite,
  )


@dg.op
def op_render_plots(
  conf: ConfigResource,
  info: ClusterInfo,
  photoz_cone_df: pd.DataFrame,
  specz_cone_df: pd.DataFrame,
  compiled_cluster_catalog_df: pd.DataFrame,
  members_df: pd.DataFrame,
  interlopers_df: pd.DataFrame,
  legacy_cone_df: pd.DataFrame,
):
  if not conf.skip_plots:
    make_plots(
      info=info,
      df_photoz_radial=photoz_cone_df,
      df_specz_radial=specz_cone_df,
      df_all_radial=compiled_cluster_catalog_df,
      df_members=members_df,
      df_interlopers=interlopers_df,
      df_legacy_radial=legacy_cone_df,
      version=conf.version,
      photoz_odds=conf.photoz_odds,
      separated=conf.separated_plots,
      overwrite=conf.overwrite,
      splus_only=conf.splus_only_plots,
      fmt=conf.plot_format,
    )


@dg.op(ins={'start_after': dg.In(dg.Nothing)})
def op_build_cluster_page(
  conf: ConfigResource, 
  info: ClusterInfo,
  df_photoz: pd.DataFrame,
  df_members: pd.DataFrame,
  df_clusters_prev: pd.DataFrame,
):
  if not conf.skip_website:
    download_xray(
      info=info, 
      overwrite=conf.overwrite, 
      fmt=conf.plot_format
    )
    copy_xray(
      cls_name=info.name, 
      version=conf.version, 
      fmt=conf.plot_format, 
      overwrite=conf.overwrite
    )
    build_cluster_page(
      info=info,
      version=conf.version,
      df_photoz_radial=df_photoz,
      df_members=df_members,
      df_clusters_prev=df_clusters_prev,
    )


@dg.op
def op_build_other_pages(
  conf: ConfigResource,
  df_clusters: pd.DataFrame,
  df_clusters_prev: pd.DataFrame,
):
  if not conf.skip_website:
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
def cluster_pipeline(
  cls_name: str,
  df_clusters: pd.DataFrame,
  df_clusters_prev: pd.DataFrame,
  specz_df: pd.DataFrame,
  specz_coords: SkyCoord | None,
):
  info = op_compute_cluster_info(df_clusters=df_clusters, cls_name=cls_name)
  
  df_specz = op_specz_cone(specz_df, specz_coords, info)
  df_specz_outrange = op_specz_cone_outrange(specz_df, specz_coords, info)
  df_photoz = op_photoz_cone(info)
  df_legacy = op_legacy_cone(info)
  df_shiftgap, df_members, df_interlopers = op_shiftgap_cone(info)
  
  df_all = op_compile_cluster_catalog(
    info=info, 
    specz_cone_df=df_specz,
    photoz_cone_df=df_photoz, 
    legacy_cone_df=df_legacy,
    shiftgap_df=df_shiftgap,
    specz_outrange_cone_df=df_specz_outrange,
  )
  
  x = op_render_plots(
    info=info,
    photoz_cone_df=df_photoz,
    specz_cone_df=df_specz,
    compiled_cluster_catalog_df=df_all,
    members_df=df_members,
    interlopers_df=df_interlopers,
    legacy_cone_df=df_legacy,
  )
  
  op_build_cluster_page(
    start_after=x,
    info=info,
    df_photoz=df_photoz,
    df_members=df_members,
    df_clusters_prev=df_clusters_prev,
  )



@dg.op(out=dg.DynamicOut(str))
def get_all_cluster_names(df_clusters: pd.DataFrame):
  for i, cluster in df_clusters.iterrows():
    yield dg.DynamicOutput(cluster['name'], mapping_key=str(i))




@dg.job(resource_defs={
  'in_memory': dg.InMemoryIOManager(), 
  'conf': ConfigResource(),
})
def dg_make_all():
  df_clusters = op_load_clusters_catalog()
  df_clusters_prev = op_load_previous_clusters_catalog()
  
  specz_df, specz_skycoord = op_load_spec()
  
  clusters = get_all_cluster_names(df_clusters)
  clusters.map(lambda cluster: cluster_pipeline(
    cluster, df_clusters, df_clusters_prev, specz_df, specz_skycoord
  ))
  
  op_build_other_pages(df_clusters, df_clusters_prev)



defs = dg.Definitions(
  jobs=[dg_make_all],
  resources={'conf': ConfigResource()},
)