import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.constants import *
from splusclusters.loaders import (LoadAllRadialStage, LoadERASS2InfoStage,
                                   LoadLegacyRadialStage,
                                   LoadPhotozRadialStage, LoadSpeczRadialStage,
                                   load_eRASS_2, load_photoz2, load_spec)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def erass2_plots_pipeline():
  df_clusters = load_eRASS_2()
  df_photoz, photoz_skycoord = load_photoz2()
  df_spec, specz_skycoord = load_spec()
  df_clusters = df_clusters[(df_clusters.BEST_Z_TYPE != 'photo_z') & (df_clusters.BEST_Z <= 0.1)]
  
  pipe = Pipeline(
    LoadERASS2InfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=False),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=False, splus_only=True),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.NAME.values, workers=4)
  # df2 = selfmatch(df_clusters, 1*u.deg, 'identify', ra='RA_OPT', dec='DEC_OPT')
  # df2 = df2.sort_values('GroupID')
  # df2 = df_clusters.sort_values('DEC_XFIT')
  df2 = df_clusters.sort_values('N_MEMBERS', ascending=False)
  plot_paths = [PLOTS_FOLDER / f'cls_{c}.pdf' for c in df2.NAME.values]
  plot_paths = [p for p in plot_paths if p.exists() and (p.stat().st_size > 100_000)]
  df2 = df2[df2.NAME.isin([p.name for p in plot_paths])]
  # for i, row in df2.iterrows():
  #   len(radial_search(SkyCoord(ra=row.ra, dec=row.dec, unit='deg'), OUT_PATH / 'photoz' / f'{row.name}.parquet'), row.) > 0
  concat_plot_path = PLOTS_FOLDER / 'eRASS_v2+nmembers.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  
  
if __name__ == '__main__':
  erass2_plots_pipeline()