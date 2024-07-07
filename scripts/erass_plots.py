import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.io import merge_pdf
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.configs import configs
from splusclusters.loaders import (LoadERASSInfoStage, load_eRASS, load_photoz,
                                   load_spec)
from splusclusters.match import PhotoZRadialSearchStage, SpecZRadialSearchStage
from splusclusters.plots import ClusterPlotStage


def erass_plots_pipeline():
  df_clusters = load_eRASS()
  df_photoz, photoz_skycoord = load_photoz()
  df_spec, specz_skycoord = load_spec()
  
  pipe = Pipeline(
    LoadERASSInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=False),
    SpecZRadialSearchStage(overwrite=False),
    ClusterPlotStage(overwrite=False)
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  pipe.map_run('cls_name', df_clusters.Cluster.values, workers=1)
  # df2 = selfmatch(df_clusters, 1*u.deg, 'identify', ra='RA_OPT', dec='DEC_OPT')
  # df2 = df2.sort_values('GroupID')
  df2 = df_clusters.sort_values('RA_OPT')
  plot_paths = [configs.PLOTS_FOLDER / f'cls_{c}.pdf' for c in df2.Cluster.values]
  concat_plot_path = configs.PLOTS_FOLDER / 'eRASS_v1.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  

if __name__ == '__main__':
  erass_plots_pipeline()