import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from astromodule.io import merge_pdf, write_table
from astromodule.pipeline import Pipeline, PipelineStorage

from splusclusters.configs import configs
from splusclusters.external import DownloadLegacyCatalogStage
from splusclusters.loaders import (LoadAllRadialStage, LoadLegacyRadialStage,
                                   LoadPauloInfoStage, LoadPhotozRadialStage,
                                   LoadSpeczRadialStage,
                                   PrepareCatalogToSubmitStage,
                                   load_catalog_v6, load_photoz2, load_spec)
from splusclusters.match import (PhotoZRadialSearchStage,
                                 PhotozSpeczLegacyMatchStage,
                                 SpecZRadialSearchStage)
from splusclusters.plots import ClusterPlotStage


def clusters_v6_pipeline(clear: bool = False):
  df_clusters = load_catalog_v6()
  df_photoz, photoz_skycoord = load_photoz2()
  df_spec, specz_skycoord = load_spec()
  
  configs.Z_SPEC_DELTA = 0.02
  
  if clear:
    for p in configs.LEG_PHOTO_FOLDER.glob('*.parquet'):
      if p.stat().st_size < 650:
        p.unlink()
        
  ls10_pipe = Pipeline(
    LoadPauloInfoStage(df_clusters),
    PhotoZRadialSearchStage(overwrite=True),
    SpecZRadialSearchStage(overwrite=True),
    DownloadLegacyCatalogStage('cls_15Mpc_deg', overwrite=True, workers=5),
    LoadPhotozRadialStage(),
    LoadSpeczRadialStage(),
    LoadLegacyRadialStage(),
    PhotozSpeczLegacyMatchStage(overwrite=True),
    LoadAllRadialStage(),
    ClusterPlotStage(overwrite=True, splus_only=False),
    PrepareCatalogToSubmitStage(overwrite=True),
  )
  
  PipelineStorage().write('df_photoz', df_photoz)
  PipelineStorage().write('photoz_skycoord', photoz_skycoord)
  PipelineStorage().write('df_spec', df_spec)
  PipelineStorage().write('specz_skycoord', specz_skycoord)
  
  ls10_pipe.map_run('cls_id', df_clusters.clsid.values, workers=1)
  
  plot_paths = [configs.PLOTS_FOLDER / f'cls_{c}.pdf' for c in df_clusters.name.values]
  plot_paths = [p for p in plot_paths if p.exists()]
  concat_plot_path = configs.PLOTS_FOLDER / 'clusters_v6+review.pdf'
  merge_pdf(plot_paths, concat_plot_path)
  
  write_table(
    df_clusters[['clsid', 'name', 'RA', 'DEC', 'zspec']], 
    configs.SUBMIT_FOLDER / 'index.dat'
  )


if __name__ == "__main__":
  clusters_v6_pipeline()