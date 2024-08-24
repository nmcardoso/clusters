from typing import Literal, Sequence, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
from astromodule.pipeline import Pipeline, PipelineStage, PipelineStorage
from astromodule.table import (concat_tables, crossmatch, fast_crossmatch,
                               guess_coords_columns, radial_search, selfmatch)
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.wcs import WCS
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from tqdm import tqdm

from splusclusters.configs import configs


def get_plot_title(
  cls_name: str,
  cls_ra: float,
  cls_dec: float,
  cls_z: float,
  cls_search_radius_deg: float,
  cls_search_radius_Mpc: float,
  z_spec_range: Tuple[float, float],
  z_photo_range: Tuple[float, float],
):
  return (
    f'Cluster: {cls_name} (RA: {cls_ra:.5f}, DEC: {cls_dec:.5f})\n'
    f'Search Radius: {cls_search_radius_Mpc:.2f}Mpc = {cls_search_radius_deg:.3f}$^\\circ$ ($z_{{cluster}}={cls_z:.4f}$)\n'
    f'$z_{{spec}}$: $z_{{cluster}} \\pm {configs.Z_SPEC_DELTA}$ = [{z_spec_range[0]:.4f}, {z_spec_range[1]:.4f}]\n'
    f'$z_{{photo}}$: $z_{{cluster}} \\pm {configs.Z_PHOTO_DELTA}$ = [{z_photo_range[0]:.4f}, {z_photo_range[1]:.4f}]\n'
    f'R Mag Range: [13, 22] $\\cdot$ Spec Class = GALAXY*\n'
  )



class PlotStage(PipelineStage):
  def add_circle(
    self, 
    ra: float, 
    dec: float, 
    radius: float,
    color: str,
    ax,
    label: str = '',
    ls: str = '-',
  ):
    circle = SphericalCircle(
      center=[ra, dec]*u.deg,
      radius=radius*u.deg,
      fc='none', 
      lw=2, 
      linestyle=ls,
      ec=color, 
      transform=ax.get_transform('icrs'), 
      label=label,
    )
    ax.add_patch(circle)
    
  def add_all_circles(
    self,
    cls_ra: float,
    cls_dec: float,
    r200_deg: float,
    r200_Mpc: float,
    r500_deg: float,
    r500_Mpc: float,
    search_radius_deg: float,
    search_radius_Mpc: float,
    ax,
  ):
    if r200_deg:
      self.add_circle(
        ra=cls_ra,
        dec=cls_dec,
        radius=5*r200_deg,
        color='tab:green',
        label=f'5$\\times$R200 ({5*r200_Mpc:.2f}Mpc $\\bullet$ {5*r200_deg:.2f}$^\\circ$)',
        ax=ax
      )
    if r500_deg:
      self.add_circle(
        ra=cls_ra,
        dec=cls_dec,
        radius=5*r500_deg,
        color='tab:green',
        ls='--',
        label=f'5$\\times$R500 ({5*r500_Mpc:.2f}Mpc $\\bullet$ {5*r500_deg:.2f}$^\\circ$)',
        ax=ax
      )
    if search_radius_deg:
      self.add_circle(
        ra=cls_ra,
        dec=cls_dec,
        radius=search_radius_deg,
        color='tab:brown',
        label=f'{search_radius_Mpc:.2f}Mpc ({search_radius_deg:.3f}$^\\circ$)',
        ax=ax
      )
    
  def add_cluster_center(self, ra: float, dec: float, ax):
    ax.scatter(
      ra, 
      dec, 
      marker='+', 
      linewidths=2, 
      s=140, 
      c='tab:red', 
      rasterized=True, 
      transform=ax.get_transform('icrs'),
      label='Cluster Center'
    )



class ClusterPlotStage(PlotStage):
  def __init__(
    self, 
    fmt: Literal['pdf', 'jpg', 'png'] = 'pdf', 
    overwrite: bool = False, 
    separated: bool = False,
    photoz_odds: float = 0.9,
    splus_only: bool = False,
    version: int = 6,
  ):
    self.fmt = fmt
    self.overwrite = overwrite
    self.separated = separated
    self.photoz_odds = photoz_odds
    self.splus_only = splus_only
    self.version = version
    
  def plot_specz(
    self,
    cls_ra: float, 
    cls_dec: float, 
    cls_r200_deg: float, 
    cls_r500_deg: float, 
    cls_r200_Mpc: float, 
    cls_r500_Mpc: float, 
    cls_search_radius_deg: float,
    cls_search_radius_Mpc: float,
    df_specz_radial: pd.DataFrame,
    df_members: pd.DataFrame,
    df_interlopers: pd.DataFrame,
    z_spec_range: Tuple[float, float],
    ax: plt.Axes,
  ):
    df_plot = df_specz_radial[df_specz_radial.z.between(*z_spec_range)]
    ra_col, dec_col = guess_coords_columns(df_plot)
    if df_members is not None and df_interlopers is not None:
      ax.scatter(
        df_plot[ra_col].values, 
        df_plot[dec_col].values, 
        c='tab:red', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'Unclassified'
      )
      ra_col, dec_col = guess_coords_columns(df_members)
      ax.scatter(
        df_members[ra_col].values, 
        df_members[dec_col].values, 
        c='tab:blue', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'Member ({len(df_members)})'
      )
      ax.scatter(
        df_interlopers[ra_col].values, 
        df_interlopers[dec_col].values, 
        c='tab:orange', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'Interloper ({len(df_interlopers)})'
      )
    else:
      ax.scatter(
        df_plot[ra_col].values, 
        df_plot[dec_col].values, 
        c='tab:blue', 
        s=6, 
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'$z_{{spec}}$'
      )
    self.add_cluster_center(cls_ra, cls_dec, ax)
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      search_radius_deg=cls_search_radius_deg,
      search_radius_Mpc=cls_search_radius_Mpc,
      ax=ax
    )
    ax.set_title(f'$z_{{spec}}$ - Objects: {len(df_plot)}')
    ax.invert_xaxis()
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
  
  def plot_photoz(
    self,
    cls_ra: float, 
    cls_dec: float, 
    cls_r200_deg: float, 
    cls_r500_deg: float, 
    cls_r200_Mpc: float, 
    cls_r500_Mpc: float, 
    cls_search_radius_deg: float,
    cls_search_radius_Mpc: float,
    df_photoz_radial: pd.DataFrame,
    z_photo_range: Tuple[float, float],
    ax: plt.Axes,
  ):
    ra_col, dec_col = guess_coords_columns(df_photoz_radial)
    if len(df_photoz_radial) > 0:
      ax.scatter(
        df_photoz_radial[ra_col].values, 
        df_photoz_radial[dec_col].values,
        c='tab:blue', 
        s=2, 
        alpha=0.02 if len(df_photoz_radial) > 200_000 else 0.2,
        rasterized=True, 
        transform=ax.get_transform('icrs'),
        label=f'iDR5 objects'
      )
    self.add_cluster_center(cls_ra, cls_dec, ax)
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      search_radius_deg=cls_search_radius_deg,
      search_radius_Mpc=cls_search_radius_Mpc,
      ax=ax
    )
    ax.set_title(f'S-PLUS Coverage - Objects: {len(df_photoz_radial)}')
    ax.invert_xaxis()
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
  
  def plot_photoz_specz(
    self,
    cls_ra: float, 
    cls_dec: float, 
    cls_r200_deg: float, 
    cls_r500_deg: float, 
    cls_r200_Mpc: float, 
    cls_r500_Mpc: float, 
    cls_search_radius_deg: float,
    cls_search_radius_Mpc: float,
    df_specz_radial: pd.DataFrame,
    df_photoz_radial: pd.DataFrame,
    df_all_radial: pd.DataFrame,
    z_photo_range: Tuple[float, float],
    z_spec_range: Tuple[float, float],
    ax: plt.Axes,
  ):
    if len(df_specz_radial) > 0 and len(df_photoz_radial) > 0:
      df_photoz_good = df_all_radial[df_all_radial.zml.between(*z_photo_range) & (df_all_radial.odds > self.photoz_odds)]
      df_photoz_good_with_spec = df_photoz_good[df_photoz_good.z.between(*z_spec_range)]
      df_photoz_good_wo_spec = df_photoz_good[~df_photoz_good.z.between(*z_spec_range) | df_photoz_good.z.isna()]
      df_photoz_bad = df_all_radial[~df_all_radial.zml.between(*z_photo_range) & (df_all_radial.odds > self.photoz_odds)]
      df_photoz_bad_with_spec = df_photoz_bad[df_photoz_bad.z.between(*z_spec_range)]
      if len(df_photoz_good_wo_spec) > 0:
        ra_col, dec_col = guess_coords_columns(df_photoz_good_wo_spec)
        ax.scatter(
          df_photoz_good_wo_spec[ra_col].values, 
          df_photoz_good_wo_spec[dec_col].values, 
          c='tab:olive', 
          s=6, 
          rasterized=True, 
          transform=ax.get_transform('icrs'),
          label=f'good $z_{{photo}}$ wo/ $z_{{spec}}$ ({len(df_photoz_good_wo_spec)} obj)'
        )
      if len(df_photoz_bad_with_spec) > 0:
        ra_col, dec_col = guess_coords_columns(df_photoz_bad_with_spec)
        ax.scatter(
          df_photoz_bad_with_spec[ra_col].values, 
          df_photoz_bad_with_spec[dec_col].values, 
          c='tab:orange', 
          s=6, 
          rasterized=True, 
          transform=ax.get_transform('icrs'),
          label=f'False Negatives ({len(df_photoz_bad_with_spec)} obj)'
        )
      if len(df_photoz_good_with_spec) > 0:
        ra_col, dec_col = guess_coords_columns(df_photoz_good_with_spec)
        ax.scatter(
          df_photoz_good_with_spec[ra_col].values, 
          df_photoz_good_with_spec[dec_col].values, 
          c='tab:blue', 
          s=6, 
          rasterized=True, 
          transform=ax.get_transform('icrs'),
          label=f'True Positives ({len(df_photoz_good_with_spec)} obj)'
        )
    self.add_cluster_center(cls_ra, cls_dec, ax)
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      search_radius_deg=cls_search_radius_deg,
      search_radius_Mpc=cls_search_radius_Mpc,
      ax=ax
    )
    ax.set_title(f'$z_{{photo}}$ $\\cap$ $z_{{spec}}$ (xmatch distance: 1 arcsec, odds > {self.photoz_odds})')
    ax.invert_xaxis()
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    
  def run(
    self, 
    cls_name: str,
    cls_ra: float, 
    cls_dec: float, 
    cls_z: float,
    cls_r200_deg: float, 
    cls_r500_deg: float, 
    cls_r200_Mpc: float, 
    cls_r500_Mpc: float, 
    cls_search_radius_deg: float,
    cls_search_radius_Mpc: float,
    z_photo_range: Tuple[float, float],
    z_spec_range: Tuple[float, float],
    df_photoz_radial: pd.DataFrame,
    df_specz_radial: pd.DataFrame,
    df_all_radial: pd.DataFrame,
    df_members: pd.DataFrame,
    df_interlopers: pd.DataFrame,
  ):
    wcs_spec =  {
      # 'CDELT1': -1.0,
      # 'CDELT2': 1.0,
      # 'CRPIX1': 8.5,
      # 'CRPIX2': 8.5,
      'CRVAL1': cls_ra,
      'CRVAL2': cls_dec,
      'CTYPE1': 'RA---AIT',
      'CTYPE2': 'DEC--AIT',
      'CUNIT1': 'deg',
      'CUNIT2': 'deg'
    }
    wcs = WCS(wcs_spec)
    
    title = get_plot_title(
        cls_name=cls_name,
        cls_ra=cls_ra,
        cls_dec=cls_dec,
        cls_z=cls_z,
        cls_search_radius_deg=cls_search_radius_deg,
        cls_search_radius_Mpc=cls_search_radius_Mpc,
        z_spec_range=z_spec_range,
        z_photo_range=z_photo_range,
      )
    
    if self.separated:
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'specz.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot(projection=wcs)
        self.plot_specz(
          cls_ra=cls_ra, 
          cls_dec=cls_dec,
          cls_r200_deg=cls_r200_deg, 
          cls_r500_deg=cls_r500_deg, 
          cls_r200_Mpc=cls_r200_Mpc, 
          cls_r500_Mpc=cls_r500_Mpc, 
          cls_search_radius_deg=cls_search_radius_deg,
          cls_search_radius_Mpc=cls_search_radius_Mpc,
          z_spec_range=z_spec_range,
          df_members=df_members,
          df_interlopers=df_interlopers,
          df_specz_radial=df_specz_radial,
          ax=ax,
        )
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'photoz.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot(projection=wcs)
        self.plot_photoz(
          cls_ra=cls_ra, 
          cls_dec=cls_dec,
          cls_r200_deg=cls_r200_deg, 
          cls_r500_deg=cls_r500_deg, 
          cls_r200_Mpc=cls_r200_Mpc, 
          cls_r500_Mpc=cls_r500_Mpc, 
          cls_search_radius_deg=cls_search_radius_deg,
          cls_search_radius_Mpc=cls_search_radius_Mpc,
          df_photoz_radial=df_photoz_radial,
          z_photo_range=z_photo_range,
          ax=ax,
        )
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'photoz_specz.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot(projection=wcs)
        self.plot_photoz_specz(
          cls_ra=cls_ra, 
          cls_dec=cls_dec,
          cls_r200_deg=cls_r200_deg, 
          cls_r500_deg=cls_r500_deg, 
          cls_r200_Mpc=cls_r200_Mpc, 
          cls_r500_Mpc=cls_r500_Mpc, 
          cls_search_radius_deg=cls_search_radius_deg,
          cls_search_radius_Mpc=cls_search_radius_Mpc,
          df_specz_radial=df_specz_radial,
          df_photoz_radial=df_photoz_radial,
          df_all_radial=df_all_radial,
          z_photo_range=z_photo_range,
          z_spec_range=z_spec_range,
          ax=ax,
        )
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
      out_path = configs.PLOTS_FOLDER / f'cls_{cls_name}.{self.fmt}'
      if not self.overwrite and out_path.exists():
        return
      if self.splus_only and len(df_photoz_radial) == 0:
        return
      
      fig, axs = plt.subplots(
        nrows=3, 
        ncols=1, 
        figsize=(12, 27), 
        subplot_kw={'projection': wcs}, 
        dpi=300
      )
      
      self.plot_specz(
        cls_ra=cls_ra, 
        cls_dec=cls_dec,
        cls_r200_deg=cls_r200_deg, 
        cls_r500_deg=cls_r500_deg, 
        cls_r200_Mpc=cls_r200_Mpc, 
        cls_r500_Mpc=cls_r500_Mpc, 
        cls_search_radius_deg=cls_search_radius_deg,
        cls_search_radius_Mpc=cls_search_radius_Mpc,
        df_members=df_members,
        df_interlopers=df_interlopers,
        df_specz_radial=df_specz_radial,
        z_spec_range=z_spec_range,
        ax=axs[0],
      )
      
      self.plot_photoz(
        cls_ra=cls_ra, 
        cls_dec=cls_dec,
        cls_r200_deg=cls_r200_deg, 
        cls_r500_deg=cls_r500_deg, 
        cls_r200_Mpc=cls_r200_Mpc, 
        cls_r500_Mpc=cls_r500_Mpc, 
        cls_search_radius_deg=cls_search_radius_deg,
        cls_search_radius_Mpc=cls_search_radius_Mpc,
        df_photoz_radial=df_photoz_radial,
        z_photo_range=z_photo_range,
        ax=axs[1],
      )
      
      self.plot_photoz_specz(
        cls_ra=cls_ra, 
        cls_dec=cls_dec,
        cls_r200_deg=cls_r200_deg, 
        cls_r500_deg=cls_r500_deg, 
        cls_r200_Mpc=cls_r200_Mpc, 
        cls_r500_Mpc=cls_r500_Mpc, 
        cls_search_radius_deg=cls_search_radius_deg,
        cls_search_radius_Mpc=cls_search_radius_Mpc,
        df_specz_radial=df_specz_radial,
        df_photoz_radial=df_photoz_radial,
        df_all_radial=df_all_radial,
        z_photo_range=z_photo_range,
        z_spec_range=z_spec_range,
        ax=axs[2],
      )
      
      fig.suptitle(title, size=18)
      plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)




class ContourPlotStage(PlotStage):
  def specz_plot(self):
    pass
  
  def run(
    self, 
    cls_ra: float,
    cls_dec: float,
    df_members: pd.DataFrame,
  ):
    pass
    


class SpecDiffPlotStage(PlotStage):
  def __init__(self, overwrite: bool = False, separated: bool = True, fmt: str = 'jpg', version: int = 6):
    self.overwrite = overwrite
    self.separated = separated
    self.fmt = fmt
    self.version = version
    
  def diagonal_plot(
    self, 
    df_members: pd.DataFrame, 
    df_interlopers: pd.DataFrame,
    df_all_radial: pd.DataFrame,
    ax: plt.Axes,
  ):
    ax.plot([0, 1], [0, 1], c='tab:gray', alpha=0.75, ls='--', transform=ax.transAxes)
    if df_members is not None and df_interlopers is not None:
      members_match = fast_crossmatch(df_members, df_all_radial)
      ax.scatter(members_match.z, members_match.zml, c='tab:red', s=5, alpha=0.85, label='Members', rasterized=True)
      interlopers_match = fast_crossmatch(df_interlopers, df_all_radial)
      ax.scatter(interlopers_match.z, interlopers_match.zml, c='tab:blue', s=5, alpha=0.85, label='Interlopers', rasterized=True)
    else:
      ax.scatter(df_all_radial.z, df_all_radial.zml, c='tab:blue', s=5, alpha=0.85, label='Objects', rasterized=True)
    ax.legend()
    ax.tick_params(direction='in')
    ax.set_xlabel('$z_{{spec}}$')
    ax.set_ylabel('$z_{{photo}}$')
    ax.set_title('$z_{{spec}}$ x $z_{{photo}}$')
    
  def histogram_plot(
    self, 
    df_members: pd.DataFrame, 
    df_interlopers: pd.DataFrame, 
    df_all_radial: pd.DataFrame, 
    ax: plt.Axes
  ):
    members_match = fast_crossmatch(df_members, df_all_radial)
    ax.hist(members_match.z, histtype='step', density=True, color='tab:red', alpha=0.75, ls='-', lw=2, label='Members z_{{spec}}')
    ax.hist(members_match.zml, histtype='step', density=True, color='tab:red', alpha=0.75, ls='--', lw=2, label='Members z_{{photo}}')
    interlopers_match = fast_crossmatch(df_interlopers, df_all_radial)
    ax.hist(interlopers_match.z, histtype='step', density=True, color='tab:blue', alpha=0.75, ls='-', lw=2, label='Interlopers z_{{spec}}')
    ax.hist(interlopers_match.zml, histtype='step', density=True, color='tab:blue', alpha=0.75, ls='--', lw=2, label='Interlopers z_{{photo}}')
    ax.legend()
    ax.tick_params(direction='in')
    ax.set_xlabel('z')
    ax.set_ylabel('Count (%)')
    ax.set_title('$z_{{spec}}$ x $z_{{photo}}$ distributions')
    
  def histogram_plot_all(
    self, 
    df_all_radial: pd.DataFrame, 
    ax: plt.Axes
  ):
    ax.hist(df_all_radial.z, histtype='step', density=True, color='tab:blue', alpha=0.75, lw=2, label='z_{{spec}}')
    ax.hist(df_all_radial.zml, histtype='step', density=True, color='tab:red', alpha=0.75, lw=2, label='z_{{photo}}')
    ax.legend()
    ax.tick_params(direction='in')
    ax.set_xlabel('z')
    ax.set_ylabel('Count (%)')
    ax.set_title('$z_{{spec}}$ x $z_{{photo}}$ distributions')
  
  def run(
    self, 
    cls_name: str,
    df_members: pd.DataFrame,
    df_interlopers: pd.DataFrame,
    df_all_radial: pd.DataFrame,
  ):
    out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'diagonal.{self.fmt}'
    fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
    ax = fig.add_subplot()
    self.diagonal_plot(
      df_members=df_members,
      df_interlopers=df_interlopers,
      df_all_radial=df_all_radial,
      ax=ax,
    )
    plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    if df_members is not None and df_interlopers is not None:
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'redshift_histogram.{self.fmt}'
      fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
      ax = fig.add_subplot()
      self.histogram_plot(
        df_members=df_members,
        df_interlopers=df_interlopers,
        df_all_radial=df_all_radial,
        ax=ax,
      )
      plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)
    
    out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'redshift_histogram_all.{self.fmt}'
    fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
    ax = fig.add_subplot()
    self.histogram_plot_all(df_all_radial=df_all_radial, ax=ax)
    plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    



class VelocityPlotStage(PlotStage):
  def __init__(
    self, 
    overwrite: bool = False, 
    fmt: Literal['pdf', 'jpg', 'png'] = 'pdf',
    separated: bool = False,
    photoz_odds: float = 0.9,
    version: int = 6,
  ):
    self.overwrite = overwrite
    self.separated = separated
    self.fmt = fmt
    self.photoz_odds = photoz_odds
    self.version = version
    
  
  def plot_velocity(self, df_members: pd.DataFrame, df_interlopers: pd.DataFrame, ax: plt.Axes):
    ax.scatter(df_members.radius_Mpc, df_members.v_offset, c='tab:red', s=5, label='Members', rasterized=True)  
    ax.scatter(df_interlopers.radius_Mpc, df_interlopers.v_offset, c='tab:blue', s=5, label='Interlopers', rasterized=True)
    ax.legend()
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('$\\Delta v [km/s]$')
    ax.set_title('Spectroscoptic velocity x distance')
    
  def plot_specz(self, df_members: pd.DataFrame, df_interlopers: pd.DataFrame, cls_z: float, ax: plt.Axes):
    df_members['z_offset'] = df_members['z'] - cls_z
    df_interlopers['z_offset'] = df_interlopers['z'] - cls_z
    ax.scatter(df_members.radius_Mpc, df_members.z_offset, c='tab:red', s=5, label='Members', rasterized=True)  
    ax.scatter(df_interlopers.radius_Mpc, df_interlopers.z_offset, c='tab:blue', s=5, label='Interlopers', rasterized=True)
    ax.legend()
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('$\\Delta z_{{spec}}$')
    ax.set_title('Spectroscoptic redshift x distance')
    ax.set_ylim(-0.03, 0.03)
    
  def plot_photoz(self, df_members: pd.DataFrame, df_interlopers: pd.DataFrame, df_photoz_radial: pd.DataFrame, cls_z: float, ax: plt.Axes):
    if len(df_photoz_radial) > 0:
      df_members_match = fast_crossmatch(df_members, df_photoz_radial)
      df_interlopers_match = fast_crossmatch(df_interlopers, df_photoz_radial)
      df_members_match['zml_offset'] = df_members_match['zml'] - cls_z
      df_interlopers_match['zml_offset'] = df_interlopers_match['zml'] - cls_z
      df_members_match2 = df_members_match[df_members_match['odds'] > self.photoz_odds]
      df_interlopers_match2 = df_interlopers_match[df_interlopers_match['odds'] > self.photoz_odds]
      ax.scatter(df_members_match2.radius_Mpc, df_members_match2.zml_offset, c='tab:red', s=5, label='Members', rasterized=True)  
      ax.scatter(df_interlopers_match2.radius_Mpc, df_interlopers_match2.zml_offset, c='tab:blue', s=5, label='Interlopers', rasterized=True)
    ax.legend()
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('R [Mpc]')
    ax.set_ylabel('$\\Delta z_{{photo}}$')
    ax.set_title(f'Photometric redshift x distance (odds > {self.photoz_odds})')
    ax.set_ylim(-0.03, 0.03)
  
  def plot_ra_dec(
    self, 
    cls_ra: float,
    cls_dec: float,
    cls_r200_deg: float,
    cls_r200_Mpc: float,
    cls_r500_deg: float,
    cls_r500_Mpc: float,
    cls_search_radius_deg: float,
    cls_search_radius_Mpc: float,
    df_members: pd.DataFrame, 
    df_interlopers: pd.DataFrame, 
    ax: plt.Axes
  ):
    self.add_all_circles(
      cls_ra=cls_ra, 
      cls_dec=cls_dec, 
      r200_deg=cls_r200_deg, 
      r200_Mpc=cls_r200_Mpc, 
      r500_deg=cls_r500_deg, 
      r500_Mpc=cls_r500_Mpc, 
      search_radius_deg=cls_search_radius_deg,
      search_radius_Mpc=cls_search_radius_Mpc,
      ax=ax
    )
    ax.scatter(
      df_members.ra, 
      df_members.dec, 
      c='tab:red', 
      s=5,
      label=f'Members ({len(df_members)})', 
      transform=ax.get_transform('icrs'), 
      rasterized=True,
    )
    ax.scatter(
      df_interlopers.ra, 
      df_interlopers.dec, 
      c='tab:blue', 
      s=5,
      label=f'Interlopers ({len(df_interlopers)})', 
      transform=ax.get_transform('icrs'), 
      rasterized=True,
    )
    self.add_cluster_center(cls_ra, cls_dec, ax)
    ax.invert_xaxis()
    ax.set_aspect('equal', adjustable='datalim', anchor='C')
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.legend()
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    ax.set_title('Spatial distribution of spectroscopic members')
    
  
  def plot_ra_dec_relative(
    self, 
    cls_ra: float,
    cls_dec: float,
    cls_r200_deg: float,
    cls_r200_Mpc: float,
    cls_r500_deg: float,
    cls_r500_Mpc: float,
    cls_search_radius_deg: float,
    cls_search_radius_Mpc: float,
    df_members: pd.DataFrame, 
    df_interlopers: pd.DataFrame, 
    ax: plt.Axes
  ):
    circle = Circle(
      (0, 0), 
      5,
      fc='none', 
      lw=2, 
      linestyle='-',
      ec='tab:green',
      label='5$\\times$R200',
    )
    ax.add_patch(circle)
    circle = Circle(
      (0, 0), 
      5*(cls_r500_deg/cls_r200_deg),
      fc='none', 
      lw=2, 
      linestyle='--',
      ec='tab:green',
      label='5$\\times$R500',
    )
    ax.add_patch(circle)
    circle = Circle(
      (0, 0), 
      cls_search_radius_deg/cls_r200_deg,
      fc='none', 
      lw=2, 
      linestyle='-',
      ec='tab:brown',
      label=f'{cls_search_radius_Mpc:.2f}Mpc',
    )
    ax.add_patch(circle)
    ax.scatter(
      (df_members.ra - cls_ra) / cls_r200_deg, 
      (df_members.dec - cls_dec) / cls_r200_deg, 
      c='tab:red', 
      s=5,
      label=f'Members ({len(df_members)})', 
      # transform=ax.get_transform('icrs'), 
      rasterized=True,
    )
    ax.scatter(
      (df_interlopers.ra - cls_ra) / cls_r200_deg, 
      (df_interlopers.dec - cls_dec) / cls_r200_deg, 
      c='tab:blue', 
      s=5,
      label=f'Interlopers ({len(df_interlopers)})', 
      # transform=ax.get_transform('icrs'), 
      rasterized=True,
    )
    ax.scatter(
      0, 0,
      marker='+', 
      linewidths=1.5, 
      s=80, 
      c='k', 
      rasterized=True, 
      label='Cluster Center'
    )
    ax.invert_xaxis()
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim', anchor='C')
    ax.grid('on', color='k', linestyle='--', alpha=.5)
    ax.tick_params(direction='in')
    ax.set_xlabel('$\\Delta$RA/R200')
    ax.set_ylabel('$\\Delta$DEC/R200')
    ax.set_title('Relative spatial distribution of spectroscopic members')
  
  
  def run(
    self, 
    cls_name: str, 
    cls_ra: float, 
    cls_dec: float, 
    cls_z: float,
    cls_search_radius_deg: float,
    cls_search_radius_Mpc: float,
    cls_r200_deg: float,
    cls_r200_Mpc: float,
    cls_r500_deg: float,
    cls_r500_Mpc: float,
    z_photo_range: Tuple[float, float],
    z_spec_range: Tuple[float, float],
    df_members: pd.DataFrame,
    df_interlopers: pd.DataFrame,
    df_photoz_radial: pd.DataFrame,
  ):
    wcs_spec =  {
      # 'CDELT1': -1.0,
      # 'CDELT2': 1.0,
      # 'CRPIX1': 8.5,
      # 'CRPIX2': 8.5,
      'CRVAL1': cls_ra,
      'CRVAL2': cls_dec,
      'CTYPE1': 'RA---AIT',
      'CTYPE2': 'DEC--AIT',
      'CUNIT1': 'deg',
      'CUNIT2': 'deg'
    }
    wcs = WCS(wcs_spec)
    title = get_plot_title(
      cls_name=cls_name,
      cls_ra=cls_ra,
      cls_dec=cls_dec,
      cls_z=cls_z,
      cls_search_radius_deg=cls_search_radius_deg,
      cls_search_radius_Mpc=cls_search_radius_Mpc,
      z_spec_range=z_spec_range,
      z_photo_range=z_photo_range,
    )
      
    if self.separated:
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'spec_velocity.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot()
        self.plot_velocity(df_members, df_interlopers, ax)
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'specz_distance.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot()
        self.plot_specz(df_members, df_interlopers, cls_z, ax)
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'photoz_distance.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot()
        self.plot_photoz(df_members, df_interlopers, df_photoz_radial, cls_z, ax)
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'spec_velocity_position.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot(projection=wcs)
        self.plot_ra_dec(
          cls_ra=cls_ra, 
          cls_dec=cls_dec, 
          cls_r200_deg=cls_r200_deg, 
          cls_r200_Mpc=cls_r200_Mpc, 
          cls_r500_deg=cls_r500_deg, 
          cls_r500_Mpc=cls_r500_Mpc, 
          cls_search_radius_deg=cls_search_radius_deg, 
          cls_search_radius_Mpc=cls_search_radius_Mpc,
          df_members=df_members, 
          df_interlopers=df_interlopers, 
          ax=ax
        )
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'spec_velocity_rel_position.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 7.5), dpi=150)
        ax = fig.add_subplot()
        self.plot_ra_dec_relative(
          cls_ra=cls_ra, 
          cls_dec=cls_dec, 
          cls_r200_deg=cls_r200_deg, 
          cls_r200_Mpc=cls_r200_Mpc, 
          cls_r500_deg=cls_r500_deg, 
          cls_r500_Mpc=cls_r500_Mpc, 
          cls_search_radius_deg=cls_search_radius_deg, 
          cls_search_radius_Mpc=cls_search_radius_Mpc,
          df_members=df_members, 
          df_interlopers=df_interlopers, 
          ax=ax
        )
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
      out_path = configs.VELOCITY_PLOTS_FOLDER / f'cls_{cls_name}.{self.fmt}'
      if not self.overwrite and out_path.exists():
        return
      
      fig = plt.figure(figsize=(7.5, 16), dpi=300)
      ax1 = fig.add_subplot(211)
      ax2 = fig.add_subplot(212, projection=wcs)
      self.plot_velocity(df_members, df_interlopers, ax1)
      self.plot_ra_dec(
          cls_ra=cls_ra, 
          cls_dec=cls_dec, 
          cls_r200_deg=cls_r200_deg, 
          cls_r200_Mpc=cls_r200_Mpc, 
          cls_r500_deg=cls_r500_deg, 
          cls_r500_Mpc=cls_r500_Mpc, 
          cls_search_radius_deg=cls_search_radius_deg, 
          cls_search_radius_Mpc=cls_search_radius_Mpc,
          df_members=df_members, 
          df_interlopers=df_interlopers, 
          ax=ax2
        )
      
      fig.suptitle(title, size=18)
      plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)




class MagDiffPlotStage(PipelineStage):
  def __init__(
    self, 
    overwrite: bool = False, 
    fmt: Literal['pdf', 'jpg', 'png'] = 'pdf', 
    separated: bool = False,
    version: int = 6,
  ):
    self.overwrite = overwrite
    self.separated = separated
    self.fmt = fmt
    self.version = version
    self.white_viridis = LinearSegmentedColormap.from_list(
      'white_viridis', 
      [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
      ], 
      N=256
    )
    
  def plot_mag_diff(
    self,
    df: pd.DataFrame,
    mag1: str, 
    mag2: str, 
    ax: plt.Axes, 
    xlabel: str = None,
    ylabel: str = None,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
  ):
    df_spec = df[~df.z.isna()]
    df_photo = df[df.z.isna()]
    ax.scatter(
      df_photo[mag1], 
      df_photo[mag1] - df_photo[mag2], 
      s=0.8, 
      label=f'Without Spec ({len(df_photo)})', 
      color='tab:blue', 
      alpha=0.8, 
      rasterized=True
    )
    ax.scatter(
      df_spec[mag1], 
      df_spec[mag1] - df_spec[mag2], 
      s=0.8, 
      label=f'With Spec ({len(df_spec)})', 
      color='tab:red', 
      alpha=0.8,
      rasterized=True
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
      ax.set_ylim(*ylim)
    if xlim is not None:
      ax.set_xlim(*xlim)
    ax.grid('on', color='k', linestyle='--', alpha=.2)
    ax.tick_params(direction='in')
    ax.legend()
    ax.set_title('SP-LS r-mag difference')
    
  def plot_histogram(
    self,
    x: np.ndarray | pd.Series | pd.DataFrame,
    ax: plt.Axes,
    xlabel: str = '',
    xrange: Tuple[float, float] = None,
  ):
    ax.hist(x, bins=60, range=xrange)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of galaxies')
    if xrange is not None:
      ax.set_xlim(*xrange)
    ax.grid('on', color='k', linestyle='--', alpha=.2)
    ax.tick_params(direction='in')
    ax.set_title('SP-LS r-mag difference distribution')
  
  def run(
    self, 
    df_all_radial: pd.DataFrame, 
    cls_ra: float, 
    cls_dec: float, 
    cls_name: str, 
    cls_z: float, 
    z_spec_range: Tuple[float, float], 
    z_photo_range: Tuple[float, float], 
    cls_search_radius_deg: float,
    cls_search_radius_Mpc: float,
  ):
    if 'r_auto' not in df_all_radial.columns or 'mag_r' not in df_all_radial:
      return
    
    df = df_all_radial[
      (df_all_radial.type != 'PSF') & 
      df_all_radial.r_auto.between(*configs.MAG_RANGE) & 
      df_all_radial.mag_r.between(*configs.MAG_RANGE) &
      (df_all_radial.z.between(*z_spec_range) | df_all_radial.z.isna()) &
      df_all_radial.zml.between(*z_photo_range)
    ]
    title = get_plot_title(
      cls_name=cls_name,
      cls_ra=cls_ra,
      cls_dec=cls_dec,
      cls_z=cls_z,
      cls_search_radius_deg=cls_search_radius_deg,
      cls_search_radius_Mpc=cls_search_radius_Mpc,
      z_spec_range=z_spec_range,
      z_photo_range=z_photo_range,
    )
    
    if self.separated:
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'mag_diff.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(figsize=(7.5, 7.5), dpi=150)
        self.plot_mag_diff(df, 'r_auto', 'mag_r', axs, '$iDR5_r$', '$iDR5_r - LS10_r$')
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
      
      out = configs.WEBSITE_PATH / f'clusters_v{self.version}' / cls_name / f'mag_diff_hist.{self.fmt}'
      if self.overwrite or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, axs = plt.subplots(figsize=(7.5, 7.5), dpi=150)
        self.plot_histogram(df['r_auto'] - df['mag_r'], axs, '$iDR5_r - LS10_r$')
        plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
      out_path = configs.sMAGDIFF_PLOTS_FOLDER / f'cls_{cls_name}.pdf'
      if not self.overwrite and out_path.exists():
        return

      fig, axs = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(5, 9),
        dpi=300,
        # subplot_kw={'projection': 'scatter_density'}, 
      )
      self.plot_mag_diff(df, 'r_auto', 'mag_r', axs[0], '$iDR5_r$', '$iDR5_r - LS10_r$')
      self.plot_histogram(df['r_auto'] - df['mag_r'], axs[1], '$iDR5_r - LS10_r$')
      plt.suptitle(title, size=10)
      plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
      plt.close(fig)