import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import streamlit as st

DATA_URL = 'https://github.com/nmcardoso/clusters/raw/main/public/clusters_spec_all.csv'

global selected_df


@st.cache_data
def load_data() -> pd.DataFrame:
  df = pd.read_csv(DATA_URL)
  df = df[['RA', 'DEC', 'z', 'cluster']]
  return df


def selection_cb_factory(df):
  def selection_callback(trace, points, selector):
    print('callback')
    selected_df = df.loc[points.point_inds]
  return selection_callback


def main():
  df = load_data()
  
  option = st.selectbox('Cluster:', options=df.cluster.unique())
  
  cluster_df = df[df.cluster == option]

  scatter = px.scatter(data_frame=cluster_df, x='RA', y='DEC', color='z', title='Subsample selector', color_continuous_scale='Plasma')
  st.plotly_chart(scatter, use_container_width=True)
  scatter.data[0].on_selection(selection_cb_factory(cluster_df))
  
  # scatter = go.FigureWidget([go.Scatter(y=cluster_df['DEC'], x=cluster_df['RA'], mode='markers')])
  # st.plotly_chart(scatter)
  
  density_plot = ff.create_2d_density(x=cluster_df.RA.values, y=cluster_df.DEC.values, point_size=6)
  st.plotly_chart(density_plot, use_container_width=True)
  
  # table = go.FigureWidget([go.Table(
  #   header=dict(values=cluster_df.columns),
  #   cells=dict(values=[cluster_df[col] for col in cluster_df.columns])
  # )])
  # st.plotly_chart(table, use_container_width=True)
  
  # def selection_fn(trace, points, selector):
  #   print('selc_fn')
  #   table.data[0].cells.values = [cluster_df.loc[points.point_inds][col] for col in cluster_df.columns]

  # scatter.data[0].on_selection(selection_fn)
  # print(scatter.data[0])
  
  # table = ff.create_table(table_text=cluster_df)
  # st.plotly_chart(table, use_container_width=True)
  
  
main()