import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from multiprocessing import Lock
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dask import config as dask_config
from dask.distributed import Client
from pylegs.io import read_table, write_table


def config_dask():
  client = Client(n_workers=12, memory_limit='64GB')
  dask_config.set({'distributed.scheduler.worker-ttl': None})
  return client


class Timming:
  def __init__(self, start: bool = True):
    self.start_time = None
    self.end_time = None
    if start:
      self.start()


  def __repr__(self) -> str:
    return self.duration()


  def start(self):
    self.start_time = datetime.now()


  def end(self) -> str:
    self.end_time = datetime.now()
    return self.duration()


  def duration(self) -> str:
    if not self.end_time:
      duration = self.end_time - self.start_time
    else:
      end_time = datetime.now()
      duration = end_time - self.start_time

    return self._format_time(duration)


  def _format_time(self, dt: timedelta) -> str:
    hours, remainder = divmod(dt.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))



class SingletonMeta(type):
  """
  Thread-safe implementation of Singleton.
  """
  _instances = {}
  """The dict storing memoized instances"""

  _lock = Lock()
  """
  Lock object that will be used to synchronize threads during
  first access to the Singleton.
  """

  def __call__(cls, *args, **kwargs):
    """
    Possible changes to the value of the `__init__` argument do not affect
    the returned instance.
    """
    # When the program has just been launched. Since there's no
    # Singleton instance yet, multiple threads can simultaneously pass the
    # previous conditional and reach this point almost at the same time. The
    # first of them will acquire lock and will proceed further, while the
    # rest will wait here.
    with cls._lock:
      # The first thread to acquire the lock, reaches this conditional,
      # goes inside and creates the Singleton instance. Once it leaves the
      # lock block, a thread that might have been waiting for the lock
      # release may then enter this section. But since the Singleton field
      # is already initialized, the thread won't create a new object.
      if cls not in cls._instances:
        instance = super().__call__(*args, **kwargs)
        cls._instances[cls] = instance
    return cls._instances[cls]


def rmse(a1, a2):
  return np.linalg.norm(a1 - a2) / np.sqrt(len(a1))


def relative_err(actual, expected):
  return (actual - expected) / expected


def compute_pdf_peak(a, binrange = None, binwidth = None):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax = sns.histplot(x=a, binrange=binrange, binwidth=binwidth, kde=True)
  x, y = ax.get_lines()[0].get_data()
  x_max, y_max = x[np.argmax(y)], np.max(y)
  plt.close()
  return x_max, y_max



def return_table_if_exists(path: Path, default: pd.DataFrame = None):
  df = None
  
  if default is not None:
    df = default
  elif path.exists():
    df = read_table(path)
  
  if path is not None:
    print(f'Table {str(path)}')
  
  if df is not None:
    print('Number of objects:', len(df))
    print('Columns')
    pprint(df.columns.tolist())
  return df



class SkipException(Exception):
  pass


class cond_overwrite(object):
  def __init__(
    self, 
    path: str | Path, 
    overwrite: bool = False, 
    mkdir: bool = False,
    time: bool = False,
    template: str = 'Elapsed time: {}',
  ):
    self.path = Path(path)
    self.overwrite = overwrite
    self.mkdir = mkdir
    self.time = time
    self.timer = None
    self.template = template
    
  def __enter__(self):
    if self.path.exists() and not self.overwrite:
      sys.settrace(lambda *args, **keys: None)
      frame = sys._getframe(1)
      frame.f_trace = self.trace
    else:
      if self.mkdir:
        self.path.parent.mkdir(parents=True, exist_ok=True)
      if self.time:
        self.timer = Timming()
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    if exc_type is None:
      return
    if issubclass(exc_type, SkipException):
      return True
    if self.time:
      print(self.template.format(self.timer.end()))
    return False

  def trace(self, *args, **kwargs):
    raise SkipException()
  
  def write_table(self, table):
    write_table(table, self.path)




if __name__ == '__main__':
  with cond_overwrite(Path(__file__), overwrite=False):
    print('teste')