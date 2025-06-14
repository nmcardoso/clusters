from typing import List

import dagster as dg


@dg.op
def op_1() -> List[int]:
  return [1, 2, 3]


@dg.op(out=dg.DynamicOut(int))
def op_2():
  for i, v in enumerate(op_1()):
    yield dg.DynamicOutput(v, str(i))



####


@dg.asset
def op_3(v):
  return v*3


@dg.asset
def op_4(v):
  return v*4


@dg.op
def op_5(v):
  print(v)


@dg.graph(out={'x3': dg.GraphOut(), 'x4': dg.GraphOut()})
def fanin(v):
  r1 = op_3(v)
  r2 = op_4(v)
  return r1, r2


@dg.graph(out={'p1': dg.GraphOut(), 'p2': dg.GraphOut()})
def single_value_pipeline(v):
  r1, r2 = fanin(v)
  return op_5(r1), op_5(r2)


####



@dg.job
def job():
  x3, x4 = op_2().map(fanin)
  x3.map(op_5)
  x4.map(op_5)



defs = dg.Definitions(jobs=[job])