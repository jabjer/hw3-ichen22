import numpy as np
from ligotools import readligo as rl

def test_dq_channel_to_seglist():
    dq = np.array([0, 1, 1, 0])
    segs = rl.dq_channel_to_seglist(dq, fs=1)
    assert len(segs) == 1
    s = segs[0]
    assert s.start == 1 and s.stop == 3

def test_SegmentList():
    segs = rl.SegmentList([[10, 20], [30, 40]])
    assert len(segs.seglist) == 2
    assert tuple(segs[0]) == (10, 20)