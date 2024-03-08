import h5py
import numpy as np
import pandas as pd

with h5py.File("/project/def-sponsor00/earthquake/data/instance/Instance_events_counts.hdf5", 'r') as f:
    print(f['data'])
    print(f['data'].keys())
    print(f['data']['11030611.IV.OFFI..HH'])
    print(f['data']['11030611.IV.OFFI..HH'][2, :15])

a = np.array([0, 1, 2])
print(a**2)