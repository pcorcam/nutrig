import numpy as np
import os
import glob

bkg_pulse_dir = '/sps/grand/pcorrea/nutrig/database/bkg/gp13_pretrigger_pulses_th1_35_th2_25'

bkg_pulse_files = sorted( glob.glob( os.path.join(bkg_pulse_dir,'*.npz') ) )

n_counts = 0

for i, file in enumerate(bkg_pulse_files):
    print(f'Counting pulses in file {i+1}/{len(bkg_pulse_files)}')
    with np.load(file) as f:
        n_counts += len(f['traces'])

print(n_counts)
