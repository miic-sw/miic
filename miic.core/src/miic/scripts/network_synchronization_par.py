# -*- coding: utf-8 -*-

import os
from miic.core.miic_utils import create_path

#### Project wide parameters
# lowest level project directory
proj_dir = ''
# directory ofr logging information
log_dir = os.path.join(proj_dir,'log')
# folder for figures
fig_dir = os.path.join(proj_dir,'figures')
# create the folders if not present
create_path(log_dir)
create_path(fig_dir)

#### parameters to for correlation (emperical Green's function creation)
# sub folder where correlations are stored
corr_subdir = 'corr'
corr_res_dir = os.path.join(proj_dir,corr_subdir)


#### parameters for the estimation of time differences
# subfolder for storage of time difference results
dt_subdir = 'time_diffs'
dt_res_dir = os.path.join(proj_dir,dt_subdir)
dt_fig_dir = os.path.join(fig_dir,dt_subdir)

# Plotting
plot_corr_matrix = True
plot_time_shifts = True

### Definition of time calender windows for the time difference measurements
dt_start_date = '2011-12-24 00:00:00.0'   # %Y-%m-%dT%H:%M:%S.%fZ'
dt_end_date = '2011-12-25 00:00:00.0'
dt_win_len = 3600                   # length of window in which EGFs are stacked
dt_date_inc = 3600                        # increment of measurements

### Frequencies
dt_freq_min = 0.01
dt_freq_max = 0.6

### Definition of lapse time window
dt_tw_start = 1     # lapse time of first sample [s]
dt_tw_len = 9       # length of window [s]
