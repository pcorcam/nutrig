# NUTRIG Database V2

Pablo Correa, 10 April 2025

- Directory of NUTRIG database source files: `/pbs/home/p/pcorrea/nutrig/database/v2`
- Directory of NUTRIG database: `nutrig_db_dir` = `/sps/grand/pcorrea/nutrig/database/v2`

## Main Updates w.r.t. V1

- We use GP80 data instead of GP13 data.
- We use DC2 simulations with the latest RF chain that matches the data instead of applying the RF chain ourselves.

## Background Data

### MD Data

- Data directory: `/sps/grand/data/GrandRoot/gp80/2025/02`
- Run number: 145

We use GP80 MD data taken in 20 Hz mode on 4 February 2025. This is the only MD data with the latest RF chain to which there is **no notch filter** applied on any of the recorded ADC channels. We exclude the following DUs: 1047, 1049, 1054, 1055, 1082, 1086. See `/sps/grand/pcorrea/dc2/noise/gp80/README.md` for more details.

From this data, we make both a selection of static noise traces and background pulses. In particular, the MD data allows us to obtain background pulses with relatively low SNRs.

### UD Data

- Data directory: `/sps/grand/data/GrandRoot/gp80/2025/03`
- Run number: 149

We use GP80 UD data taken on 12 March 2025. This is the only UD data with the latest RF chain to which there is **no notch filter** applied on any of the recorded ADC channels. It only contains data from 3 DUs: 1033, 1044, 1046. We only take data in the first ~5 hour period (12:33:01–17:17:40) where all three DUs are online and have typical trigger rates between 50–200 Hz.

The pulses in this data are reprocessed through the selection procedure described below, so not all online-triggered pulses will pass the offline trigger conditions. In particular, this allows us to obtain a sample of pulses with relatively high SNR, which complements the selection of pulses from MD data.

### Static Noise Traces

- Data directory: `/sps/grand/pcorrea/dc2/noise/gp80`

As part of DC2, we selected data that corresponds to static noise. See `/sps/grand/pcorrea/dc2/noise/gp80/README.md`. In summary, for an event to be tagged as static noise it has to comply with the following criteria:

- Data has to be taken with one of the following DUs: 1011, 1023, 1043, 1045, 1077, 1081;
- $\rm RMS < 20~ ADC$ in all channels;
- $\rm max(amplitude) < 5 \times RMS$ in all channels.

## Simulations

- Simulation directory: `/sps/grand/DC2RF2Alpha`
- Software: ZHAireS (can be extended to CoREAS as well)

Generally, we use the simulation files in the subdirectories with the `ZHAireS-AN` (added noise) extension. Here, ZHAireS-simulated electric fields are propagated through the antenna and RF chain using GRANDlib, and subsequently superimposed with a random static noise trace as described above.

## Selection Procedure

We apply the following selection procedure for all data.

<!-- 1. Apply notch filters to the traces that mimic those on the FEB.

    | Frequency [MHz] | Pole radius |
    |-----------------|-------------|
    | 39              | 0.9         |
    | 119.2           | 0.98        |
    | 119.4           | 0.98        |
    | 121.5           | 0.96        |
    | 124.7           | 0.96        |
    | 132             | 0.95        |
    | 134.2           | 0.96        |
    | 137.8           | 0.98        | -->

1. Apply a band-pass filter with cutoff frequency at 115 MHz. This mimics the DIRECT form FIR filter applied on the GP300 firmware since April 2025. The coefficients of the online filter, provided by Xing Xu, are stored in `/sps/grand/pcorrea/nutrig/database/v2/lowpass115MHz.txt`. These coefficients are used for the offline filtering.

2. Apply the FLT-0 algorithm to select pulses. Most of them will be fixed, but there is always room to play with all the parameters if required.

    | FLT-0 parameter       | Units      | Value        |
    |-----------------------|------------|--------------|
    | $T_1$                 | ADC counts | **variable** |
    | $T_2$                 | ADC counts | **variable** |
    | $T_{\mathrm{quiet}}$  | ns         | 500          |
    | $T_{\mathrm{period}}$ | ns         | 1000         |
    | $T_{\mathrm{sepmax}}$ | ns         | 200          |
    | $\rm NC_{min}$        | ---        | 2            |
    | $\rm NC_{max}$        | ---        | 10           |

    - In order to mitigate boundary artifacts caused by the notch filters (typically random spikes), we ignore the first 100 samples (200 ns) in the evaluation of the FLT-0.

    - A pulse is selected if an FLT-0 is triggered on **X OR Y**. We do **not** trigger on the Z channel.

## Database Structure

```text
nutrig_db_dir
├── bkg/
│   ├── bkg_data_dir_1/
│   │   ├── filtered/
│   │   │   ├── FILTERED_bkg_file_1.npz
│   │   │   ├── FILTERED_bkg_file_2.npz
│   │   │   └── ...
│   │   ├── raw/
│   │   │   ├── RAW_bkg_file_1.npz
│   │   │   ├── RAW_bkg_file_2.npz
│   │   │   └── ...
│   │   └── metadata.npz
│   ├── bkg_data_dir_2/
│   └── ...
├── sig/
│   ├── sig_data_dir_1/
│   │   ├── filtered/
│   │   │   ├── FILTERED_sig_file_1.npz
│   │   │   ├── FILTERED_sig_file_2.npz
│   │   │   └── ...
│   │   ├── raw/
│   │   │   ├── RAW_sig_file_1.npz
│   │   │   ├── RAW_sig_file_2.npz
│   │   │   └── ...
│   │   └── metadata.npz
│   ├── sig_data_dir_2/
│   └── ...
└── README.md
```

### Background

Only background events that pass the FLT-0 are stored. We save both filtered traces, which we pass on to the FLT-1, and raw traces for completeness. In essence, this is a "UD event selection" from MD data.

- Every `bkg_data_dir` is named according to the chosen FLT-0 parameters: `GP80_RUN_**_CH_**_MODE_**_TH1_**_TH2_**_TQUIET_**_TPER_**_TSEPMAX_**_NCMIN_**_NCMAX_**/`

- Each `bkg_data_dir` contains a `metadata.npz` file with information about the data, the FLT-0,...

    **CONTENTS OF BACKGROUND `metadata.npz`**

    | Field                 | Type    | Description                                                                                     |
    |-----------------------|---------|-------------------------------------------------------------------------------------------------|
    | `dict_trigger_params` | `dict`  | Dictionary of FLT-0 parameters. See description above                                           |
    | `root_data_dir`       | `str`   | Directory of the original data files in GrandRoot format                                        |
    | `run_number`          | `int`   | Run number of the data used for the selection                                                   |
    | `du_ids_exclude`      | `list`  | DU IDs that are excluded from the selection                                                     |
    | `samples_from_edge`   | `int`   | Number of samples to exclude from the start of the trace for the FLT-0 evaluation               |
    | `t_eff`               | `float` | Effective time per trace scanned for FLT-0 triggers. For pulses from UD data, `t_eff = -1.` Units: ns |
    | `channel_pol`         | `dict`  | Map of trace channel to antenna arm / polarization                                              |
    | `channels_flt0`       | `list`  | Channels used for the evaluation of FLT-0                                                       |
    | `mode_flt0`           | `list`  | Mode used to trigger FLT-0. Options are `OR` (one channel enough) `AND` (all channels required) |

- Every `FILTERED_bkg_file` mirrors the original GP80 data file name, for example: `FILTERED_GP80_20250204_141251_RUN145_MD_RAW-ChanXYZ-20dB-GP43-20hz-0001.npz`

    **CONTENTS OF `FILTERED_bkg_file.npz`**

    Normally `N_channels = 3`, `N_samples = 1024`

    | Field                 | Type                | Array shape                        | Description                                       |
    |-----------------------|---------------------|------------------------------------|---------------------------------------------------|
    | `traces`              | `np.ndarray[int]`   | `(N_entries,N_channels,N_samples)` | Filtered ADC traces                               |
    | `snr`                 | `np.ndarray[float]` | `(N_entries)`                      | SNRs of filtered ADC traces                       |
    | `t_pulse`             | `np.ndarray[float]` | `(N_entries,N_channels)`           | Sample/index of pulse maximum. `-1` if no FLT-0.  |
    | `du_ids`              | `np.ndarray[int]`   | `(N_entries)`                      | DU IDs                                            |
    | `du_seconds`          | `np.ndarray[int]`   | `(N_entries)`                      | Timestamp up to second precision                  |
    | `du_nanoseconds`      | `np.ndarray[int]`   | `(N_entries)`                      | Timestamp decimals up to nanosecond precision     |
    | `FLT0_flags`          | `np.ndarray[bool]`  | `(N_entries,N_channels)`           | Flag of the FLT-0, per channel                    |
    | `FLT0_first_T1_idcs`  | `np.ndarray[int]`   | `(N_entries,N_channels)`           | Position of the first T1 threshold in the FLT-0 evaluation, per channel. If no trigger, the value is `-1`|
    | `n_FLT0`              | `np.ndarray[int]`   | `(N_entries,N_channels)`           | Number of FLT-0 triggers in a trace, per channel  |
    | `trigger_rate_per_ch` | `np.ndarray[float]` | `(N_DU,N_channels)`                | Mean trigger rate per DU, per channel. Units: Hz  |
    | `trigger_rate_OR`     | `np.ndarray[float]` | `(N_DU)`                           | Mean trigger rate per DU applying an `OR` logic. Units: Hz |
    | `trigger_rate_AND`    | `np.ndarray[float]` | `(N_DU)`                           | Mean trigger rate per DU applying an `AND` logic. Units: Hz |
    | `t_bin_trigger_rate`  | `np.ndarray[float]` | `(N_DU)`                           | Time span for each DU over which the mean trigger rates are computed. Units: s |

- Every `RAW_bkg_file` mirrors the original GP80 data file name, for example: `RAW_GP80_20250204_141251_RUN145_MD_RAW-ChanXYZ-20dB-GP43-20hz-0001.npz`

    **CONTENTS OF `RAW_bkg_file.npz`**

    | Field                 | Type                | Array shape                        | Description                                       |
    |-----------------------|---------------------|------------------------------------|---------------------------------------------------|
    | `traces`              | `np.ndarray[int]`   | `(N_entries,N_channels,N_samples)` | Raw ADC traces                                    |

### Signal

If $\geq 1$ DU is triggered by the FLT-0 in an event, the entire event is tagged as FLT-0 pass for SLT studies. Each simulated event that passes the FLT-0 is then stored in a single data file. We save both filtered traces, which we will pass on to the FLT-1 (and SLT?), and raw traces for completeness.

- Every `sig_data_dir` is named according to the chosen FLT-0 parameters: `ZHAireS_DC2RF2Alpha_CH_**_MODE_**_TH1_**_TH2_**_TQUIET_**_TPER_**_TSEPMAX_**_NCMIN_**_NCMAX_**/`

- Each `sig_data_dir` contains a `metadata.npz` file with information about the data, the FLT-0,...

    **CONTENTS OF SIGNAL `metadata.npz`**

    | Field                 | Type    | Description                                                                                     |
    |-----------------------|---------|-------------------------------------------------------------------------------------------------|
    | `dict_trigger_params` | `dict`  | Dictionary of FLT-0 parameters. See description above                                           |
    | `root_sim_dir`        | `str`   | Directory of the original simulation files in GrandRoot format                                  |
    | `sim_software`        | `str`   | Software used to simulate the air-shower radio emission                                         |
    | `samples_from_edge`   | `int`   | Number of samples to exclude from the start of the trace for the FLT-0 evaluation               |
    | `channel_pol`         | `dict`  | Map of trace channel to antenna arm / polarization                                              |
    | `channels_flt0`       | `list`  | Channels used for the evaluation of FLT-0                                                       |
    | `mode_flt0`           | `list`  | Mode used to trigger FLT-0. Options are `OR` (one channel enough) `AND` (all channels required) |

- Every `FILTERED_sig_file` mirrors the original DC2 simulation file name, and includes the corresponding event number at the end, for example: `FILTERED_adc_13056-158_L1_0000_run_0_event_11980.npz`

    **CONTENTS OF `FILTERED_sig_file.npz`**

    Normally `N_channels = 3`, `N_samples = 1024`

    | Field                 | Type                | Array shape                        | Description                                       |
    |-----------------------|---------------------|------------------------------------|---------------------------------------------------|
    | `traces`              | `np.ndarray[int]`   | `(N_entries,N_channels,N_samples)` | Filtered ADC traces                               |
    | `snr`                 | `np.ndarray[float]` | `(N_entries)`                      | SNRs of filtered ADC traces                       |
    | `t_pulse`             | `np.ndarray[float]` | `(N_entries,N_channels)`           | Sample/index of pulse maximum. `-1` if no FLT-0.  |
    | `du_ids`              | `np.ndarray[int]`   | `(N_entries)`                      | DU IDs                                            |
    | `du_seconds`          | `np.ndarray[int]`   | `(N_entries)`                      | Timestamp up to second precision                  |
    | `du_nanoseconds`      | `np.ndarray[int]`   | `(N_entries)`                      | Timestamp decimals up to nanosecond precision     |
    | `event_number`        | `int`               | ---                                | Event number                                      |
    | `run_number`          | `int`               | ---                                | Run number                                        |
    | `du_xyz`              | `np.ndarray[int]`   | `(N_DU,3)`                         | DU positions in GRAND detector frame. Units: m    |
    | `primary_type`        | `int`               | ---                                | Code for the primary particle (proton = 2212, iron = 1000260560) |
    | `energy_primary`      | `float`             | ---                                | Energy of the primary particle. Units: GeV        |
    | `zenith`              | `float`             | ---                                | Zenith of the primary particle. Units: deg        |
    | `azimuth`             | `float`             | ---                                | Azimuth of the primary particle. Units: deg       |
    | `omega`               | `np.ndarray[float]` | `(N_DU)`                           | Opening angle of antenna w.r.t. shower core. Units: deg |
    | `omega_c`             | `float`             | ---                                | Cherenkov angle for the shower geometry. Units: deg |
    | `shower_core_pos`     | `np.ndarray[float]` | `(3)`                              | Shower core position in GRAND detector frame. Units: m |
    | `xmac_pos_shc`        | `np.ndarray[float]` | `(3)`                              | Coordinates of $X_{\rm max}$ in shower-core frame. Units: m |
    | `FLT0_first_T1_idcs`  | `np.ndarray[int]`   | `(N_entries,N_channels)`           | Position of the first T1 threshold in the FLT-0 evaluation, per channel. If no trigger, the value is `-1`|
    | `n_FLT0`              | `np.ndarray[int]`   | `(N_entries,N_channels)`           | Number of FLT-0 triggers in a trace, per channel  |

- Every `RAW_sig_file` mirrors the original DC2 simulation file name, and includes the corresponding event number at the end, for example: `RAW_adc_13056-158_L1_0000_run_0_event_11980.npz`

    **CONTENTS OF `RAW_bkg_file.npz`**

    | Field                 | Type                | Array shape                        | Description                                       |
    |-----------------------|---------------------|------------------------------------|---------------------------------------------------|
    | `traces`              | `np.ndarray[int]`   | `(N_entries,N_channels,N_samples)` | Raw ADC traces                                    |
    <!-- | `snr`                 | `np.ndarray[float]` | `(N_entries,N_channels)`           | SNR of the simulated air-shower pulse in the raw traces | -->
