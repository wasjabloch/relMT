(file-formats)=
# File Formats

## Units and coordinates

All units are SI units. Location in `m`, Seismic moment in `Nm`.

We assume a right-handed Cartesian coordinate system where $x_1, x_2, x_3$ are North, East, Down. Azimuth is in degree east of north ($x_1 \rightarrow x_2$) and plunge in degree down from horizontal ($(x_1,x_2) \rightarrow x_3$).

Time and space coordinates are relative to an arbitrary origin.

Time measurements are required to tie time shifts of seismic traces to an
absolute reference frame and can be useful when importing and exporting seismic
traces. They are not required for moment tensor calculation.

## Input file formats

Any lines stating with `#` are ignored.

Trailing columns are ignored.

For each cluster of seismic events we assume the following directory structure:

```none
root/
+-- config.yaml
+-- exclude.yaml
+-- data/
    +-- stations.txt
    +-- events.txt
    +-- phases.txt
    +-- reference_mt.txt
    +-- default-hdr.yaml
    +-- STATION_PHASE-hdr.yaml
    +-- STATION_PHASE-wvarr.npy
```

The name of the `root/` directory is arbitrary.

### `config.yaml` configuration file

The configuration file holds the options that control the runtime behavior of
`relMT`. It is located in the `root/` directory.

An empty configuration file can be created with:

```python
from relmt import core
core.Config().to_file("config.yaml")
```

which yields the file ``config.yaml``:

```yaml
# relMT configuration

# Input files
# -----------
#
# Path to the seismic event catalog, e.g. 'data/events.txt'
# (str)
event_file:

# Path to the station location file, e.g. 'data/stations.txt'
# (str)
station_file:

# Path to the phase file, e.g. 'data/phases.txt'
# (str)
phase_file:

# Path to the reference moment tensor file, e.g. 'data/reference_mt.txt'
# (str)
reference_mt_file:

# Runtime options
# ---------------
#
# Logging verbosity for relMT modules. One of 'DEBUG', 'INFO', 'WARNING', 'ERROR',
# 'CRITICAL', 'NOTSET'
# (str)
loglevel:

# Number of threads to use in some parallel computations
# (int)
ncpu:

# Amplitude parameters
# --------------------
#
# Suffix appended to files, naming the parameters parsed to 'amplitude'
# (str)
amplitude_suffix:

# Method to meassure relative amplitudes. One of:
# - 'indirect': Estimate relative amplitude as the ratio of principal seismogram
#     contributions to each seismogram.
# - 'direct': Compare each event combination seperatly.
# (str)
amplitude_measure:

# Filter method to apply for amplitude measure. One of:
# - 'manual': Use 'highpass' and 'lowpass' of the waveform header files.
# - 'auto': compute filter corners using the 'auto_' options below
# (str)
amplitude_filter:

# Method to estimate lowpass filter that eliminates the source time function. One
# of:
# - 'fixed': Use the value 'fixed_lowpass' (not implemented)
# - 'corner': Estimate from apparent corner frequency in event spectrum
# - 'duration': Filter by 1/source duration of event magnitude.
#     Requires 'auto_lowpass_stressdrop_range'
# (str)
auto_lowpass_method:

# When estimating the lowpass frequency of an event as the corner frequency
# (auto_lowpass_method: 'corner'), assume a stressdrop within this range (Pa).
# (list)
auto_lowpass_stressdrop_range:

# Include frequencies with this signal-to-noise ratio to optimal bandpass filter.
# Respects lowpass constraint. If not supplied, do not attempt to optimize
# passband.
# (float)
auto_bandpass_snr_target:

# Minimum ratio (dB) of low- / highpass filter bandwidth in an amplitude ratio
# measurement. When positive, discard observation outside dynamic range. When
# negative, extend lower highpass until (positive) dynamic range is reached.
# (float)
min_dynamic_range:

# Quality control paramters
# -------------------------
#
# Suffix appended to the amplitude suffix, naming the quality control parameters
# parsed to 'qc'
# (str)
qc_suffix:

# Discard amplitude measurements with a higher misfit than this.
# (float)
max_amplitude_misfit:

# Maximum first normalized singular value to allow for an S-wave reconstruction. A
# value of 1 indicates that S-waveform adheres to rank 1 rather than rank 2 model.
# The relative amplitudes Babc and Bacb are then not linearly independent.
# (float)
max_s_sigma1:

# Maximum difference in magnitude between two events to allow an amplitude
# measurement.
# (float)
max_magnitude_difference:

# Maximum allowed distance (m) between two events.
# (float)
max_event_distance:

# Minimum number of equations required to constrain one moment tensor
# (int)
min_equations:

# Maximum azimuthal gap allowed for one moment tensor
# (float)
max_gap:

# Use two equations per S-amplitude observation (`False` only includes the one
# with the highest norm of the polarization vector.
# Warning: `False` appears broken)
# (bool)
keep_other_s_equation:

# Maximum number of S-wave equation in the linear system. If more are available,
# iterativley discard those with redundant pair-wise observations, on stations
# with many observations gap, and with a higher misfit
# (int)
max_s_equations:

# When reducing number of S-wave equations, increase importance of these events by
# not counting them in the redundancy score. Use to keep many equations e.g. for
# the reference event or specific events of interest.
# (list)
keep_events:

# When reducing the number of S-wave equations, rank observations iteratively this
# many times by redundancy and remove the most redundant ones. A higher number is
# faster, but may result in discarding less-redundant observations.
# (int)
equation_batches:

# Solve parameters
# ----------------
#
# Suffix appended to amplitude and qc suffices indicating the parameter set parsed
# to 'solve'
# (str)
result_suffix:

# Event indices of the reference moment tensor(s) to use
# (list)
reference_mts:

# Weight of the reference moment tensor
# (float)
reference_weight:

# Constrain the moment tensor. 'none' or 'deviatoric'
# (str)
mt_constraint:

# Minimum misfit to assign a full weight of 1. Weights are scaled lineraly from
# `min_amplitude mistfit` = 1 to `max_amplitude_misfit` = `min_amplitude_weight`"
#
# (float)
min_amplitude_misfit:

# Weight assigned to the maxumum amplitude misfit
# (float)
min_amplitude_weight:

# Number of samples to draw for calculating uncertainties. If not given, do not
# bootstrap.
# (int)
bootstrap_samples:
```

### `exclude.yaml` exclude file

The exclude file lists the events, stations, phases and waveforms to exclude
from processing. It is located in the `root/` directory. An empty exclude file can
be created with:

```python
from relmt import core, io
io.save_yaml("exclude.yaml", core.exclude)
```

which yields the file:

```yaml
station: []
event: []
waveform: []
phase_manual: []
phase_auto_nodata: []
phase_auto_snr: []
phase_auto_cc: []
phase_auto_ecn: []
```

### `stations.txt` station file

The station file holds the station names and locations. It is located in the
`data/` subdirectory.

`stations.txt` has four columns:

1. Station name (must not contain `_`)
2. Northing (meter)
3. Easting (meter)
4. Depth (meter)

### `events.txt` event file

The event files holds the seismic event catalog. It is located in the `data/`
subdirectory.

`events.txt` has seven columns:

1. Event index
2. Northing (meter)
3. Easting (meter)
4. Depth (meter)
5. Origin time (arbitrary float in seconds, e.g. epoch seconds)
6. Magnitude
7. Event name (arbitrary string)

We use the event *index* (1.) to internally refer to events. We use origin time
(5.) for external reference and magnitude (6.) for quality assurance. Time and
magnitude can be `nan` if unknown. The event *name* (7.) is an external event
reference (e.g.  event ID within a larger catalog), which may be used for data
import and export.

### `phases.txt` phase file

The phase file holds the arrival times and take-off angles of the seismic phases
at the stations. It is located in the `data/` subdirectory.

`phases.txt` has six columns:

1. Event index (as in the `Event file`)
2. Station name (as in the `Station file`)
3. Phase type (`P` or `S`)
4. Arrival time (float)
5. Azimuth (east of north)
6. Plunge (down from horizontal)

Arrival time (4.) is required to tie time shifts of seismic traces to an
absolute reference frame and can be useful when importing and exporting seismic
traces. We recommend to use absolute time in epoch seconds or seconds after
origin time.

### `reference_mt.txt` reference moment tensor file

The reference moment tensor file holds the components of the reference moment
tensor(s). It is located in the `data/` subdirectory.

`reference_mt.txt` has seven columns:

1. Event index (as in the `Event file`)
2. nn -
3. ee -
4. dd -
5. ne -
6. nd -
7. ed - component of the moment tensor (Nm)

The components of the moment tensor are given in units of Newton meter.
Scientific notation (e.g. `1.2e19`) is encouraged for the large floats.

(header-file)=
### `STATION_PHASE-hdr.yaml` waveform header files

The waveform header files hold the meta information about the seismic waveforms.
Default values are read from `default-hdr.yaml` located in the `data/`
subdirectory. Values for a specific waveform are overwritten by the values from
files named `STATION_PHASE-hdr.yaml` located in the `data/` subdirectory, where
`STATION` is the station name as in the station and phase files, and `PHASE` is
the phase type as in the phase file (`P` or `S`).

A template header file can be created with:

```python
from relmt import core
core.Header().to_file("default-hdr.yaml")
```

which yields the file ``default-hdr.yaml``:

```yaml
# relMT waveform header

# Station code
# (str)
station:

# Seismic phase type to consider ('P' or 'S')
# (str)
phase:

# Optional variable name that holds the waveform array
# (str)
variable_name:

# One-character component names ordered as in the waveform array, as one string
# (e.g. 'ZNE')
# (str)
components:

# Sampling rate of the seismic waveform (Hertz)
# (float)
sampling_rate:

# Time window symmetric about the phase pick (i.e. pick is near the central
# sample) (seconds)
# (float)
data_window:

# Start of the phase window before the arrival time pick (negative seconds before
# pick).
# (float)
phase_start:

# End of the phase window after the arrival time pick (seconds after pick).
# (float)
phase_end:

# Combined length of taper that is applied at both ends beyond the phase window.
# (seconds)
# (float)
taper_length:

# Common high-pass filter corner of the waveform (Hertz)
# (float)
highpass:

# Common low-pass filter corner of the waveform (Hertz)
# (float)
lowpass:

# Regard absolute amplitudes at and below this value as null
# (float)
null_threshold:

# Minimum allowed signal-to-noise ratio (dB) of signals for event exclusion
# (float)
min_signal_noise_ratio:

# Minimum allowed absolute averaged correlation coefficient of a waveform for
# event exclusion
# (float)
min_correlation:

# Minimum allowed norm of the principal component expansion coefficients
# contributing to the waveform reconstruction for event exclusion
# (float)
min_expansion_coefficient_norm:

# Read combinations from file names STATION_PHASE-combination.txt
# (bool)
combinations_from_file:

# Event indices corresponding to the first dimension of the waveform array. Do not
# edit.
# (list)
events_:
```

### `STATION_PHASE-wvarr.npy` waveform array files

The waveform array files hold the event waveforms of one phase at one
station. They are `NumPy` array files that obey the naming convention
`STATION_PHASE-wvarr.npy`, where `STATION` is the station name and `PHASE` is
the phase type (`P` or `S`). The files are located in the `data/` subdirectory.

Arrays are three dimensional, with events sorted along the first dimension,
components along the second dimension and seismogram samples along the
third dimension. The resulting shape is ``(n_events, n_channels, n_samples)``,
where ``n_events`` and ``n_channels`` are the length of the ``events`` and
``channels`` values in the [header file](#header-file), and ``n_samples`` is `data_window`
multiplied by `sampling_rate`.

Event indices stored in the `events:` keyword of the waveform header must
correspond to the seismic traces stored in the first array dimension. Component
names stored in the `components:` keyword of the waveform header must correspond
to the seismogram components in the second dimension of the waveform array.
