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

We assume the following directory structure

```none
./
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

### `config.yaml` configuration file

The configuration file holds the options that control the runtime behavior of
`relMT`. It is located in the root directory.

An empty configuration file can be created with:

```python
from relmt import core
core.Config().to_file("config.yaml")
```

which yields the file ``config.yaml``:

```yaml
# relMT configuration

# Event indices of the reference moment tensors to use
# (list)
reference_mts:

# Weight of the reference moment tensor
# (float)
reference_weight:

# Constrain the moment tensor. 'none' or 'deviatoric'
# (str)
mt_constraint:

# Maximum misfit allowed for amplitude reconstruction
# (float)
max_amplitude_misfit:

# Number of samples to draw for calculating uncertainties
# (int)
bootstrap_samples:

# Number of threads to use for parallel computations
# (int)
ncpu:

# Minimum ratio (dB) of low- / highpass filter bandwidth in an amplitude ratio measurement
# (float)
min_dynamic_range:
```

### `exclude.yaml` exclude file

The exclude file is located in the root directory.

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

`events.txt` has six columns:

1. Event index (must be consecutive starting at 0)
2. Northing (meter)
3. Easting (meter)
4. Depth (meter)
5. Origin time (arbitrary float in seconds, e.g. epoch seconds)
6. Magnitude
7. Event name (arbitrary string)

We use the event *index* (1.) to internally refer to events. We use origin time
(5.) for external reference and magnitude (6.) for quality assurance. They can
be `nan` if unknown. The event *name* (7.) is an external event reference (e.g.
event ID within a larger catalog).

### `phases.txt` phase file

The phase file holds the arrival times and take-off angles of the seismic phases
at the stations. It is located in the `data/` subdirectory

`phases.txt` has six columns:

1. Event index (as in the `Event file`)
2. Station name (as in the `Station file`)
3. Phase (`P` or `S`)
4. Arrival time (float)
5. Azimuth (east of north)
6. Plunge (down from horizontal)

Arrival time (4.) is required to tie time shifts of seismic traces to an
absolute reference frame and can be useful when importing / exporting seismic
traces.to tie arbitrary. We recommend to use absolute time in epoch seconds or
seconds after origin time.

### `reference_mt.txt` reference moment tensor file

The reference moment tensor file holds the components of the reference moment
tensor(s) is located in the `data/` subdirectory.

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

yields the file ``default-hdr.yaml``:

```yaml
# relMT waveform header

# Station code
# (str)
station:

# Seismic phase type to consider ('P' or 'S')
# (str)
phase:

# One-character component names ordered as in the waveform array, as one string (e.g. 'ZNE')
# (str)
components:

# Sampling rate of the seismic waveform (Hertz)
# (float)
sampling_rate:

# Event indices corresponding to the first dimension of the waveform array.
# (list[int])
events:

# Time window symmetric about the phase pick (i.e. pick is near the central sample) (seconds)
# (float)
data_window:

# Start of the phase window before the arrival time pick (negative seconds before pick).
# (float)
phase_start:

# End of the phase window after the arrival time pick (seconds after pick).
# (float)
phase_end:

# Combined length of taper that is applied at both ends beyond the phase window. (seconds)
# (float)
taper_length:

# Common high-pass filter corner of the waveform (Hertz)
# (float)
highpass:

# Common low-pass filter corner of the waveform (Hertz)
# (float)
lowpass:

# Maximum shift to allow in multi-channel cross correlation (seconds)
# (float)
maxshift:
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
