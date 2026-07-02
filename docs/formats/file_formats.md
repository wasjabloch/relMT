(file-formats)=
# File Formats

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

(config-file)=
## Configuration file `config.yaml`

The configuration file holds the options that control the runtime behavior of
`relMT`. It is located in the `root/` directory.

An empty configuration file can be created with:

```python
from relmt import core
core.Config().to_file("config.yaml")
```

which yields the file ``config.yaml``:

```{literalinclude} config.yaml
:caption: config.yaml
:language: yaml
```

(exclude-file)=
## Exclude file `exclude.yaml`

The exclude file lists the events, stations, phases and waveforms to exclude
from processing. It is located in the `root/` directory. An empty exclude file
can be created with:

```python
from relmt import core, io
io.save_yaml("exclude.yaml", core.exclude)
```

which yields the file:

```{literalinclude} exclude.yaml
:caption: exclude.yaml
:language: yaml
```

(station-file)=
## Station file

The station file holds the station names and locations. It is located in the
`data/` subdirectory and has four columns:

1. Station name (must not contain `_`)
2. Northing (meter)
3. Easting (meter)
4. Depth (meter)

A station file can be created with:

```python
from relmt import core, io
io.write_station_table(
    {"STA": core.Station(0.0, 0.0, 0.0, "STA")}, "stations.txt"
)
```

which yields the file:

```{literalinclude} stations.txt
:caption: stations.txt
:language: none
```

(event-file)=
## Event file

The event files holds the seismic event catalog. It is located in the `data/`
subdirectory. It has seven columns:

1. Event index (arbitrary integer)
2. Northing (meter)
3. Easting (meter)
4. Depth (meter)
5. Origin time (arbitrary float in seconds, e.g. epoch seconds)
6. Magnitude
7. Event name (arbitrary string)

We use the arbitrary event *index* (1.) to internally refer to events. We use
origin time (5.) for external reference and magnitude (6.) for quality
assurance. Time and magnitude can be `nan` if unknown. The event *name* (7.) is
an arbitrary event reference (e.g. event ID within a larger catalog), which may
be used for data import and export.

An event file can be created with:

```python
from relmt import core, io
io.write_event_table(
    {0: core.Event(0.0, 0.0, 0.0, 0.0, 0.0, "SomeName")}, "events.txt"
)
```

which yields the file:

```{literalinclude} events.txt
:caption: events.txt
:language: none
```

(phase-file)=
## Phase file

The phase file holds the arrival times and take-off angles of the seismic phases
at the stations. It is located in the `data/` subdirectory.

`phases.txt` has six columns:

1. Event index (as in the [event file](#event-file))
2. Station name (as in the [station file](#station-file))
3. Phase type (`P` or `S`)
4. Arrival time (float)
5. Azimuth (east of north)
6. Plunge (down from horizontal)

Arrival time (4.) is required to tie time shifts of seismic traces to an
absolute reference frame and can be useful when importing and exporting seismic
traces. We recommend to use absolute time in epoch seconds or seconds after
origin time.

A phase file can be created with:

```python
from relmt import core, io
io.write_phase_table(
    {core.join_phaseid(0, "STA", "P"): core.Phase(0.0, 0.0, 0.0)}, "phases.txt"
)
```

which yields the file:

```{literalinclude} phases.txt
:caption: phases.txt
:language: none
```

(reference-mt-file)=
## Reference moment tensor file

The reference moment tensor file holds the components of the reference moment
tensor(s). It is located in the `data/` subdirectory.

`reference_mt.txt` has seven columns:

1. Event index (as in the [event file](#event-file))
2. nn -
3. ee -
4. dd -
5. ne -
6. nd -
7. ed - component of the moment tensor (Nm)

The components of the moment tensor are given in units of Newton meter.
Scientific notation (e.g. `1.2e19`) is encouraged for the large floats.

```python
from relmt import core, io
io.write_mt_table(
    {0: core.MT(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}, "reference_mts.txt"
)
```

which yields the file:

```{literalinclude} reference_mts.txt
:caption: reference_mts.txt
:language: none
```

(header-file)=
## Waveform header files `STATION_PHASE-hdr.yaml`

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

```{literalinclude} default-hdr.yaml
:caption: default-hdr.yaml
:language: yaml
```

(waveform-file)=
## Waveform array files `STATION_PHASE-wvarr.npy`

The waveform array files hold the event waveforms of one phase at one
station. They are `NumPy` array files that obey the naming convention
`STATION_PHASE-wvarr.npy`, where `STATION` is the station name and `PHASE` is
the phase type (`P` or `S`). The files are located in the `data/` subdirectory.

Arrays are three dimensional, with events sorted along the first dimension,
components along the second dimension and seismogram samples along the third
dimension. The resulting shape is ``(n_events, n_channels, n_samples)``, where
``n_events`` and ``n_channels`` are the length of ``events_`` and ``channels``
values in the [header file](#header-file), and ``n_samples`` is `data_window`
multiplied by `sampling_rate`.

Event indices stored in the `events_:` keyword of the waveform header must
correspond to the seismic traces stored in the first array dimension. Component
names stored in the `components:` keyword of the waveform header must correspond
to the seismogram components in the second dimension of the waveform array.
