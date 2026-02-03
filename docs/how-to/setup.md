# Set up a project

## 1. Initialize directory

In the folder where the files of your new project should reside, call from a terminal

```sh
relmt init
```

Alternativley, to create a new, empty project folder, e.g. `myproject/`, call

```sh
relmt init myproject
```

This creates the following structure of directories and template files:

```none
myproject/
+-- config.yaml
+-- exclude.yaml
+-- data/
|   +-- default-hdr.yaml
+-- align1/
+-- align2/
+-- amplitude/
+-- result/
```

## 2. Create the additional text files

*relMT* needs to know, where the seismic stations and events are located, at
what angle the seismic rays take off and which are the values of the
reference moment tensors. This information is stored in the *station*,
*event*, *phase* and *reference MT* files in the `data/` subfolder:

```none
data/
+-- stations.txt
+-- events.txt
+-- phases.txt
+-- reference_mt.txt
```

The file names are arbitrary and must correspond to the respective entries
in `config.yaml`:

```{literalinclude} config-template.yaml
:caption: config.yaml
:language: yaml
:start-at: Path to the seismic event catalog
:end-at: reference_mt_file
```

The files obey a simple, whitespace-seperated text file format. For details,
see:

* [Station file](#station-file)
* [Event file](#event-file)
* [Phase file](#phase-file)
* [Reference MT file](#reference-mt-file)

:::{tip}
The following functions may be useful when creating the text files from
external resources.

* To create a station file from an *ObsPy* `Inventory` object

  * {py:class}`relmt.extra.read_obspy_inventory_files`
  * {py:class}`relmt.extra.read_station_inventory`
  * {py:class}`relmt.io.write_station_table`

* To create an event file from an external table

  * {py:class}`relmt.io.read_ext_event_table`
  * {py:class}`relmt.io.write_event_table`

* To create a phase file from a *NonLinLoc* `.hyp` file

  * {py:class}`relmt.io.read_phase_nll_hypfile`
  * {py:class}`relmt.io.write_phase_table`

* To create a phase file from a station file, event file and a velocity model using the `SKHASH` ray tracer

  * {py:class}`relmt.utils.phase_dict_azimuth`
  * {py:class}`relmt.utils.phase_dict_hash_plunge`
  * {py:class}`relmt.io.write_phase_table`

* To create a reference moment tensor file from an external moment tensor table

  * {py:class}`relmt.io.read_ext_mt_table`
  * {py:class}`relmt.io.write_mt_table`

:::

## 3. Create the waveform files

For each station and both (*P*, *S*) phases, gather all event waveforms and
store them as a 3-dimensional *NumPy* array as [waveform files](#waveform-file).
Note that the approximate wave arrival ("pick") is assumed at the center sample.

For each waveform file, populate a corresponding [header file](#header-file)
with at least the following attributes

```{code-block} yaml
:caption: STATION_PHASE-hdr.yaml
# Station code
station:

# Seismic phase type to consider ('P' or 'S')
phase:

# Event numbers corresponding to the first dimension of the waveform array.
events_:
```

The `events_` parameter is a list of integer numbers. The position in the list
corresponds to the position of the waveform along the first dimentsion of the
[waveform array](#waveform-file), while the value corresponds to the event
number (first row) in the [event file](#event-file).

Default parameters that are equal for multiple stations and phases may be
declared only once in `default-hdr.yaml`. Any values found in a specific
`STATION_PHASE-hdr.yaml` will overwrite the values defined here:

```{literalinclude} default-hdr.yaml
:caption: default-hdr.yaml
:language: yaml
:start-at: One-character component names
:end-at: lowpass:
```

:::{admonition} Example
A waveform array containing *P* wavetrains recorded at station *"BSTA"* of
events *1*, *3*, *7*, *11* and *4* (in that order), recorded on three chanels
with the names *"Z"*, *"N"*, *"E"* (in that order) with *500* samples (e.g. 5
seconds length with 100 samples per seconds) would have a shape of ``(5, 3,
500)``. The header file would have the fields:

```{code-block} yaml
:caption: BSTA_P-hdr.yaml

station: BSTA
phase: P
components: "ZNE"
sampling_rate: 100
data_window: 5
events_: [1, 3, 7, 11, 4]
:::

:::{tip}
To create waveform arrays from an *ObsPy* `Stream`, have a look at:

* {py:class}`relmt.extra.make_waveform_array`

:::

## 4. Example data structure

For the case of three stations 'ASTA', 'BSTA', and 'CSTA', all of which have
*P*- and *S*-wave observations, the resulting file structure would look like
this:

```none
data/
+-- stations.txt
+-- events.txt
+-- phases.txt
+-- reference_mt.txt
+-- default-hdr.yaml
+-- ASTA_P-hdr.yaml
+-- ASTA_P-wvarr.npy
+-- ASTA_S-hdr.yaml
+-- ASTA_S-wvarr.npy
+-- BSTA_P-hdr.yaml
+-- BSTA_P-wvarr.npy
+-- BSTA_S-hdr.yaml
+-- BSTA_S-wvarr.npy
+-- CSTA_P-hdr.yaml
+-- CSTA_P-wvarr.npy
+-- CSTA_S-hdr.yaml
+-- CSTA_S-wvarr.npy
```

:::{admonition} Congratulations
You are now ready to align the waveforms!
:::
