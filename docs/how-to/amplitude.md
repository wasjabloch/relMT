# Measure amplitudes

## 1. Prerequisites

We assume that we have a directory with aligned waveforms. Following [the
alignment guide](align.md), waveforms are located in `align2/`

```none
myproject/
+-- config.yaml
+-- exclude.yaml
+-- data/
    +-- stations.txt
    +-- events.txt
    +-- ...
+-- align1/
    +-- ...
+-- align2/
    +-- ASTA_P-hdr.yaml
    +-- ASTA_P-wvarr.npy
    +-- ASTA_S-hdr.yaml
    +-- ASTA_S-wvarr.npy
    +-- BSTA_P-hdr.yaml
    +-- BSTA_P-wvarr.npy
    +-- BSTA_S-hdr.yaml
    +-- BSTA_S-wvarr.npy
```

## 2. Choosing a set of amplitude options

In the configuration file, find the "Amplitude parameters" block that lists the
options that control how amplitudes are measured. We start by measuring
amplitudes within the frequency band defined in the `highpass` and `lowpass`
parameters of the individual `-hdr.yaml` files.

```{code-block} yaml
:caption: config.yaml
# Amplitude parameters
# --------------------
#
# Suffix appended to files, naming the parameters parsed to 'amplitude'
amplitude_suffix:

# Method to meassure relative amplitudes. One of:
# - 'indirect': Estimate relative amplitude as the ratio of principal seismogram
#     contributions to each seismogram.
# - 'direct': Compare each event combination seperatly.
amplitude_measure: direct

# Filter method to apply for amplitude measure. One of:
# - 'manual': Use 'highpass' and 'lowpass' of the waveform header files.
# - 'auto': compute filter corners using the 'auto_' options below
amplitude_filter: manual
```

## 3. Measure the amplitudes

To measure the amplitudes, we call `relmt amplitude`. We indicate that the aligned
waveforms are stored in the `align2/` subdirectory by supplying the `-a 2` option.

```{code-block} sh
:caption: shell
relmt amplitude -a 2
```

This creates the amplitdue files in the `amplitude/` subdirectory

```none
myproject/
+-- config.yaml
+-- exclude.yaml
+-- data/
    +-- ...
+-- align1/
    +-- ...
+-- align2/
    +-- ...
+-- amplitude/
    +-- P-amplitudes.txt
    +-- S-amplitudes.txt
```

## 4. Trying a different parameter set

You may want to try some of the options to find optimally suited filter bands
for the respective event combinations. We here fill in some values that have
proven useful for certrain data sets. Note that we parse a different
configuration file and set `amplitude_suffix`, so that the alternative option
set becomes apparent through naming. We here choose the suffix *auto_amp*, and
rename the confiugration file accordingly.

```{code-block} yaml
:caption: config-auto_amp.yaml

# Suffix appended to files, naming the parameters parsed to 'amplitude'
amplitude_suffix: auto_amp

# Method to meassure relative amplitudes. One of:
amplitude_measure: direct

# Filter method to apply for amplitude measure.
amplitude_filter: auto

# Method to estimate lowpass filter that eliminates the source time function. One
# of:
# - 'fixed': Use the value 'fixed_lowpass' (not implemented)
# - 'corner': Estimate from apparent corner frequency in event spectrum
# - 'duration': Filter by 1/source duration of event magnitude.
#     Requires 'auto_lowpass_stressdrop_range'
auto_lowpass_method: corner

# When estimating the lowpass frequency of an event as the corner frequency
# (auto_lowpass_method: 'corner'), assume a stressdrop within this range (Pa).
auto_lowpass_stressdrop_range: [1e5, 1e7]  # 0.1 to 10 MPa

# Include frequencies with this signal-to-noise ratio to optimal bandpass filter.
# Respects lowpass constraint. If not supplied, do not attempt to optimize
# passband.
auto_bandpass_snr_target: 0

# Minimum ratio (dB) of low- / highpass filter bandwidth in an amplitude ratio
# measurement. When positive, discard observation outside dynamic range. When
# negative, lower the highpass until the (positive) dynamic range is reached.
min_dynamic_range: 5
```

We use the `-c` option to explicitly parse the renamed configuration file.

```{code-block} sh
:caption: shell
relmt amplitude -a 2 -c config-auto_amp.yaml
```

Each phase observation is now assigned a bandpass, which is stored in
*amplitude/bandpass.yaml* (with the optional `amplitude_suffix` applied). If you
change any of the `auto_` options and re-run the above command, remember to
parse the `-o` option to overwrite the *bandpass.yaml* or change the
`amplitude_suffix`. If a *bandpass.yaml* with the matching `amplitude_suffix` is
found, we read from that file instead of computing new filter corners.

`min_dynamic_range` is only applied to waveform *combinations* and can therfore be changed after computing filter bands.

With the changed naming conventions, we now find an alternative set of amplitudes besides the previously computed.

```{code-block} none
:emphasize-lines: 14-16

myproject/
+-- config.yaml
+-- config-auto_amp.yaml
+-- ...
+-- data/
    +-- ...
+-- align1/
    +-- ...
+-- align2/
    +-- ...
+-- amplitude/
    +-- P-amplitudes.txt
    +-- S-amplitudes.txt
    +-- bandpass-auto_amp.yaml
    +-- P-amplitudes-auto_amp.txt
    +-- S-amplitudes-auto_amp.txt
    +-- ...
```

## 5. Result

The amplitude files may grow very large and can contain outliers. Before
building the liniear system it is important quality control the amplitudes.
