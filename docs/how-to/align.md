# Align waveforms

## 1. Prerequisites

We assume a project has been [set up](setup.md). For simplicity, we here only
align *P* and *S* on one station, 'ASTA'. The corresponding minimal project
directory looks like this:

```none
myproject/
+-- config.yaml
+-- exclude.yaml
+-- data/
    +-- stations.txt
    +-- default-hdr.yaml
    +-- ASTA_P-hdr.yaml
    +-- ASTA_P-wvarr.npy
    +-- ASTA_S-hdr.yaml
    +-- ASTA_S-wvarr.npy
+-- align1/
+-- align2/
```

Note that the event file, phase file, and reference MT file are not required at this point.

## 2. Synopsis

The aim is to have waveforms aligned to sub-sample accuracy, so that as much
energy as possible is projected onto the first principal component of the
waveform matrix for *P*-waves, or the first two principal components for
S-waves.

The objective function to be minimized for *P*-waves is:

```{math}
\epsilon_P = 1 - C_{ij}^2
```

and for *S*-waves

```{math}
\epsilon_S = 1 + 2 C_{ij} C_{jk} C_{ik} - C_{ij}^2 - C_{jk}^2 - C_{ik}^2
```

where the indices ${i, j, k}$ indicate the event indices of the combined pairs
(for *P*-waves) and triplets (for *S*-waves). The objective functions imply that
a polarity reversal between any two waveforms is expected. For this reason, the
choice of an accurate time window is critical and waveform alignment is most of
the times a two-step process.

We will first exclude corrupt or noisy traces. We then choose relatively wide
time windows and pre-align waveforms. We may then choose narrower time windows
and different filter passbands. We then compare measures that indicate success
of alignment, before we ultimately align the waveforms with high accuracy.

## 3. Data quality control

For a data set of local seismicity with a magnitude of 1 to 2, with wavetrains
about 1 to 2 seconds long and a pick accuracy better than 1 second, the
following configuration could be a reasonable one:

```{code-block} yaml
:caption: data/default-hdr.yaml

# Sampling rate of the seismic waveform (Hertz)
sampling_rate: 100

# Time window symmetric about the phase pick (i.e. pick is near the central
# sample) (seconds)
data_window: 8

# Start of the phase window before the arrival time pick (negative seconds before
# pick)
phase_start: -1

# End of the phase window after the arrival time pick (seconds after pick).
phase_end: 3

# Combined length of taper that is applied at both ends beyond the phase window.
# (seconds)
taper_length: 0.5

# Common high-pass filter corner of the waveform (Hertz)
highpass: 0.2

# Common low-pass filter corner of the waveform (Hertz)
lowpass: 10
```

Corrupt traces may take the form of `nan` values within the trace or traces that
are all `0`. Such traces can be excluded using `relmt exclude --nodata`. Very
small values, e.g. `1e-3`, can also be interpreted as corrupt using the
`null_threshold` parameter.

Noisy traces can be excluded via their signal-to-noise ratio using `relmt
exclude --snr`. The signal amplitude is meassured within the time window given
by `phase_start` and `phase_end` and the noise amplitude before `phase_start`
within the frequency band enclosed by `highpass` and `lowpass`.

To apply the *no data* and the *too noisy* criteria at once, one can define:

```{code-block} yaml
:caption: data/default-hdr.yaml
# Regard absolute amplitudes at and below this value as null
null_threshold: 0.001

# Minimum allowed signal-to-noise ratio (dB) of signals for event exclusion
min_signal_noise_ratio: 0
```

and call on the command line, from within `myproject/`

```{code-block} sh
:caption: shell
relmt exclude --nodata --snr
```

Assuming the *P*-phase of event `1` on station 'ASTA' had a poor signal-to-noise
ratio, and the *S*-phase of event `2` on station 'ASTA' was corrupt, this would
generate the following entries in `exclude.yaml`:

```{code-block} yaml
:caption: exclude.yaml
phase_auto_snr:
- 1_ASTA_P

phase_auto_nodata:
- 2_ASTA_S
```

These phases will be excluded from further processing.

## 4. Pre-alignment

Waveforms can be aligned using multi-channel cross-correlation (MCCC) and
principal component analysis (PCA). MCCC can accommodate time-shifts as long as
`data_window` and is limited to sample accuracy, while PCA assumes that
waveforms are already aligned to 1/4 wavelength and can achieve sub-sample
accuracy. For pre-alignment, MCCC suffices.

```{code-block} sh
:caption: shell
relmt align --mccc
```

:::{hint}
To align many stations in parallel, consider granting access to various CPUs.
Some internal *NumPy* routines will still use all available resources,
regardless of this setting.

```{code-block} yaml
:caption: config.yaml

# Number of threads to use in some parallel computations
ncpu:
```

:::

This call reads waveforms from `data/` and writes aligned waveforms to
`align1/`, so that the project directory now looks like this:

```none
myproject/
+-- config.yaml
+-- exclude.yaml
+-- data/
    +-- stations.txt
    +-- default-hdr.yaml
    +-- ASTA_P-hdr.yaml
    +-- ASTA_P-wvarr.npy
    +-- ASTA_S-hdr.yaml
    +-- ASTA_S-wvarr.npy
+-- align1/
    +-- ASTA_P-hdr.yaml
    +-- ASTA_P-wvarr.npy
    +-- ASTA_S-hdr.yaml
    +-- ASTA_S-wvarr.npy
+-- align2/
```

## 5. Alignment control

One can check visually if the alignment did succeed by inspecting the aligned waveforms:

```{code-block} sh
:caption: shell
relmt plot alignment align1/ASTA_P-wvarr.npy
relmt plot alignment align1/ASTA_S-wvarr.npy
```

The resulting plot shows the applied time shifts, the waveforms, the
signal-to-noise ratio, the averaged cross correlation coeffcient per trace, the
cross-correlation matrix and the expansion coefficient norm per trace in an
interactive window.

Based on these plots, one can find optimal filter corners and time windows for
which waveforms appear similar and characteristic. Let us assume the
*P*-wavetrain on station 'ASTA' turns out to be 1.5 seconds long, was shifted so
that it starts 0.5 seconds after the central sample and is most pronounced in
the 1-5 Hz band.  We manipulate `align1/ASTA_P-hdr.yaml` in the following way:

```{code-block} yaml
:caption: align1/ASTA_P-hdr.yaml
# Start of the phase window before the arrival time pick (negative seconds before
# pick)
phase_start: 0.5

# End of the phase window after the arrival time pick (seconds after pick).
phase_end: 2.0

# Common high-pass filter corner of the waveform (Hertz)
highpass: 1

# Common low-pass filter corner of the waveform (Hertz)
lowpass: 5
```

Re-running the plot function shows the plot with the changed parameters:

```{code-block} sh
:caption: shell
relmt plot alignment align1/ASTA_P-wvarr.npy
```

From intuition or reasoning one can now define parameters to exclude waveforms
that do not align well. Two parameters are most relevant here:

* `min_correlation`: This parameter between *0* and *1* measures
  how well on average a waveform correlates or anti-correlates with one other
  (*P*-waves) or two others (*S*-waves), where *1* is best and indicates perfect
  correlation. This parameter is optimized by the MCCC algorithm.

* `min_expansion_coefficient_norm`: This parameter between *0* and *1* measures
  how well a waveform is reprented by the first principal seismogram (*P*-waves)
  or the first two principal seismograms (*S*-waves), where *1* is best and
  indicates perfect representation. This parameter is optimized by the PCA
  algotithm.

It is possible and can be meaningful to define different parameters for
different stations and/or phases, for example:

```{code-block} yaml
:caption: align1/ASTA_*P*-hdr.yaml
# Minimum allowed absolute averaged correlation coefficient of a waveform for
# event exclusion
min_correlation: 0.8
```

```{code-block} yaml
:caption: align1/ASTA_*S*-hdr.yaml
# Minimum allowed norm of the principal component expansion coefficients
# contributing to the waveform reconstruction for event exclusion
min_expansion_coefficient_norm: 0.7
```

When exluding phases, we must now provide the `-a 1` argument to indicate that
we are reading from the `align1/` directory. To exclude based on the average
cross-correlation coefficient, we provide the `--cc` flag. To exclude based on
the expansion coefficient norm, we provide the `--ecn` flag. Note that unset
(i.e. `None`) values in the header files indicate that the criterion is not used
for phase exclusion.

```{code-block} sh
:caption: shell
relmt exclude -a 1 --cc --ecn
```

Assuming the *P*-wave of event `3` had a `cc` value lower than 0.8 on station
'ASTA' and the *S*-wave of event `4` an `ecn` value below 0.7, the following new
entries would be written to `exclude.yaml`:

```{code-block} yaml
:caption: exclude.yaml
phase_auto_cc:
- 3_ASTA_P

phase_auto_ecn:
- 4_ASTA_S
```

One can as well exclude phases manually from further processing, e.g.:

```{code-block} yaml
:caption: exclude.yaml
phase_manual:
- 5_ASTA_P
```

## 6. Re-align

To re-align waveforms, we recommend to use the PCA algorithm as a last step. If
one is confident that all waveforms are aligned to 1/4 wavelength one can only
perform PCA alignment. Remember to read from `align1/` using the `-a 1` flag.

```{code-block} sh
:caption: shell
relmt align -a 1 --pca
```

In case cycle skips are still present we recomment to align using MCCC and
PCA in succession. This is the default behaviour of `relmt align` when no method
is specified.

```{code-block} sh
:caption: shell
relmt align -a 1
```

## 7. Result

After this step, the `align2/` directory is populated with the aligned waveforms.

```none
myproject/
+-- config.yaml
+-- exclude.yaml
+-- data/
    +-- stations.txt
    +-- default-hdr.yaml
    +-- ASTA_P-hdr.yaml
    +-- ASTA_P-wvarr.npy
    +-- ASTA_S-hdr.yaml
    +-- ASTA_S-wvarr.npy
+-- align1/
    +-- ASTA_P-hdr.yaml
    +-- ASTA_P-wvarr.npy
    +-- ASTA_S-hdr.yaml
    +-- ASTA_S-wvarr.npy
+-- align2/
    +-- ASTA_P-hdr.yaml
    +-- ASTA_P-wvarr.npy
    +-- ASTA_S-hdr.yaml
    +-- ASTA_S-wvarr.npy
```

You can inspect the results with `relmt plot alignment`. When re-doing
individual stations, delete the corresponding `-hdr.yaml` and `-wvarr.npy` files
from the target directory. When re-doing all stations, activate the `-o` flag to
overwrite in the target directory, e.g. `relmt align -a 1 -o`.
