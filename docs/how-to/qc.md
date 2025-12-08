# Quality control amplitudes

## 1. Prerequisites

The aim is to select the "best" relative *P* and
*S* amplitudes from the potentially very large sets stored in
`P-amplitudes.txt` and `S-amplitudes.txt`. We assume the following file
structure:

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

As relative *S* amplitudes represent comparission of event *triplets* the number
of possible *S* combinations is $\binom{N}{3}$, for $N$ events, while the
maximum number of *P* pairs is only $\binom{N}{2}$.

## 2. Quality control parameters

The following parameters in `config.yaml` define the critera by which relative
amplitude observations should be excluded. As before, different parameter sets
are distinguished using `qc_suffix`, which is appended to the name of the output
files.

```{code-block} yaml
:caption: config.yaml

# Quality control paramters
# -------------------------
#
# Suffix appended to the amplitude suffix, naming the quality control parameters
# parsed to 'qc'
qc_suffix: qced
```

The first parameters pertain to the quality of waveform reconstruction:

* `max_amplitude_misfit` discribes how well a waveform is reconstructed, where a
  value of *0* indicates perfect reconstruction, *>1* indicates that the misfit
  amplitude is larger than the singal amplitude itself. This value should be *<1*.
* `max_s_sigma1` discribes the linear independence of the two relative *S*
  amplitudes. The values of $B_{abc}$ and $B_{acb}$ may become arbitrary as this
  value approaches *1* and the equation may not be linearily independent of
  others. However, in the case of very similar *S* waveforms, $\sigma_1$ may be
  close to *1* for almost all waveform combinations. One then runs into danger
  of excluding all observations when choosing too low a value.

```{code-block} yaml
:caption: config.yaml
# Discard amplitude measurements with a higher misfit than this.
max_amplitude_misfit:

# Maximum first normalized singular value to allow for an S-wave reconstruction. A
# value of 1 indicates that S-waveform adheres to rank 1 rather than rank 2 model.
# The relative amplitudes Babc and Bacb are then not linearly independent.
max_s_sigma1:
```

The next parameters pertain to properties of event combinations.

* `max_magnitude_difference`: Large differences in event magnitude may cause
  problems in the relative amplitude measurement when the lowpass filter corner
  was chosen too high and the depletion in high frequency energy of the
  larger-magnitude event becomes significant.
* `max_event_distance`: Large inter-event distances may be an indication that
  the *common Green's function assumption* is violated.

```{code-block} yaml
:caption: config.yaml
# Maximum difference in magnitude between two events to allow an amplitude
# measurement.
max_magnitude_difference:

# Maximum allowed distance (m) between two events.
max_event_distance:
```

The next parameters assure that only observations that contribute to a
meaningful solution enter the system of equations. `min_equations` and `max_gap`
are applied iteratively until no equations are left that violate either
constraint.

```{code-block} yaml
:caption: config.yaml
# Minimum number of equations required to constrain one moment tensor
min_equations:

# Maximum azimuthal gap allowed for one moment tensor
max_gap:
```

The next paramters allow to decimate the number of *S* observations. There are
$\binom{3}{N}$ possible *S* triplets, but only $\binom{N}{2}$ possible
*P* pairs. However, *S* observations should not necessarily dominate the
equation system. We therefore provide `max_s_equations` to exclude *S*
observations that are redundant in the sense that the events contributing to the
triplet have been observed many times and on stations that have many
observations. Observations are exluded until the threshold is reached.
`keep_events` is a list of events that should not be considered as redundant.
`equation_batches` controls how often the equations should be re-ranked.

```{code-block} yaml
:caption: config.yaml
# Maximum number of S-wave equation in the linear system. If more are available,
# iterativley discard those with redundant pair-wise observations, on stations
# with many observations, and with a higher misfit
max_s_equations:

# When reducing number of S-wave equations, increase importance of these events by
# not counting them in the redundancy score. Use to keep many equations e.g. for
# the reference event or specific events of interest.
keep_events:

# When reducing the number of S-wave equations, rank observations iteratively this
# many times by redundancy and remove the most redundant ones. A lower number is
# faster, but may result in discarding less-redundant observations.
equation_batches:
```

## 3. Applying quality control

When a set of parameters has been found, run

```{code-block} sh
:caption: shell
relmt qc
```

which will place the QCed files next to the original ones

```{code-block} none
:emphasize-lines: 13,14
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
    +-- P-amplitudes-qced.txt
    +-- S-amplitudes-qced.txt
```
