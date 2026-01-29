# Admit amplitudes into the linear system

## 1. Prerequisites

The aim is to admit the "best" relative *P* and
*S* amplitudes from the potentially very large sets stored in
`P-amplitudes.txt` and `S-amplitudes.txt` into the linear system of equations
that constraints the relative moment tensors. We assume the following file
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

## 2. Admit parameters

The following parameters in `config.yaml` define the critera by which relative
amplitude observations should be excluded. As before, different parameter sets
are distinguished using `admit_suffix`, which is appended to the name of the
output files.

```{code-block} yaml
:caption: config.yaml

# Admission paramters
# -------------------------
#
# Suffix appended to the amplitude suffix, naming the admission parameters
# parsed to 'admit'
# (str)
admit_suffix: admitted
```

The first parameters pertain to the quality of waveform reconstruction:

* `max_amplitude_misfit` discribes how well a waveform is reconstructed, where a
  value of *0* indicates perfect reconstruction, *>1* indicates that the misfit
  amplitude is larger than the singal amplitude itself. This value should be *<1*.
* `max_s_amplitude_misfit` As `max_amplitude_misfit`, but sets a different
  value for *S*-waves.
* `max_s_sigma1` discribes the linear independence of the two relative *S*
  amplitudes. The values of $B_{abc}$ and $B_{acb}$ may become arbitrary as this
  value approaches *1* and the equation may not be linearily independent of
  others. However, in the case of very similar *S* waveforms, $\sigma_1$ may be
  close to *1* for almost all waveform combinations. One then runs into danger
  of excluding all observations when choosing too low a value.

```{literalinclude} config-template.yaml
:caption: config.yaml
:language: yaml
:start-at: Discard amplitude measurements
:end-at: max_s_sigma1:
```

The next parameters pertain to properties of event combinations.

* `max_magnitude_difference`: Large differences in event magnitude may cause
  problems in the relative amplitude measurement when the lowpass filter corner
  was chosen too high and the depletion in high frequency energy of the
  larger-magnitude event becomes significant.
* `max_event_distance`: Large inter-event distances may be an indication that
  the *common Green's function assumption* is violated.

```{literalinclude} config-template.yaml
:caption: config.yaml
:language: yaml
:start-at: Maximum difference in magnitude
:end-at: max_event_distance:
```

The next parameters assure that only observations that contribute to a
meaningful solution enter the system of equations.

* `min_equations` indicates that each relative moment tensor must be constrained
  by at least this many equations
* `max_gap` is the maximum azimuthal gap allowed for a moment tensor. If this
  value is exceeded, all ob

If one of the values is exceeded for a moment tensor, all observations
pertaining to that MT will be discarded. These criteria are applied iteratively
until no equations are left that violate either.

```{literalinclude} config-template.yaml
:caption: config.yaml
:language: yaml
:start-at: Minimum number of equations
:end-at: max_gap:
```

The next paramters allow to decimate the number of *S* observations. There are
$\binom{3}{N}$ possible *S* triplets, but only $\binom{N}{2}$ possible
*P* pairs. However, *S* observations should not necessarily dominate the
equation system.

* `two_s_equations` determines, if each *S* amplitude observation contributes
  two equations (of the two longest polarization vectors $g$ for the
  station-event configuration), or only one (of the longest polarization
  vector)
* `max_s_equations` allows to exclude *S* observations that are redundant in the
  sense that the events contributing to the triplet have been observed many
  times and on stations that have many observations. Observations are exluded
  until the threshold is reached. When set, `keep_events` is a list of events
  that should not be considered as redundant and `equation_batches` controls how
  often the equations should be re-ranked.

```{literalinclude} config-template.yaml
:caption: config.yaml
:language: yaml
:start-at: Use two equations per S-amplitude
:end-at: equation_batches:
```

## 3. Applying admission parameters

When a set of parameters has been found, run

```{code-block} sh
:caption: shell
relmt admit
```

which will place the admitted files next to the original ones

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
    +-- P-amplitudes-admitted.txt
    +-- S-amplitudes-admitted.txt
```
