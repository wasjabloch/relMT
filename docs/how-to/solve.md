# Solve the linear system of equations

## 1. Synopsis

After [amplitude admission](admit.md), each line of the *P* amplitude file represents
one equation of the linears system and each line of the *S* amplitude file either one or two (depending on the `two_s_equations` parameter).

We now combine the amplitdues with the take-off angles stored in the [phase
file](#phase-file) and the reference MT stored in the [reference MT
file](#reference-mt-file) into a linear system of equations. We apply a
weighting scheme as to counter-act large amplitude differences and honour
variable data quality. We further show how to apply model constraints. Finally,
we solve the system in a least square sense and show how to detect outliers
trough the bootstrapping method.

## 2. Defining the reference MT

As before, the parameter sets can be made distinguishable by defining
`result_suffix`, which will be appended to the filenames created here.

```{literalinclude} config-template.yaml
:caption: config.yaml
:language: yaml
:start-at: Solve parameters
:end-at: result_suffix:
```

One or multiple reference MTs can be inserted into the right hand side of the
equation system. The event number must correspond to an entry in the [event
file](#event-file) and the [reference MT file](#reference-mt-file).
We usually apply a weight of *1000* to the reference MT, which forces that
referece event to attain the reference MT. Note that the linear system of
equations is normalized so that parameters are in the *-1* to *1* range.

```{code-block} yaml
:caption: config.yaml
# Event indices of the reference moment tensor(s) to use
# (list)
reference_mts: [0] # For example

# Weight of the reference moment tensor
# (float)
reference_weight: 1000
```

## 3. Applying a constraint

A deviatoric or no constraint (i.e. full MT) can be applied to the solutions,
meaning that we either solve for *5* or *6* MT elements. Note that when a
deviatoric constraint is applied, we do not consider the isotropic part of the
reference MT and the resulting magnitudes will be lower.

```{literalinclude} config-template.yaml
:caption: config.yaml
:language: yaml
:start-at: Constrain the moment tensor
:end-at: mt_constraint:
```

## 4. Weighting observations by misfit

To emphasize better observations, those with a lower misfit are given a larger
weight (i.e. unity), which is assigned to all observations with a misfit lower
than `min_amplitude_misfit`. To avoid that observation with a misfit approaching
the largest allowed misfit (`max_misfit`, see [admit amplitudes](admit.md)) get a
weight close to *0*, one can set the `min_amplitude_weight`.

```{literalinclude} config-template.yaml
:caption: config.yaml
:language: yaml
:start-at: Minimum misfit to assign
:end-at: min_amplitude_weight:
```

## 5. Drawing bootstrap samples

To detect oulying observation one can draw bootstrap samples. This will create
an additional relative MT file wiht a *"-boot"* suffix.

```{literalinclude} config-template.yaml
:caption: config.yaml
:language: yaml
:start-at: Number of samples to draw
:end-at: bootstrap_samples:
```

## 6. Calling the solve routine

The solution is calculated by calling.

```{code-block} sh
:caption: shell
relmt solve
```

which will read the amplitude files with the combined `amplitude_suffix` and
`admit_suffix` defined in the `config.py` and write the solution to the `result/`
subdirectory, with the `result_suffix` appended.

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
    +-- ...
+-- result/
    +-- relative_mts.txt
    +-- relative_mts-boot.txt  # Optional bootstrapping results
```

## 7. Computing synthetic amplitudes

Amplitude predictions for the found solution are computed using the `--predict`
argument. In that case, we also need to specify which original waveform
observations the synthetic amplitudes should be compared to. That is, when the
amplitudes were measured on the waveforms stored in `align2`, we supply the `-a
2` argument

```{code-block} sh
:caption: shell
relmt solve --predict -a 2
```
