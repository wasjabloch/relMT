# Synopsis of processing steps

This guide gives a brief overview over the steps necessary to compute relative
moment tensors.

The algorithm is broken down in 5 steps:

1. [Initialize](setup) a project directory (`relmt init`) and create the input
   files
2. [Iteratively align](align), exclude and re-align event waveforms
   (`relmt align` and `relmt exclude`)
3. Determine optimal bandpass windows and [measure relative amplitudes](amplitude) (`relmt amplitude`)
4. [Admit](admit) a subset of amplitudes into the linear system of equations (`relmt admit`)
5. [Create the linear system](solve) of equations with a reference MT, apply
   weighting, and solve the equation system (`relmt solve`)

See the detailed description below to follow them or try out an [example](../examples/index).
