# Examples

This page collects a number of examples that help you to get going with *relMT*.

* [Example 0](example_0_setup_external_Pamir_data) shows how an external event and moment tensor catalog is converted to *relMT* format. It downloads the data required for *example 1*.
* [Example 1a](example_1a_setup_Pamir_Muji_aftershocks) shows how a sensible, rather small (167 event) subset is chosen from the whole catalog. Seismic waveform data is downloaded, a phase table with take-off angles is written, and the waveform arrays and header files are created.

  The examples downloads about 600 MB of seismic data from [GEOFON](https://geofon.gfz.de/) and may run about 1h, depending on server latency. Re-running the notebook with downloaded data is a matter of minutes.
* [Example 1b](example_1b_align_Pamir_Muji_aftershocks.ipynb) demonstrates how the seismic waveforms are checked for similarity, how noisy or dissimilar event waveforms are excluded from further processing and how the similar waveforms are aligned, a prerequisite to measure relative amplitudes.

  On a well-equipped workstation with 20 CPU cores, the notebook runs in about 4 hours. It can be re-run in about a minute.
* [Example 1c](example_1c_solve_Pamir_Muji_aftershocks) exhibits a workflow to determine the optimal frequency band per event waveform, measure relative amplitudes of event combinations, construct the linear system of equations and solve for 143 relative moment tensors.

  This notebook takes a few minutes to complete
* [Example 1d](example_1d_interpret_Pamir_Muji_aftershocks) offers various ways to look at and interpret relative moment tensor data.

```{toctree}
:hidden:
example_0_setup_external_Pamir_data.ipynb
example_1a_setup_Pamir_Muji_aftershocks.ipynb
example_1b_align_Pamir_Muji_aftershocks.ipynb
example_1c_solve_Pamir_Muji_aftershocks.ipynb
example_1d_interpret_Pamir_Muji_aftershocks.ipynb
```
