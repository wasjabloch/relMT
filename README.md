![relMT-alpha Logo](images/relMT-alpha.png)

*Software to determine relative earthquake moment tensors*

***WARNING: This is an alpha release. The package is not yet complete.***

<!-- SPHINX-START -->

## Installation
We recommend to create a conda environment and activate it. Choose any Python version
greater or equal 3.10

```
# Create and activate the environment
conda create -n relmt python=3.10
conda activate relmt

# Install the Fastest Fourier Transform in the West
conda install -c conda-forge fftw
```

```
# Install relMT
pip install relmt
```

If you are working in `IPython`, or `Jupyter`, install the package in the same conda environment to avoid version conflicts

```
conda install ipython
```
or
```
conda install jupyter
```

## Documentation
The documentation is hosted at:

## Rationale

The algorithm consists of the following steps:

1. Choose a cluster of seismic events for which the Green's function can be assumed equal
2. Align P and S wave train observations to the sample
3. At each seismic station, decompose the wave trains into principal components
4. Measure relative P-wave and S-wave amplitudes
5. Set up a linear system that relates relative amplitudes to moment tensors
6. Solve the linear system using algebraic methods


## References
The algorithms are based on the following research articles:

Dahm, T., J. Horalek, and J. Sileny (2000). Comparison of absolute and relative
moment tensor solutions for the January 1997 West Bohemia Earthquake swarm.
Studia Geophys. et Geod., https://doi.org/10.1023/A:1022166926987

Plourde, A. P., and M. G. Bostock (2019). Relative moment tensors and deep
Yakutat seismicity, Geophys. J. Int., https://doi.org/10.1093/gji/ggz375

Bostock, M. G., A. P. Plourde, D. Drolet, and G. Littel (2021). Multichannel
alignment of S waves, Bull. Seismol. Soc. Am.,
https://doi.org/10.1785/0120210076

Drolet, D., M. G. Bostock, A. P. Plourde, and C. G. Sammis (2022). Aftershock
distributions, moment tensors and stress evolution of the 2016 Iniskin and 2018
Anchorage Mw 7.1 Alaskan intraslab earthquakes. Geophys. J. Int.,
https://doi.org/10.1093/gji/ggac165

Drolet, D., M.G. Bostock, and S. Peacock (2023). Relative Moment Tensor
Inversion for Microseismicity: Application to Clustered Earthquakes in the
Cascadia Forearc. Seismica. https://doi.org/10.26443/seismica.v2i4.1311


## Acknowledgments
This software package is part of the *relMT* project that has received funding
from the European Union’s Horizon Europe research and innovation programme under
the Marie Skłodowska-Curie grant agreement No. 101146483

![Funded by the EU](images/EN_FundedbytheEU_RGB_POS.png)
