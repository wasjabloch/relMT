![relMT-alpha Logo](images/relMT-alpha.png)

*Software package to determine relative earthquake moment tensors*

***WARNING: This is an alpha release. The package is not yet complete.***

# Installation

1. Install `fftw` using the package manager of your system. Alternativley, download and install it from the source (https://www.fftw.org/download.html) and place it "somewhere where CMake can find it".

2. Create a conda environment and activate it. Choose any Python version greater 3.9

```
conda create -n relmt python=3.9
conda activate relmt
```

3. Clone this repository:

```
git clone https://github.com/wasjabloch/relMT
```

4. Install `relMT` and its dependencies using pip

```
cd relMT
pip install .
```

5. If you are working in ipython, install it in the environment to avoid version conflicts
```
conda install ipython
```

# Rationale

The algorithm consists of the following steps:

1. Choose clusters of seismic events for which the Green's function can be assumed equal
2. Align P and S wave train observations to the sample
3. At each seismic station, decompose the wave trains into principal components
4. Measure relative amplitudes from the normalized singular values and vectors
5. Set up the linear system that relates relative amplitudes to moment tensors
6. Solve the linear system using algebraic methods

The algorithms are based on the following research articles:

Dahm, T., J. Horalek, and J. Sileny (2000). Comparison of absolute and relative
moment tensor solutions for the January 1997 West Bohemia Earthquake swarm.
Studia Geophys. et Geod., https://doi.org/10.1023/A:1022166926987

Plourde, A. P., and M. G. Bostock (2019). Relative moment tensors and deep
Yakutat seismicity, Geophys. J. Int., https://doi.org/10.1093/gji/ggz375

Bostock, M. G., A. P. Plourde, D. Drolet, and G. Littel (2021). Multichannel alignment of S waves, Bull. Seismol. Soc. Am., https://doi.org/10.1785/0120210076

Drolet, D., M. G. Bostock, A. P. Plourde, and C. G. Sammis (2022). Aftershock
distributions, moment tensors and stress evolution of the 2016 Iniskin and 2018
Anchorage Mw 7.1 Alaskan intraslab earthquakes. Geophys. J. Int.,
https://doi.org/10.1093/gji/ggac165

# Acknowledgments
This software package is part of the *relMT* project that has received funding
from the European Union’s Horizon Europe research and innovation programme under
the Marie Skłodowska-Curie grant agreement No. 101146483

![Funded by the EU](images/EN_FundedbytheEU_RGB_POS.png)