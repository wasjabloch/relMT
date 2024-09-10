![relMT-alpha Logo](images/relMT-alpha.png)

*Software package to determine relative earthquake moment tensors*

***WARNING: This is an alpha release. The package is not yet complete.***

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