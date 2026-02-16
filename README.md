# relMT

![relMT-beta Logo](images/relMT-beta.png)

*Software to determine relative earthquake moment tensors*

<!-- SPHINX-START -->

## Installation
 <!-- INSTALLATION-START -->

Installation of *relMT* is easy. The basic installation requires:

* The *FFTW* library (often provided via the system package manager)
* The Python packages *NumPy*, *SciPy*, *PyYAML*

:::{note}
*relMT* is currently tested only on *Linux* systems.

*Windows* users, please use the [Linux Subsystem for Windows (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) and proceed with the instructions below.

*Mac* users will need to install the fortran compiler `gfortran` on your system.
Please consult the [Fortran
documentation](https://fortran-lang.org/learn/os_setup/install_gfortran/#macos)
for details. Then proceed below.
:::

### Pre-requisite

This installation instruction assumes installation with *Conda*, *pip* and
*git*. Experienced users may use *uv* or other tools instead.

As a pre-requisite, please [install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2).

### Based on Conda and Pip

We recommend to create a conda environment or to install into an existing one.
Choose any Python version greater or equal 3.10

```sh
# Create and activate the environment
conda create -n relmt python
conda activate relmt

# Install the Fastest Fourier Transform in the West
conda install -c conda-forge fftw
```

If not already present, install `pip`

```sh
conda install pip
```

If not already present, install `git`

```sh
conda install git
```

Now install *relMT* locally

```sh
git clone https://github.com/wasjabloch/relMT
cd relMT
pip install .
```

### Additional dependencies

For plotting, we require:

* *Matplotlib* for all plotting
* *networkx* to visualize connections of equations in the linear system
* *Pyrocko* to plot moment tensors

Consider installing these packages using the `plot` optional dependency:

```sh
pip install .[plot]
```

Some additional functionality requires community packages:

* Import of waveforms and station inventories via *ObsPy*
* Computation of spectra with *Multitaper*
* Conversion to and from Cartesian coordinates with *UTM*

Consider installing these packages using the `extra` optional dependency:

```sh
pip install .[extra]
```

:::{note}
Users experiencing problems installing *ObsPy* may [temporarily need to pin *setuptools* below version 82](https://discourse.obspy.org/t/obspy-runtests-fails-after-fresh-install/2353)

```sh
pip install "setuptools<82"
```

:::

If you are working in *IPython*, or *Jupyter*, install the package in the same Conda environment to avoid version conflicts

```sh
conda install ipython
```

or

```sh
conda install jupyter
```

If you consider contributing to *relMT*, please install the development version

```sh
pip install .[dev]
```

<!-- INSTALLATION-END -->
## Documentation

The documentation is hosted at: <https://relmt.readthedocs.io/en/latest/>

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
Studia Geophys. et Geod., <https://doi.org/10.1023/A:1022166926987>

Plourde, A. P., and M. G. Bostock (2019). Relative moment tensors and deep
Yakutat seismicity, Geophys. J. Int., <https://doi.org/10.1093/gji/ggz375>

Bostock, M. G., A. P. Plourde, D. Drolet, and G. Littel (2021). Multichannel
alignment of S waves, Bull. Seismol. Soc. Am.,
<https://doi.org/10.1785/0120210076>

Drolet, D., M. G. Bostock, A. P. Plourde, and C. G. Sammis (2022). Aftershock
distributions, moment tensors and stress evolution of the 2016 Iniskin and 2018
Anchorage Mw 7.1 Alaskan intraslab earthquakes. Geophys. J. Int.,
<https://doi.org/10.1093/gji/ggac165>

Drolet, D., M.G. Bostock, and S. Peacock (2023). Relative Moment Tensor
Inversion for Microseismicity: Application to Clustered Earthquakes in the
Cascadia Forearc. Seismica. <https://doi.org/10.26443/seismica.v2i4.1311>

## Acknowledgments

This software package is part of the *relMT* project that has received funding
from the European Union’s Horizon Europe research and innovation programme under
the Marie Skłodowska-Curie grant agreement No. 101146483

![Funded by the EU](images/EN_FundedbytheEU_RGB_POS.png)
