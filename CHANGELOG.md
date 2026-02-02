# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),

## [Unreleased]


## [0.4.0-beta] - 2026-02-02

### Added
- Example notebooks in documentation
- `relmt plot-mt` with many coloring and sorting options
- `extra.read_catalog_picks()` to import picks from Obspy Catalog object
- New configuration 'max_s_amplitude_misfit'

### Changed
- `relmt qc` renamed `relmt admit`
- `keep_other_s_equation` renamed `two_s_equations`
- `variable_name` renamed `matlab_variable`
- Improved `relmt plot-spectra`
- Improved `relmt plot-alignment`
- `utils.phse_dict_hash_plugin()` has new 'strict' option
- All `plot-` functions with option to save figures to file
- Default loglevel is CRITICAL

## [0.3.1-beta] - 2025-12-12

### Fixed

- `relmt plot alignment --highlight` handles absent events
- `relmt plot alignment` handles absent event and station file
- Improved documentation
- Fixed list formatting in Config documentation
- Improved log messages
- `relmt exclude` handles 'waveform' entry in exclude file

## [0.3.0-beta] - 2025-12-09

### Added

- Bug template
- Changelog
- Code of Conduct
- Initial versioned commit of the project
