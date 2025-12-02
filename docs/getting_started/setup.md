# Setting up a project

## Set up the input files

1. Set up a directory for one cluster of seismic events you wish to investigate

2. In that directory, create the following subdirectories: ``data/``,
``align/``, ``amplitude/``, ``result/``

3. Launch a python interpreter and run the commands:

    ```python
    from relmt import core, io
    import yaml

    core.Config().to_file(core.file("config"))
    core.Header().to_file(core.file("waveform_header"))

    io.make_station_table({}, core.file("station"))
    io.make_event_table([], core.file("event"))
    io.make_phase_table({}, core.file("phase"))
    io.make_mt_table({}, core.file("reference_mt"))

    with open(core.file("exclude"), "w") as fid:
        yaml.safe_dump(core.exclude, fid)
    ```

    This creates the following structure of directories and template files:

    ```none
    my_relmt_project/
    +-- config.yaml
    +-- exclude.yaml
    +-- data/
    |   +-- stations.txt
    |   +-- events.txt
    |   +-- phases.txt
    |   +-- reference_mt.txt
    |   +-- default-hdr.yaml
    +-- align/
    +-- amplitude/
    +-- result/
    ```

4. Open each of the files and fill them with the information of your project.
See [file formats](project:./formats.md) for details.

    :::{tip}
    There are several functions that may be useful when creating the files from external resources.

    * To create a station file from an *ObsPy* `Inventory` object
        * {py:class}`relmt.extra.read_obspy_inventory_files`
        * {py:class}`relmt.extra.read_station_inventory`

    * To create an event file from an external table
        * {py:class}`relmt.io.read_ext_event_table`

    * To create a phase file from a *NonLinLoc* `.hyp` file
        * {py:class}`relmt.io.read_phase_nll_hypfile`

    * To create a reference moment tensor file from an external moment tensor table
        * {py:class}`relmt.io.read_ext_mt_table`

    * To create waveform arrays from an *ObsPy* `Stream`
        * {py:class}`relmt.extra.make_waveform_array`
    :::

5. From the `default-hdr.yaml`, create the waveform array files and waveform headers.

### With a subsection
