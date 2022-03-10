# Common catchment processing workflows

This repository contains examples and documentation for some common workflows used by Catchment Processes (Section 317).

## Getting started

To run the example notebooks, login to [NIVA's JupyterHub](https://jupyterhub.niva.no/) and clone this repository. All packages and datasets should be pre-installed. Please contact James Sample if you have any problems.

**Note:** To avoid duplication, some of the example notebooks linked below (e.g. the one for HydAPI) are hosted in a different repository (the [dstoolkit_cookbook](https://nivanorge.github.io/dstoolkit_cookbook/)). You may need to clone that too.

## Bug reports and feature requests

Please use the [issue tracker](https://github.com/NIVANorge/catchment_processing_workflows/issues) to report problems or request new features. [Pull requests](https://github.com/NIVANorge/catchment_processing_workflows/pulls) are also welcome!

## 1. Catchment delineation

Delineate watersheds for outflow points in Norway based on Kartverket's national 10 m resolution DTM. Reduced resolution versions (20 m and 40 m) are also available for faster processing and will be suitable for many applications.

**To get started, see the example [here](https://nbviewer.org/github/NIVANorge/catchment_processing_workflows/blob/main/notebooks/catchment_delineation/04_catchment_delineation.ipynb).**

For reference, notebooks describing development of the workflow are here:

 1. [Reproject, merge and resample the raw DTM](https://nbviewer.org/github/NIVANorge/catchment_processing_workflows/blob/main/notebooks/catchment_delineation/01_merge_reproject_raw_dtm.ipynb)
 2. [Split data by vassdragsområde for efficiency](https://nbviewer.org/github/NIVANorge/catchment_processing_workflows/blob/main/notebooks/catchment_delineation/02_split_by_vassom.ipynb)
 3. [Terrain conditioning](https://nbviewer.org/github/NIVANorge/catchment_processing_workflows/blob/main/notebooks/catchment_delineation/03a_terrain_processing_pysheds.ipynb)

## 2. Land cover proportions

**To do**.

## 3. Accessing historic weather, climate and hydrological data

### 3.1. NVE's Grid Time Series (GTS) API

The [Grid Time Series API](http://api.nve.no/doc/gridtimeseries-data-gts/) provides the most up-to-date gridded weather and hydrology data available for **Norway**. It's part of [seNorge 2018](https://essd.copernicus.org/articles/11/1531/2019/) and offers daily data from 1957 to present with 1 km x 1 km spatial resolution. The GTS API includes a wide range of weather and hydrology variables and it's probably the best gridded historic data currently available *if your region of interest is entirely within Norway*.

**To get started, see the example [here](https://nbviewer.org/github/NIVANorge/catchment_processing_workflows/blob/main/notebooks/nve_gts_api_example.ipynb).**

### 3.2. Met.no's Norwegian Gridded Climate Dataset (NGCD)

Gridded precipitation and temperature (min, mean and max) data for **Norway, Sweden and Finland** are available from [Met.no's Thredds server](https://thredds.met.no/thredds/catalog/ngcd/catalog.html). The data have a daily time step (1971 to present) and a spatial resolution of 1 km x 1 km. Two variants are available: `Type1` (based on a residual kriging) and `Type2` (based on Bayesian spatial interpolation), which are part of seNorge v1 and seNorge v2, respectively. This means they are slightly older than the datasets availble via the GTS API (above), which is part of seNorge 2018 (see [here](https://github.com/metno/seNorge_docs/wiki) for more information). 

Note that querying data via Thredds can be slow due to bandwidth limitations imposed by Met.no. **Expect queries to take several hours**. For this reason, it is recommended to use the GTS API if possible: it is faster, newer, and includes more variables. The main advantage of NGCD is that it provides consistent data for Norway, Sweden and Finland.

**To get started, see the example [here](https://nbviewer.org/github/NIVANorge/catchment_processing_workflows/blob/main/notebooks/met_ngcd_thredds_example.ipynb).**

### 3.3. ERA5-Land via Copernicus

[ERA5-Land](https://www.ecmwf.int/en/era5-land) is a **global** reanalysis dataset providing a consistent view of the evolution of land variables over several decades at an enhanced resolution compared to ERA5. The dataset has a resolution of 0.1 by 0.1 degrees and is available from the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/#!/home) as either [hourly](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview) data or aggregated using [monthly](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means?tab=overview) averages. See the links for further details.

**An example notebook showing how to download and process ERA5-Land data is [here](https://nbviewer.org/github/NIVANorge/catchment_processing_workflows/blob/main/notebooks/era5land_cds_api_example.ipynb).**

### 3.4. Met.no's Frost API

Met.no's [Frost API](https://frost.met.no/index.html) provides access to observed data recorded by weather stations in Norway (e.g. daily, monthly and annual measurements of temperature, precipitation and wind speed).

**To do.**

### 3.5. NVE's HydAPI

NVE's [Hydrological API](https://hydapi.nve.no/UserDocumentation/) provides access to historical and real-time hydrological time series. 

**See the example notebook [here](https://nbviewer.org/github/NIVANorge/dstoolkit_cookbook/blob/master/notebooks/nve_hydapi_example.ipynb) for an introduction.**

