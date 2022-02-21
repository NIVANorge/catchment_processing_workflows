# Common catchment processing workflows

This repository contains examples and documentation for some common workflows used by Catchment Processes (Section 317).

## Getting started

To run the example notebooks, simply login to [NIVA's JupyterHub](https://jupyterhub.niva.no/) and clone this repository (all packages and datasets should be pre-installed). Please contact James Sample if you have any problems.

## Bug reports and feature requests

Please use the [issue tracker](https://github.com/NIVANorge/catchment_processing_workflows/issues) to report problems or request new features. [Pull requests](https://github.com/NIVANorge/catchment_processing_workflows/pulls) are also welcome!

## Contents

 1. [Catchment delineation](https://github.com/NIVANorge/catchment_processing_workflows#1-catchment-delineation)
 2. [Land cover proportions](https://github.com/NIVANorge/catchment_processing_workflows#2-land-cover-proportions)
 3. [Accessing historic weather, climate and hydrological data](https://github.com/NIVANorge/catchment_processing_workflows#3-accessing-historic-weather-climate-and-hydrological-data)
    1. [NVE's Grid Time Series API](https://github.com/NIVANorge/catchment_processing_workflows#31-nves-grid-time-series-gts-api)
    2. [Met.no's Norwegian Gridded Climate Dataset](https://github.com/NIVANorge/catchment_processing_workflows#32-metnos-norwegian-gridded-climate-dataset-ngcd)
    3. [ERA5 via Copernicus](https://github.com/NIVANorge/catchment_processing_workflows#33-era5-via-copernicus)
    4. [Met.no's Frost API](https://github.com/NIVANorge/catchment_processing_workflows#34-metnos-frost-api)
    5. [NVE's HydAPI](https://github.com/NIVANorge/catchment_processing_workflows#35-nves-hydapi)

## 1. Catchment delineation

**To do**.

## 2. Land cover proportions

**To do**.

## 3. Accessing historic weather, climate and hydrological data

### 3.1. NVE's Grid Time Series (GTS) API

The Grid Time Series API provides the most up-to-date gridded weather and hydrology data available for Norway. It's part of [seNorge 2018](https://essd.copernicus.org/articles/11/1531/2019/) and offers daily data from 1957 to present with 1 km x 1 km spatial resolution. The GTS API includes a wide range of weather and hydrology variables and it's probably the best gridded historic data currently available *if your region of interest is entirely within Norway*.

To get started, see the example [here](https://nbviewer.org/github/NIVANorge/catchment_processing_workflows/blob/main/notebooks/nve_gts_api_example.ipynb).

### 3.2. Met.no's Norwegian Gridded Climate Dataset (NGCD)

Gridded precipitation and temperature (min, mean and max) for **Norway, Sweden and Finland** are available from [Met.no's Thredds server](https://thredds.met.no/thredds/catalog/ngcd/catalog.html). The data have a daily time step (1971 to present) and a spatial resolution of 1 km x 1 km. Two variants are available: `Type1` (based on a residual kriging) and `Type2` (based on Bayesian Spatial Interpolation), which are part of seNorge v1 and seNorge v2, respectively. This means they are slightly older than the datasets availble via the GTS API (above), which is part of seNorge 2018 (see [here](https://github.com/metno/seNorge_docs/wiki) for more information). 

Note that querying data via Thredds can be **slow**, due to bandwidth limitations imposed by Met.no. **Expect queries to take several hours**. For this reason, it is recommended to use the GTS API if possible, since it is faster, newer, and includes more variables. The main advantage of NGCD is that it provides consistent data for Norway, Sweden and Finland.

### 3.3. ERA5 via Copernicus


### 3.4. Met.no's Frost API

Met.no's [Frost API](https://frost.met.no/index.html) provides access to observed data recorded by weather stations in Norway (e.g. daily, monthly and annual measurements of temperature, precipitation and wind speed).

### 3.5. NVE's HydAPI

NVE's [Hydrological API](https://hydapi.nve.no/UserDocumentation/) provides access to historical and real-time hydrological time series. See the example notebook [here](https://nbviewer.org/github/NIVANorge/dstoolkit_cookbook/blob/master/notebooks/nve_hydapi_example.ipynb) for an introduction.

