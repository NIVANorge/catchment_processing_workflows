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
    2.	[Met.no's Norwegian Gridded Climate Dataset](https://github.com/NIVANorge/catchment_processing_workflows#32-metnos-norwegian-gridded-climate-dataset-ngcd)
    3. [ERA5 via Copernicus](https://github.com/NIVANorge/catchment_processing_workflows#33-era5-via-copernicus)
    4. [Met.no's Frost API](https://github.com/NIVANorge/catchment_processing_workflows#34-metnos-frost-api)
    5. [NVE's HydAPI](https://github.com/NIVANorge/catchment_processing_workflows#35-nves-hydapi)

## 1. Catchment delineation

**To do**.

 * [Delineate watersheds within Norway]() by supplying a dataframe of outlet co-ordinates.

## 2. Land cover proportions

**To do**.

 * [Calculate land cover proportions]() for regions in Norway by supplying a geodataframe of polygons. Available land cover datasets include AR50 and Corine.

## 3. Accessing historic weather, climate and hydrological data

### 3.1. NVE's Grid Time Series (GTS) API

The most up-to-date gridded weather and hydrology data available for Norway (part of [seNorge 2018](https://essd.copernicus.org/articles/11/1531/2019/)). Daily data from 1957 to present with 1 km x 1 km spatial resolution. Includes a wide range of weather and hydrology variables. Probably the best gridded historic data currently available if your region of interest is **entirely within Norway**.

### 3.2. Met.no's Norwegian Gridded Climate Dataset (NGCD)

Gridded precipitation and temperature (min, mean and max) data for **Norway, Sweden and Finland** from [Met.no's Thredds server](https://thredds.met.no/thredds/catalog/ngcd/catalog.html). Data have a daily time step (1971 to present) and spatial resolution of 1 km x 1 km. Two variants are available: `Type1`, based on a residual kriging, and `Type2`, based on Bayesian Spatial Interpolation. These datasets are part of seNorge v1 and seNorge v2, respectively, so they are slightly older than datasets availble via the GTS API, which is part of seNorge 2018 (see [here](https://github.com/metno/seNorge_docs/wiki) for more information). 

Querying data via Thredds can be **slow** - expect queries to take several hours. It is recommended to use the GTS API if possible, because it is faster, more recent, and includes more variables. The main advantage of NGCD is that it provides consistent data for Norway, Sweden and Finland.

### 3.3. ERA5 via Copernicus


### 3.4. Met.no's Frost API

Met.no's [Frost API](https://frost.met.no/index.html) provides access to observed data recorded by weather stations in Norway (e.g. daily, monthly and annual measurements of temperature, precipitation and wind speed).

### 3.5. NVE's HydAPI

NVE's [Hydrological API](https://hydapi.nve.no/UserDocumentation/) provides access to historical and real-time hydrological time series.

