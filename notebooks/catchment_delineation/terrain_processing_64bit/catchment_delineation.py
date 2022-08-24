import json
import logging
import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import nivapy3 as nivapy
import numpy as np
import rasterio
import rasterio.mask
from rasterio import features
from scipy import ndimage
from shapely.geometry import Polygon, box, shape
from skimage.morphology import skeletonize
from tqdm.notebook import tqdm


def derive_watershed_boundaries(
    df,
    id_col="station_code",
    xcol="longitude",
    ycol="latitude",
    crs="epsg:4326",
    min_size_km2=None,
    dem_res_m=40,
    buffer_km=None,
    temp_fold=None,
    reproject=True,
):
    """Calculate watershed boundaries in Norway based on outflow co-ordinates provided
    as a dataframe.

    Args
        df:           Dataframe. Containing co-ordinates for catchment outflow points
        id_col:       Str. Name of column in 'df' containing a unique ID for each
                      outflow point. This will be used to link derived catchments
                      to the original points. Must be unique
        xcol:         Str. Name of column in 'df' containing 'eastings' (i.e. x or
                      longitude)
        ycol:         Str. Name of column in 'df' containing 'northings' (i.e. y or
                      latitude)
        crs:          Str. A valid co-ordinate reference system for Geopandas. Most
                      easily expressed using EPSG codes e.g. 'epsg:4326' for WGS84
                      lat/lon, or 'epsg:25833' for ETRS89 UTM zone 33N etc. See
                          https://epsg.io/
        min_size_km2: Int, Float or None. Default None. If None, the catchment is derived
                      upstream of the exact cell containing the specified outflow point.
                      If the provided co-ordinates do not exactly match the stream
                      location, this may result in small/incorrect catchments being
                      delineated. Setting 'min_size_km2' will snap the outflow point to
                      the nearest cell with an upstream catchment area of at least this
                      many square kilometres. It is usually a good idea to explicitly set
                      this parameter
        dem_res_m:    Int. Default 40. Resolution of elevation model to use. One of
                      [10, 20, 40]. Smaller values give better cacthments but take
                      longer to process
        buffer_km:    Int, Float or None. Default None. If None, the code will search
                      the entire vassdragsområde. This is a good default, but it can be
                      slow and memory-intensive. Setting a value for this parameter will
                      first subset the DEM to a square region centred on the outflow point
                      with a side length of (2*buffer_km) kilometres. E.g. if you know your
                      catchments do not extend more than 20 km in any direction from the
                      specified outflow points, set 'buffer_km=20'. This will significantly
                      improve performance
        temp_fold:    Str. Default None. Only used if 'buffer_km' is specified. Must be a
                      path to a folder on 'shared'. Will be used to store temporary files
                      (which are deleted at the end)
        reproject:    Bool. Default True. Whether to reproject the derived catchments
                      back to the original 'crs' that the outflow points were provided
                      in. If False, catchments are returned in the CRS of the underlying
                      DEM, which is ETRS89-based UTM zone 33N (EPSG 25833)

    Returns
        GeoDataframe of catchments.
    """
    # Import here as the Numba imports/compilation are slow and it makes using NivaPy for
    # non-catchment workflows annoying
    from pysheds.grid import Grid

    method = "pysheds"

    # Check user input
    assert len(df[id_col].unique()) == len(df), "ERROR: 'id_col' is not unique."
    assert dem_res_m in [
        10,
        20,
        40,
    ], "ERROR: 'dem_res_m' must be one of [10, 20, 40]."
    if min_size_km2:
        assert isinstance(
            min_size_km2, (float, int)
        ), "'min_size_km2' must be an integer, float or None."
    if buffer_km:
        assert isinstance(
            buffer_km, (int, float)
        ), "'buffer_km' must be an integer, float or None."

        assert isinstance(
            temp_fold, (str)
        ), "'temp_fold' is required when 'buffer_km' is specified."

        shared_path = Path("/home/jovyan/shared")
        child_path = Path(temp_fold)
        assert (
            shared_path in child_path.parents
        ), "'temp_fold' must be a folder on the 'shared' drive."

    if temp_fold:
        assert buffer_km, "'buffer_km' is required when 'temp_fold' is specified."
        work_dir = os.path.join(temp_fold, "cat_delin_temp")
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    # Build geodataframe and reproject to CRS of DEM
    dem_crs = "epsg:25833"
    gdf = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df[xcol], df[ycol], crs=crs)
    )
    gdf = gdf.to_crs(dem_crs)
    gdf["x_proj"] = gdf["geometry"].x
    gdf["y_proj"] = gdf["geometry"].y

    # Get vassdragsområder
    eng = nivapy.da.connect_postgis()
    vass_gdf = nivapy.da.read_postgis("physical", "norway_nve_vassdragomrade_poly", eng)
    vass_gdf = vass_gdf.to_crs(dem_crs)

    # Assign points to vassdragsområder
    gdf = gpd.sjoin(
        gdf, vass_gdf[["vassdragsomradenr", "geom"]], predicate="intersects", how="left"
    )
    n_cats = len(gdf)
    gdf.dropna(subset=["vassdragsomradenr"], inplace=True)
    if len(gdf) != n_cats:
        msg = "Some outlet locations could not be assigned to a vassdragsområde. These will be skipped."
        warnings.warn(msg)

    # Loop over vassdragsområder
    cat_ids = []
    cat_geoms = []
    for vassom in tqdm(
        sorted(gdf["vassdragsomradenr"].unique()), desc="Looping over vassdragsområder"
    ):
        if method == "pysheds":
            dirmap = (1, 2, 3, 4, 5, 6, 7, 8)
            vassom_fdir_path = (
                f"/home/jovyan/shared/01_datasets/spatial/dtm_merged_utm33/dtm_{dem_res_m}m"
                f"/by_vassom/flow_direction/vassom_{vassom}_{dem_res_m}m_fdir.tif"
            )
            vassom_facc_path = (
                f"/home/jovyan/shared/01_datasets/spatial/dtm_merged_utm33/dtm_{dem_res_m}m"
                f"/by_vassom/flow_accumulation/vassom_{vassom}_{dem_res_m}m_facc.tif"
            )
        elif method == "wbt":
            dirmap = (128, 1, 2, 4, 8, 16, 32, 64)
            vassom_fdir_path = (
                f"/home/jovyan/shared/01_datasets/spatial/dtm_merged_utm33/dtm_{dem_res_m}m"
                f"/by_vassom/wbt_fdir/vassom_{vassom}_{dem_res_m}m_fdir.tif"
            )
            vassom_facc_path = (
                f"/home/jovyan/shared/01_datasets/spatial/dtm_merged_utm33/dtm_{dem_res_m}m"
                f"/by_vassom/wbt_facc/vassom_{vassom}_{dem_res_m}m_facc.tif"
            )
        else:
            raise ValueError("Method not valid.")

        if buffer_km is None:
            # Read the full grids in outer loop
            fdir_grid = Grid.from_raster(vassom_fdir_path)
            fdir = fdir_grid.read_raster(vassom_fdir_path)
            facc_grid = Grid.from_raster(vassom_facc_path)
            facc = facc_grid.read_raster(vassom_facc_path)

        # Loop over points in each vassdragsområde
        pts_vass_gdf = gdf.query("vassdragsomradenr == @vassom")
        for idx, row in tqdm(
            pts_vass_gdf.iterrows(),
            total=pts_vass_gdf.shape[0],
            desc=f"Looping over outlets in vassdragsområder {vassom}",
            # leave=False,
        ):
            cat_id = row[id_col]
            x = row["x_proj"]
            y = row["y_proj"]

            # Subset rasters for this point, if desired
            if buffer_km:
                xmin = x - (buffer_km * 1000)
                xmax = x + (buffer_km * 1000)
                ymin = y - (buffer_km * 1000)
                ymax = y + (buffer_km * 1000)
                bbox = (xmin, ymin, xmax, ymax)

                fdir_temp = os.path.join(work_dir, "fdir.tif")
                clip_raster_to_bounding_box(vassom_fdir_path, fdir_temp, bbox)

                facc_temp = os.path.join(work_dir, "facc.tif")
                clip_raster_to_bounding_box(vassom_facc_path, facc_temp, bbox)

                fdir_grid = Grid.from_raster(fdir_temp)
                fdir = fdir_grid.read_raster(fdir_temp)
                facc_grid = Grid.from_raster(facc_temp)
                facc = facc_grid.read_raster(facc_temp)

            # Snap outflow if desired
            if min_size_km2:
                # Convert min area to number of pixels
                acc_thresh = int((min_size_km2 * 1e6) / (dem_res_m**2))
                x_snap, y_snap = facc_grid.snap_to_mask(facc > acc_thresh, (x, y))
            else:
                x_snap, y_snap = x, y

            # Delineate catchment
            fdir = fdir.astype(np.int32)
            fdir.nodata = int(fdir.nodata)
            catch = fdir_grid.catchment(
                x=x_snap,
                y=y_snap,
                fdir=fdir,
                dirmap=dirmap,
                xytype="coordinate",
            )

            # Create a vector representation of the catchment mask
            catch_view = fdir_grid.view(catch, dtype=np.uint8)
            shapes = fdir_grid.polygonize(catch_view)
            for shapedict, value in shapes:
                # 'value' is 1 for the catchment and 0 for "not the catchment"
                if value == 1:
                    cat_ids.append(cat_id)
                    cat_geoms.append(shape(shapedict))

    if temp_fold:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    res_gdf = gpd.GeoDataFrame({id_col: cat_ids, "geometry": cat_geoms}, crs=dem_crs)
    res_gdf = res_gdf.merge(df, on=id_col)
    res_gdf = res_gdf[list(df.columns) + ["geometry"]]

    res_gdf.geometry = res_gdf.geometry.apply(lambda p: remove_polygon_holes(p))
    res_gdf = res_gdf.dissolve(by=id_col).reset_index()

    if reproject:
        res_gdf = res_gdf.to_crs(crs)

    return res_gdf


def remove_polygon_holes(poly):
    """Delete polygon holes by limitation to the exterior ring.

    https://stackoverflow.com/a/61466689/505698

    Args
        poly: Input shapely Polygon

    Returns
        Polygon
    """
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly


def get_features(gdf):
    """Helper function for clip_raster_to_gdf(). Converts 'gdf' to the format required
       by rasterio.mask.

    Args:
        gdf: Geodataframe. Must be of (multi-)polygon geometry type

    Returns:
        List of geometries.
    """
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


def clip_raster_to_bounding_box(raster_path, out_gtiff, bounding_box):
    """Clip a raster dataset to a bounding box and save the result
       as a new GeoTiff.

    Args:
        raster_path:  Str. Path to input raster dataset
        out_gtiff:    Str. Name and path of GeoTiff to be created. Should have a '.tif'
                      file extension
        bounding_box: Tuple. (xmin, ymin, xmax, ymax) in the same co-ordinate system as
                      'raster_path'

    Returns:
        None. The new raster is saved to the specified location.
    """
    # Read raster
    ras = rasterio.open(raster_path)
    crs = ras.crs.data["init"]

    bbox = box(*bounding_box)
    clip_gdf = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=crs)

    # Apply mask
    shapes = get_features(clip_gdf)
    out_image, out_transform = rasterio.mask.mask(ras, shapes, crop=True)
    out_meta = ras.meta
    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    # Save result
    with rasterio.open(out_gtiff, "w", **out_meta) as dest:
        dest.write(out_image)

    ras.close()


def get_elvis_streams_as_shapes(crs="epsg:25833"):
    """Get the ELVIS river network from PostGIS and convert to a format
    suitable for rasterisation.

    Args
        crs: Str. Valid CRS string for geopandas

    Returns
        Tuple of tuples. Geometries for rasterising
    """
    eng = nivapy.da.connect_postgis()
    riv_gdf = nivapy.da.read_postgis(
        "physical", "norway_nve_elvis_river_network_line", eng
    )
    riv_gdf = riv_gdf.to_crs(crs)
    shapes = ((geom, 1) for geom in riv_gdf.geometry)

    return shapes


def get_nve_lakes_as_shapes(crs="epsg:25833"):
    """Get NVE lake polygons from PostGIS and convert them to a format for
    rasterisation.

    Args
        crs: Str. Valid CRS string for geopandas

    Returns
        Tuple of tuples. Geometries for rasterising
    """
    eng = nivapy.da.connect_postgis()
    lake_gdf = nivapy.da.read_postgis("physical", "norway_nve_innsjo_poly", eng)
    lake_gdf = lake_gdf.to_crs(crs)
    shapes = ((geom, 1) for geom in lake_gdf.geometry)

    return shapes


def pysheds_array_to_raster(data, base_tiff, out_tiff, dtype, nodata):
    """Pysheds doesn't seem to write array metadata correctly (especially the CRS
    information). This function takes result arrays from pysheds, but saves them
    using rasterio.

    Args
        data:      Array-like. Data to save
        base_tiff: Str. Path to TIFF with same properties/profile as desired output
        out_tiff:  Str. Path to TIFF to be created
        dtype:     Obj. Dtype for output array
        nodata:    Int. No data value for output array

    Returns
        None. Array is saved to disk.
    """
    with rasterio.open(base_tiff) as src:
        out_meta = src.meta.copy()
    data = data.astype(dtype)
    out_meta.update(
        {
            "driver": "GTiff",
            "dtype": dtype,
            "compress": "lzw",
            "nodata": nodata,
            "BIGTIFF": "IF_SAFER",
        }
    )
    with rasterio.open(out_tiff, "w", **out_meta) as dest:
        dest.write(data, 1)


def burn_stream_shapes(grid, dem, shapes, dz, sigma=None):
    """Burn streams represented by 'shapes' into 'dem'. The burn depth
    is 'dz' and Gaussian blur is applied with std. dev. 'sigma'. 'grid'
    is used to provide the transform etc. for the output raster.

    Args
        grid:   Obj. Pysheds grid object
        dem:    Obj. Pysheds raster object
        shapes: Tuple of obj. Stream shapes
        dz:     Float or int. Burn depth
        sigma:  Float or None. Default None. Std. dev. for (optional)
                Gaussian blur

    Returns
        Pysheds Raster object (essentially a numpy array)
    """
    if sigma:
        assert isinstance(
            sigma, (int, float)
        ), "'sigma' must be of type 'float' or 'int'."

    stream_raster = features.rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.affine,
        all_touched=False,
    )
    stream_raster = skeletonize(stream_raster).astype(np.uint8)
    mask = stream_raster.astype(bool)

    if sigma:
        # Blur mask using a gaussian filter
        blurred_mask = ndimage.filters.gaussian_filter(
            mask.astype(np.float32), sigma=sigma
        )

        # Set central river channel to max of Gaussian to prevent pits, then normalise
        # s.t. max blur = 1 (and decays to 0 in Gaussian fashion either side)
        blur_max = blurred_mask.max()
        # blurred_mask[mask.astype(bool)] = blur_max
        blurred_mask = blurred_mask / blur_max
        mask = blurred_mask

    # Burn streams
    dem[(mask > 0)] = dem[(mask > 0)] - (dz * mask[(mask > 0)])

    return dem


def burn_lake_shapes(grid, dem, shapes, dz):
    """Burn streams represented by 'shapes' into 'dem'. The burn depth
    is 'dz'.

    'grid' is used to provide the transform etc. for the output raster.

    Args
        grid:   Obj. Pysheds grid object
        dem:    Obj. Pysheds raster object
        shapes: Tuple of obj. Lake shapes
        dz:     Float or int. Burn depth

    Returns
        Pysheds Raster object (essentially a numpy array)
    """
    lake_raster = features.rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.affine,
        all_touched=False,
    )
    mask = lake_raster.astype(bool)
    dem[(mask > 0)] = dem[(mask > 0)] - (dz * mask[(mask > 0)])

    return dem


def condition_dem(
    raw_dem_path,
    fill_dem_path,
    fdir_path,
    facc_path,
    dem_dtype=np.int16,
    dem_ndv=-32767,
    burn=False,
    stream_shapes=None,
    lake_shapes=None,
    stream_sigma=None,
    stream_dz=None,
    lake_dz=None,
    max_iter=1e9,
    eps=1e-12,
):
    """Burns streams into DEM, fills pits and depressions, and calculates flow
    direction and accumulation. The filled DEM is converted to the specificed
    dtype and saved. Flow direction and accumulation rasters are also saved.

    Args
        raw_dem_path:  Str. Path to raw DEM to be processed
        fill_dem_path: Str. Output path for filled (and optionally burned) DEM
        fdir_path:     Str. Output path for flow direction
        facc_path:     Str. Output path for flow accumulation
        dem_dtype:     Obj. Numpy data type for output DEM
        dem_ndv:       Float or int. NoData value for output DEM
        burn:          Bool. Whether to burn streams
        stream_shapes: Tuple of tuples. Stream shapes to burn. Only valid if
                       'burn' is True
        lake_shapes:   Tuple of tuples. Lake shapes to burn. Only valid if
                       'burn' is True
        stream_sigma:  Float. Std. dev. for Gaussian blur applied to streams.
                       Only valid if 'burn' is True
        stream_dz:     Float or int. Stream burn depth. Only valid if 'burn' is
                       True
        lake_dz:       Float or int. Lake burn depth. Only valid if 'burn' is
                       True
        max_iter:      Int. Default 1e9. Maximum iterations for filling flats
        eps:           Float. Default 1e-12. Parameter for flat-filling
                       algorithm. See example notebook

    Returns
        Tuple of PySheds 'Raster' arrays (dem, fdir, facc).
    """
    # from pysheds.grid import Grid
    from fill64 import Grid64 as Grid

    # Check user input
    if burn:
        assert None not in (
            stream_shapes,
            lake_shapes,
            stream_dz,
            lake_dz,
        ), "'stream_shapes', 'lake_shapes', 'stream_dz' and 'lake_dz' are all required when 'burn' is True."
    else:
        assert all(
            v is None
            for v in (stream_shapes, lake_shapes, stream_sigma, stream_dz, lake_dz)
        ), "'stream_shapes', 'lake_shapes', 'stream_sigma', 'stream_dz' and 'lake_dz' are not required when 'burn' is False."

    dirmap = (1, 2, 3, 4, 5, 6, 7, 8)
    grid = Grid.from_raster(raw_dem_path)
    dem = grid.read_raster(raw_dem_path).astype(np.float32)
    ndv = dem.nodata.copy()

    # 'fill_depressions' isn't designed for NoData (see
    #     https://github.com/scikit-image/scikit-image/issues/4078)
    # Either set NoData and values < 0 to zero or -dz, depending on
    # whether streams are being burned. This forces all cells to drain
    # to the edge of the grid
    mask = dem == ndv
    if burn:
        dem[mask] = -(stream_dz + lake_dz)
        dem[dem < 0] = -(stream_dz + lake_dz)
    else:
        dem[mask] = 0
        dem[dem < 0] = 0

    if burn:
        dem = burn_lake_shapes(grid, dem, lake_shapes, lake_dz)
        dem = burn_stream_shapes(grid, dem, stream_shapes, stream_dz, stream_sigma)
    dem = grid.fill_pits(dem, nodata_in=np.nan, nodata_out=np.nan)
    # dem = grid.fill_depressions(dem, nodata_in=np.nan, nodata_out=np.nan)
    dem = grid.fill_depressions64(dem, nodata_in=np.nan, nodata_out=np.nan)
    dem = grid.resolve_flats(
        dem, max_iter=max_iter, eps=eps, nodata_in=np.nan, nodata_out=np.nan
    )

    npits = grid.detect_pits(dem).sum()
    nflats = grid.detect_flats(dem).sum()
    if (npits > 0) or (nflats > 0):
        fill_fname = os.path.split(fill_dem_path)[1]
        msg = f"        {fill_fname} has {npits} pits and {nflats} flats."
        print(msg)
        logging.info(msg)

    # Flow dir and accum
    fdir = grid.flowdir(
        dem, routing="d8", dirmap=dirmap, nodata_in=np.nan, nodata_out=0
    )
    facc = grid.accumulation(
        fdir, routing="d8", dirmap=dirmap, nodata_in=0, nodata_out=0
    )

    # Save results to disk
    if np.issubdtype(dem_dtype, np.integer):
        dem = np.rint(dem)
    dem[np.isnan(dem)] = dem_ndv
    dem = dem.astype(dem_dtype)
    pysheds_array_to_raster(dem, raw_dem_path, fill_dem_path, dem_dtype, dem_ndv)

    fdir = fdir.astype(np.int16)
    pysheds_array_to_raster(fdir, raw_dem_path, fdir_path, np.int16, 0)

    facc = facc.astype(np.uint32)
    pysheds_array_to_raster(facc, raw_dem_path, facc_path, np.uint32, 0)

    return (dem, fdir, facc)
