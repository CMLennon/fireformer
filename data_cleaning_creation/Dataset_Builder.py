import geopandas as gpd
import pandas as pd
import numpy as np
import netCDF4 as nc
import re
import zarr
import datetime

import xarray as xr
import rioxarray as rxr
import rioxarray.merge as rmrg
import rasterio
import earthpy as et
import earthpy.spatial as es
import fiona
from shapely.geometry import mapping
from shapely.geometry import Point
from geocube.api.core import make_geocube
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform

import sys
import os
import errno
from glob import glob
import warnings
from pathlib import Path
from PIL import Image
import json


# import metpy #to test methods - won't function on my local machine

# args = sys.argv[1:]

# Irwin_FP = args[0]
Irwin_FP = "/home/connor/Desktop/IrwinHistoricData/IrwinHistoricData.csv"
# RasterIn = args[1]
RasterIn = "/home/connor/Desktop/MassCrop/InputRasters"
# RasterOut = args[2]
RasterOut = "/home/connor/Desktop/MassCrop/OutputFolder/raster"
# TrasterIn = args[3]
TrasterIn = "/home/connor/Desktop/MassCrop/traster"
# TrasterOut = args[4]
TrasterOut_viirs = "home/connor/Desktop/MassCrop/OutputFolder/traster/viirs/"
# TrasterOut_noaa =args[5]
TrasterOut_noaa = "home/connor/Desktop/MassCrop/OutputFolder/traster/noaa/"
# IrwinReq_FP = args[5]
IrwinReq_FP = "IrwinResource_Request.csv"
# IrwinTS = args[6]
IrwinTS = "/home/connor/Desktop/MassCrop/OutputFolder/tabular"
# StartDate = args[7]
StartDate = "2019-12-15"  # If looking at resources must be 2020-04-15 or later

"""
Step 1: Build up research fires from Irwin data - some initial data processing/cleaning

"""


def fill_dates(df):
    df["date"] = pd.to_datetime(df["date"])
    if df.loc[~df["ControlDateTime"].isna()].empty:
        min_date, max_date = df["date"].min(), df["date"].max()
    else:
        min_date, max_date = (
            df["date"].min(),
            df.loc[~df["ControlDateTime"].isna(), "ControlDateTime"].max(),
        )
    all_dates = pd.to_datetime(
        pd.date_range(min_date, max_date, freq="D").to_series(name="date")
    )
    out = df.merge(all_dates, how="outer", on="date")
    out["IrwinID"] = df["IrwinID"].unique()[0]
    return out


def initial_data_processor(Irwin_FP, pretraining=True, from_disk=True):

    """
    "Simple" utility function to clean the Irwin fire data

    """

    if from_disk:
        file = gpd.GeoDataFrame(
            pd.read_pickle(
                "/home/connor/Desktop/MassCrop/OutputFolder/tabular/tabular_fixed_effects.pckl"
            ).set_index("index"),
            crs="EPSG:5070",
        )
        return file.loc[file["date"] > StartDate]

    else:

        IrwinDataHist = pd.read_csv(Irwin_FP)
        IrwinDataHist = gpd.GeoDataFrame(
            IrwinDataHist[~pd.isna(IrwinDataHist["geometry"])],
            geometry=IrwinDataHist[~pd.isna(IrwinDataHist["geometry"])][
                "geometry"
            ].apply(lambda x: Point(eval(x))),
            crs={"init": "epsg:4269"},
        )
        IrwinDataHist["date"] = pd.to_datetime(
            IrwinDataHist["GDB_FROM_DATE"], unit="ms"
        ).dt.date
        IrwinDataHist = IrwinDataHist.to_crs(epsg=5070)

        # Big Filter Op
        studyfires = IrwinDataHist[IrwinDataHist["date"] > pd.to_datetime(StartDate)]

        if pretraining:

            list_of_valid = (
                studyfires[
                    ~studyfires["EstimatedCostToDate"].isin([np.nan])
                    & (
                        ~studyfires["POOState"].isin(
                            ["US-AK", "us-ak", "us-hi", "US-HI"]
                        )
                    )
                    & (studyfires["IncidentTypeCategory"].isin(["WF", "wf"]))
                ]
                .groupby("UniqueFireIdentifier", axis=0)["date"]
                .nunique()
                .gt(2)
            )
        else:

            list_of_valid = (
                studyfires[
                    ~studyfires["TotalIncidentPersonnel"].isin([np.nan])
                    & (
                        ~studyfires["POOState"].isin(
                            ["US-AK", "us-ak", "us-hi", "US-HI"]
                        )
                    )
                    & (studyfires["IncidentTypeCategory"].isin(["WF", "wf"]))
                    & (~studyfires["EstimatedFinalCost"].isin([np.nan]))
                ]
                .groupby("UniqueFireIdentifier", axis=0)["date"]
                .nunique()
                .gt(2)
            )
        fire_ids = list_of_valid[list_of_valid].reset_index()["UniqueFireIdentifier"]
        fire_panel_data = studyfires[studyfires["UniqueFireIdentifier"].isin(fire_ids)]
        nan_map1 = (
            fire_panel_data[["ControlDateTime", "UniqueFireIdentifier", "IrwinID"]]
            .dropna()
            .set_index("UniqueFireIdentifier")
            .to_dict()
        )
        nan_map2 = (
            fire_panel_data[["FireOutDateTime", "UniqueFireIdentifier", "IrwinID"]]
            .dropna()
            .set_index("UniqueFireIdentifier")
            .to_dict()
        )
        nan_map3 = (
            fire_panel_data[["ContainmentDateTime", "UniqueFireIdentifier", "IrwinID"]]
            .dropna()
            .set_index("UniqueFireIdentifier")
            .to_dict()
        )
        nan_map4 = (
            fire_panel_data[
                ["FireGrowthCessationDateTime", "UniqueFireIdentifier", "IrwinID"]
            ]
            .dropna()
            .set_index("UniqueFireIdentifier")
            .to_dict()
        )

        newdat = pd.DataFrame(nan_map1)
        newdat2 = pd.DataFrame(nan_map2)
        newdat3 = pd.DataFrame(nan_map3)
        newdat4 = pd.DataFrame(nan_map4)

        newdat_bool1 = pd.to_datetime(
            newdat["ControlDateTime"], unit="ms"
        ).dt.date > pd.to_datetime("2020-01-01")
        newdat_bool2 = pd.to_datetime(
            newdat2["FireOutDateTime"], unit="ms"
        ).dt.date > pd.to_datetime("2020-01-01")
        newdat_bool3 = pd.to_datetime(
            newdat3["ContainmentDateTime"], unit="ms", errors="coerce"
        ).dt.date > pd.to_datetime("2020-01-01")
        newdat_bool4 = pd.to_datetime(
            newdat4["FireGrowthCessationDateTime"], unit="ms", errors="coerce"
        ).dt.date > pd.to_datetime("2020-01-01")

        outdat = pd.concat(
            [
                newdat[newdat_bool1],
                newdat2[newdat_bool2],
                newdat3[newdat_bool3],
                newdat4[newdat_bool4],
            ]
        )

        IrwinDataHist["date"] = pd.to_datetime(IrwinDataHist["date"])

        full_data = IrwinDataHist[IrwinDataHist["IrwinID"].isin(outdat["IrwinID"])]

        g = full_data.groupby("IrwinID")

        list_of_data = [fill_dates(df) for _, df in g]

        full_data = pd.concat(list_of_data).sort_values(["IrwinID", "date"])

        full_data = full_data.reset_index()

        no_bfill_columns = [
            "Fatalities",
            "Injuries",
            "PercentContained",
            "OtherStructuresThreatened",
            "OtherStructuresDestroyed",
            "IsFireCauseInvestigated",
            "IsFSAssisted",
            "IsInitialFireStrategyMet",
            "IsUnifiedCommand",
            "ResidencesThreatened",
            "ResidencesDestroyed",
            "EstimatedCostToDate",
            "geometry",
            "CalculatedAcres",
            "DailyAcres",
        ]

        full_data = full_data.set_index("date")

        full_data[no_bfill_columns] = full_data.groupby("IrwinID")[
            no_bfill_columns
        ].ffill()

        no_bfill_columns.append("IrwinID")

        full_data[full_data.columns[~(full_data.columns.isin(no_bfill_columns))]] = (
            full_data.groupby("IrwinID")[
                full_data.columns[~(full_data.columns.isin(no_bfill_columns))]
            ]
            .ffill()
            .bfill()
        )

        full_data = full_data.reset_index()

        # drop first 6 vars.
        cols = list(full_data.columns.values)
        cols.pop(cols.index("date"))
        full_data = full_data[cols + ["date"]]
        full_data = full_data.iloc[:, 6:]
        full_data.reset_index()

        # search over all columns, find those with 'Time'
        selector_time = full_data.columns.map(lambda x: bool(re.search("Time", x)))

        # pandas datetime only works on series obj - so, stack into series and drop into columns afterwards
        full_data[full_data.columns[selector_time]] = full_data[
            full_data.columns[selector_time]
        ].apply(lambda v: pd.to_datetime(v, unit="ms"))
        # del(selector_time, no_bfill_columns, list_of_data, newdat_bool1, newdat_bool2, newdat_bool3, newdat4, newdat, newdat2, newdat3, newdat4, nan_map1, nan_map2, nan_map3, nan_map4, fire_panel_data, fire_ids, IrwinDataHist)

        print(full_data)
        return full_data


def create_buffered_data(full_data):
    """
    Utility function that does minor cleaning and buffers points for eventual passage into a ViT
    """

    clipper_dat_geom = full_data[["geometry", "IrwinID"]].drop_duplicates(
        subset=["IrwinID"]
    )

    clipper_dat_geom["geometry"] = clipper_dat_geom["geometry"].buffer(
        6000, cap_style=3
    )

    return clipper_dat_geom


def slopify(rxr_dat, cell_size):
    """
    Utility function to generate slope data from elevation

    ----
    Args: rxr_dat - rioxarray (or other numpy friendly data)
          cell_size - size, in meters, of a single cell

    Returns - DataArray with same coordinates of
    """

    # calculate directional gradients
    px, py = np.gradient(rxr_dat.data.squeeze(), cell_size)

    # calculate slope
    slope = np.sqrt(px ** 2 + py ** 2)

    # convert from NP to xarray
    slope_out = xr.DataArray(data=np.expand_dims(slope, axis=0), coords=rxr_dat.coords)

    return slope_out


# dir_path = "/home/connor/Desktop/MassCrop"

# file_extension = "*img"

# files_img = [y for x in os.walk(dir_path) for y in glob(os.path.join(x[0], '*.' + file_extension))]

# file_extension = "*tif"

# files_tif = [y for x in os.walk(dir_path) for y in glob(os.path.join(x[0], '*.' + file_extension))]

# files = files_tif + files_img

import warnings


class DatasetBuilder:
    def __init__(self, raster, traster, timeseries, study_data):

        """
        takes list of files, cuts and stacks them based on input type and then sorts them into one of several folders,
        or builds directory structure directly if missing.

        inputs - 3 lists of file directories (raster, traster, timeseries)
               - study_data: study_data takes a geopandas dataframe with all relevant points (ignition points for
                   wildfires in this case) around which our data will be clipped/summarized using.

               - study_data: req - 1 column named 'IrwinID' (will produce name of files) and 1 column named 'geometry'
        """

        self.raster_list = False
        self.traster_list = False
        self.timeseries_list = False
        self.viirs_empty = None

        # store raster list
        self.raster = raster

        # store raster list
        self.traster = traster

        # store timeseries list
        self.timeseries = timeseries
        # study_data - check it has required columns and is geopandas
        if {"IrwinID", "geometry"}.issubset(study_data.columns):

            try:

                self.crs = study_data.crs
                self.study_data = study_data

            except AttributeError:

                warnings.warn(
                    "You input a non-geopandas obj or file does not contain a CRS. Assumes CRS of EPSG:5070",
                    UserWarning,
                )

                self.study_data = gpd.GeoDataFrame(
                    study_data, geometry=geometry
                ).to_crs(epsg=5070)

                self.crs = self.study_data.crs

        else:
            raise Warning("Necessary columns ('IrwinID','geometry') not present")

    def __checklist__(self, obj):

        # simple util fcn to check if object is a list
        if isinstance(obj, list):
            return True
        else:
            return False

    def _slopify(self, rxr_dat, cell_size):
        """
        Utility function to generate slope data from elevation

        ----
        Args: rxr_dat - rioxarray (or other numpy friendly data)
              cell_size - size, in meters, of a single cell

        Returns - DataArray with same coordinates of
        """

        # calculate directional gradients
        px, py = np.gradient(rxr_dat.data.squeeze(), cell_size)

        # calculate slope
        slope = np.sqrt(px ** 2 + py ** 2)

        # convert from NP to xarray
        slope_out = xr.DataArray(data=slope, coords=rxr_dat.coords)

        slope_out.rio.write_nodata(-9999, inplace=True)

        return slope_out

    def _raster_maker(self, destination_folder, buffer_meter, exclude=False):

        """
        User function to create directory of stacked multi-band raster data for input into fixed-effect pipeline

        input: destination_folder - path of eventual folder where you want raster data to get written to

        output: folder full of raster 3d arrays

        NOTE: searches for .img and .tif files in all directories and subdirectories included
        """
        print(self.study_data.date)
        study_buff = self.study_data.reset_index()
        date_store = study_buff.date
        study_buff.geometry = study_buff.buffer(buffer_meter, cap_style=3)

        print(
            study_buff[["geometry", "IrwinID", "date"]]
            .groupby(["IrwinID"])
            .min()
            .to_dict("index")
        )

        dir_path = self.raster

        file_extension = "*img"

        files_img = [
            y
            for x in os.walk(dir_path)
            for y in glob(os.path.join(x[0], "*." + file_extension))
        ]

        file_extension = "*tif"

        files_tif = [
            y
            for x in os.walk(dir_path)
            for y in glob(os.path.join(x[0], "*." + file_extension))
        ]

        files = files_img + files_tif

        # files.remove('/home/connor/Desktop/MassCrop/InputRasters/LF2016_VDep_200_CONUS/Tif/LC16_VDep_200.tif')
        files = [i for i in files if "ushu10.tif" not in i]

        fnames = [os.path.basename(path) for path in files]
        try:
            with open(self.raster + "/Variable_OrderREADME.txt", "w") as filehandle:
                for var in fnames:
                    filehandle.write(var + "\n")
        except FileNotFoundError:
            with open(self.raster + "/Variable_OrderREADME.txt", "w+") as filehandle:
                for var in fnames:
                    filehandle.write(var)

        files_list = [files[i : i + 3] for i in range(0, len(files), 3)]

        print("The number of raster files found is: " + str(len(files)))

        print(files_list)

        study_buff["date"] = date_store
        date_clip_dict = (
            study_buff[["geometry", "IrwinID", "date"]]
            .groupby(["IrwinID"])
            .min()
            .to_dict("index")
        )

        # indices = study_buff[['geometry', 'IrwinID', 'date']].groupby(['IrwinID']).idxmax().reset_index()['date'].tolist()

        clip_dict = (
            study_buff[["geometry", "IrwinID"]]
            .set_crs(5070)
            .set_index("IrwinID")
            .to_dict("index")
        )
        # indices,:
        rast_out = {}

        slope_switch = False

        if exclude:
            pre_run_IrwinID = [
                x[1]
                for x in os.walk("/home/connor/Desktop/MassCrop/OutputFolder/raster/")
            ][0]

            [clip_dict.pop(IrwinID) for IrwinID in pre_run_IrwinID]

        for k in clip_dict:
            rast_out = {}
            i = 0
            geometry = clip_dict[k]
            minx, miny, maxx, maxy = geometry["geometry"].bounds
            # for files in files_list:
            #     i += 1
            #     print('beginning sublist number ' + str(i))
            for file in files:
                print(file)

                if (
                    file
                    == "/home/connor/Desktop/MassCrop/InputRasters/LF2016_Elev_200_CONUS/Tif/LC16_Elev_200.tif"
                ):
                    slope_switch = True

                with rasterio.open(file) as src:

                    print("reading files from " + file)
                    print("Input file CRS is {}".format(src.crs))
                    print("Input file shape is {}".format(src.shape))
                    print("Input file transform is {}".format(src.transform))

                    with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as vrt:

                        print("VRT shape is {}".format(vrt.shape))
                        rast = rxr.open_rasterio(
                            vrt, from_disk=True, masked=True
                        ).rio.clip_box(minx, miny, maxx, maxy)
                # try:
                #     rast_stats = open('/home/connor/Desktop/MassCrop/InputRasters/raster_summary_stats.json', 'r')
                #     #rast_max = json.load(rast_stats)[file]['max']
                #     #rast_min = json.load(rast_stats)[file]['min']
                #     #rast_stats.close()
                # except:
                #     #rast_max = 1
                #     #rast_min = 0

                if rast_out == {}:

                    rast_dict = {}
                    slope_dict = {}
                    remove_list = []

                    # for k in clip_dict:

                    geometry = clip_dict[k]
                    try:
                        print("trying squeeze")
                        rast = rast.squeeze()

                    except ValueError as err:
                        print(err)
                        print(k)
                        print(file)
                        remove_list.append(k)
                        rast_dict[k] = np.zeros(raster_shape)

                    print("trying clip")
                    rast_dict[k] = rast.rio.clip(
                        [geometry["geometry"]],
                        self.crs,
                        from_disk=True,
                        all_touched=True,
                    )
                    print(rast_dict[k])
                    print("creating shape")
                    raster_shape = rast_dict[k].shape
                    print("trying to reproject")
                    rast_dict[k] = rast_dict[k].rio.reproject(
                        dst_crs=rast.rio.crs, resolution=30
                    )
                    print("writing nodata")
                    rast_dict[k].rio.write_nodata(-9999, inplace=True)

                    print("extracting nodata")
                    nodata = rast_dict[k].rio.nodata

                    print("filtering nodata")
                    rast_dict[k] = rast_dict[k].where(rast_dict[k] != nodata)
                    print("normalizing (skipping)")

                    # rast_dict[k] = (rast_dict[k]-rast_min)/(rast_max)
                    if slope_switch:
                        # print(k)
                        print(rast_dict[k].coords)
                        slope = self._slopify(rxr_dat=rast_dict[k], cell_size=30)
                        print(slope)
                        slope_dict[k] = slope

                    for k in remove_list:
                        print("would remove")
                        # clip_dict.pop(k, 'None')

                    if len(remove_list) > 0:
                        self.study_data = self.study_data[
                            ~self.study_data["IrwinID"].isin(remove_list)
                        ]

                    rast_out[k] = (
                        rast_dict[k].rio.reproject(self.crs).to_dataset(name=str(file))
                    )

                else:

                    # for k in clip_dict:
                    geometry = clip_dict[k]
                    try:
                        print("trying squeeze")
                        rast = rast.squeeze()

                    except ValueError as err:
                        print(err)
                        print(k)
                        print(file)
                        remove_list.append(k)
                        rast_dict[k] = np.zeros(raster_shape)

                    print("trying clip")
                    rast_dict[k] = rast.rio.clip(
                        [geometry["geometry"]],
                        self.crs,
                        from_disk=True,
                        all_touched=True,
                    )
                    print(rast_dict[k])
                    print("creating shape")
                    raster_shape = rast_dict[k].shape
                    print("trying to reproject")
                    rast_dict[k] = rast_dict[k].rio.reproject(
                        dst_crs=rast.rio.crs, resolution=30
                    )
                    print("writing nodata")
                    rast_dict[k].rio.write_nodata(-9999, inplace=True)

                    print("extracting nodata")
                    nodata = rast_dict[k].rio.nodata

                    print("filtering nodata")
                    rast_dict[k] = rast_dict[k].where(rast_dict[k] != nodata)
                    print("normalizing")
                    # rast_dict[k] = rast_dict[k]/rast_max
                    if slope_switch:
                        # print(k)
                        print(rast_dict[k].coords)
                        slope = self._slopify(rxr_dat=rast_dict[k], cell_size=30)
                        slope_dict[k] = slope

                    for k in remove_list:
                        clip_dict.pop(k, "None")

                    if len(remove_list) > 0:
                        self.study_data = self.study_data[
                            ~self.study_data["IrwinID"].isin(remove_list)
                        ]

                    for k in rast_dict:
                        rast_dict[k].rio.write_nodata(-9999, inplace=True)
                        try:
                            rast_dict[k] = rast_dict[k].rio.reproject_match(rast_out[k])
                            rast_out[k][file] = rast_dict[k]
                        except AttributeError:
                            print("No data found for IrwinID " + k)
                            x = rast_out[k].x.values
                            y = rast_out[k].y.values
                            try:
                                rast_out[k][file] = xr.DataArray(
                                    rast_dict[k],
                                    coords={"x": x, "y": y},
                                    dims=["y", "x"],
                                )
                            except:
                                rast_out[k][file] = xr.DataArray(
                                    rast_dict[k],
                                    coords={
                                        "x": x[0 : rast_dict[k].shape[0]],
                                        "y": y[0 : rast_dict[k].shape[1]],
                                    },
                                    dims=["y", "x"],
                                )
                                rast_out[k][file] = rast_out[k][file].reindex(
                                    {"x": x, "y": y}, fill_value=0
                                )
                        # print(rast_dict[k].crs.epsg)

                        # print(rast_dict[k])
                        # print(rast_out[k])

                        if slope_switch:
                            slope_dict[k] = self._slopify(
                                rxr_dat=rast_dict[k], cell_size=30
                            )
                            # print(k)
                            rast_out[k]["slope"] = slope_dict[k]

                        filename = str(k) + ".tif"
                        subfolder = str(k)
                        destination_folder = destination_folder
                        filepath = os.path.join(destination_folder, subfolder, filename)

                        if not os.path.exists(os.path.dirname(filepath)):
                            try:
                                os.makedirs(os.path.dirname(filepath))
                            except OSError as exc:  # Guard against race condition
                                if exc.errno != errno.EEXIST:
                                    raise
                        print("Rast out looks like: ")
                        print(rast_out[k])

                        rast_dict = {}
                        slope_switch = False

            # soil moisture

            with rasterio.open(
                "/home/connor/Desktop/MassCrop/TimedRasters/soil_moist/soilw.mon.mean_clip.nc"
            ) as src:

                print(
                    "reading files from "
                    + "/home/connor/Desktop/MassCrop/TimedRasters/soil_moist/soilw.mon.mean_clip.nc"
                )
                print("Input file CRS is {}".format(src.crs))
                print("Input file shape is {}".format(src.shape))
                print("Input file transform is {}".format(src.transform))

                print(isinstance(date_clip_dict[k]["date"], datetime.date))
                print(date_clip_dict[k]["date"])

                with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as tmp:
                    tmp = (
                        rxr.open_rasterio(tmp, from_disk=True)
                        .rio.write_crs("EPSG:5070")
                        .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                        .sel(time=date_clip_dict[k]["date"], method="nearest")
                        .rio.reproject("EPSG:5070", resolution=30)
                    )
            # tmp = rxr.open_rasterio('/home/connor/Desktop/MassCrop/TimedRasters/soil_moist/soilw.mon.mean_clip.nc').rio.write_crs('EPSG:5070').rio.reproject('EPSG:5070', resolution = 30).rio.clip([clip_dict[k]['geometry']]).sel(time = date_clip_dict[k]['date'], method="nearest")
            rast_out[k]["soil moisture"] = (
                xr.DataArray(tmp.to_array().squeeze(drop=True), dims=["y", "x"])
                .squeeze()
                .rio.reproject_match(rast_out[k])
                .rio.write_nodata(-9999)
            )
            # max temperature
            if date_clip_dict[k]["date"].year > 2020:
                with rasterio.open(
                    "/home/connor/Desktop/MassCrop/TimedRasters/temp/tmax.2021_clip.nc"
                ) as src:

                    print(
                        "reading files from "
                        + "/home/connor/Desktop/MassCrop/TimedRasters/temp/tmax.2021_clip.nc"
                    )
                    print("Input file CRS is {}".format(src.crs))
                    print("Input file shape is {}".format(src.shape))
                    print("Input file transform is {}".format(src.transform))

                    with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as tmp:
                        tmp = (
                            rxr.open_rasterio(tmp, from_disk=True)
                            .rio.write_crs("EPSG:5070")
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                            .sel(time=date_clip_dict[k]["date"], method="nearest")
                            .rio.reproject("EPSG:5070", resolution=30)
                        )

                rast_out[k]["temp"] = (
                    xr.DataArray(tmp.to_array().squeeze(drop=True), dims=["y", "x"])
                    .rio.reproject_match(rast_out[k])
                    .rio.write_nodata(-9999)
                )
                with rasterio.open(
                    "/home/connor/Desktop/MassCrop/TimedRasters/temp/tmax.2021_clip.nc"
                ) as src:

                    print(
                        "reading files from "
                        + "/home/connor/Desktop/MassCrop/TimedRasters/temp/tmax.2021_clip.nc"
                    )
                    print("Input file CRS is {}".format(src.crs))
                    print("Input file shape is {}".format(src.shape))
                    print("Input file transform is {}".format(src.transform))

                    with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as tmp:
                        tmp = (
                            rxr.open_rasterio(tmp, from_disk=True)
                            .rio.write_crs("EPSG:5070")
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                            .sel(time=date_clip_dict[k]["date"], method="nearest")
                            .rio.reproject("EPSG:5070", resolution=30)
                        )

                rast_out[k]["temp"] = (
                    xr.DataArray(tmp.to_array().squeeze(drop=True), dims=["y", "x"])
                    .rio.reproject_match(rast_out[k])
                    .rio.write_nodata(-9999)
                )
            else:
                with rasterio.open(
                    "/home/connor/Desktop/MassCrop/TimedRasters/temp/tmax.2020_clip.nc"
                ) as src:

                    print(
                        "reading files from "
                        + "/home/connor/Desktop/MassCrop/TimedRasters/temp/tmax.2020_clip.nc"
                    )
                    print("Input file CRS is {}".format(src.crs))
                    print("Input file shape is {}".format(src.shape))
                    print("Input file transform is {}".format(src.transform))

                    with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as tmp:
                        tmp = (
                            rxr.open_rasterio(tmp, from_disk=True)
                            .rio.write_crs("EPSG:5070")
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                            .sel(time=date_clip_dict[k]["date"], method="nearest")
                            .rio.reproject("EPSG:5070", resolution=30)
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                        )

                # "6326"tmp = rxr.open_rasterio('/home/connor/Desktop/MassCrop/TimedRasters/temp/tmax.2021_clip.nc').rio.write_crs('EPSG:5070').rio.reproject('EPSG:5070', resolution = 30).rio.clip([clip_dict[k]['geometry']]).sel(time = date_clip_dict[k]['date'], method="nearest")
                rast_out[k]["temp"] = (
                    xr.DataArray(tmp.to_array().squeeze(drop=True), dims=["y", "x"])
                    .rio.reproject_match(rast_out[k])
                    .rio.write_nodata(-9999)
                )

            # precip rate

            if date_clip_dict[k]["date"].year <= 2020:
                print(
                    "reading files from "
                    + "/home/connor/Desktop/MassCrop/TimedRasters/precip_rate/prate_2020_clipped.nc"
                )
                # print('Input file CRS is {}'.format(src.crs))
                # print('Input file shape is {}'.format(src.shape))
                # print('Input file transform is {}'.format(src.transform))

                tmp = (
                    xr.open_dataset(
                        "/home/connor/Desktop/MassCrop/TimedRasters/precip_rate/prate_2020_clipped.nc"
                    )
                    .rio.write_crs(
                        "+proj=lcc +lat_1=50 +lat_2=50 +lat_0=50 +lon_0=-107 +x_0=5632642.22547 +y_0=4612545.65137 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
                    )
                    .reset_coords(["lat", "lon"], drop=True)
                    .rio.reproject("EPSG:5070")
                )
                tmp = (
                    tmp.rio.reproject("EPSG:5070")
                    .rio.write_crs("EPSG:5070")
                    .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                    .sel(time=date_clip_dict[k]["date"], method="nearest")
                    .rio.reproject("EPSG:5070", resolution=30)
                    .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                )
                tmp = tmp["prate"]
                rast_out[k]["prate"] = (
                    xr.DataArray(tmp.squeeze(drop=True), dims=["y", "x"])
                    .rio.reproject_match(rast_out[k])
                    .rio.write_nodata(-9999)
                )
            else:
                print(
                    "reading files from "
                    + "/home/connor/Desktop/MassCrop/TimedRasters/precip_rate/prate_2021_clipped.nc"
                )
                # print('Input file CRS is {}'.format(src.crs))
                # print('Input file shape is {}'.format(src.shape))
                # print('Input file transform is {}'.format(src.transform))
                tmp = (
                    xr.open_dataset(
                        "/home/connor/Desktop/MassCrop/TimedRasters/precip_rate/prate_2021_clipped.nc"
                    )
                    .rio.write_crs(
                        "+proj=lcc +lat_1=50 +lat_2=50 +lat_0=50 +lon_0=-107 +x_0=5632642.22547 +y_0=4612545.65137 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
                    )
                    .reset_coords(["lat", "lon"], drop=True)
                    .rio.reproject("EPSG:5070")
                )
                tmp = (
                    tmp.rio.reproject("EPSG:5070")
                    .rio.write_crs("EPSG:5070")
                    .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                    .sel(time=date_clip_dict[k]["date"], method="nearest")
                    .rio.reproject("EPSG:5070", resolution=30)
                    .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                )

                # "6326"tmp = rxr.open_rasterio('/home/connor/Desktop/MassCrop/TimedRasters/temp/tmax.2021_clip.nc').rio.write_crs('EPSG:5070').rio.reproject('EPSG:5070', resolution = 30).rio.clip([clip_dict[k]['geometry']]).sel(time = date_clip_dict[k]['date'], method="nearest")
                print(tmp)
                tmp = tmp["prate"]
                print(tmp)
                rast_out[k]["prate"] = (
                    xr.DataArray(tmp.squeeze(drop=True), dims=["y", "x"])
                    .rio.reproject_match(rast_out[k])
                    .rio.write_nodata(-9999)
                )

            # moisture anomaly
            date = date_clip_dict[k]["date"]
            date_str = (
                "/home/connor/Desktop/MassCrop/TimedRasters/moist_anom/"
                + "w.anom."
                + str(date.year)
                + str(date.month).zfill(2)
                + str(date.day).zfill(2)
                + ".tif"
            )
            if (
                date_str
                == "/home/connor/Desktop/MassCrop/TimedRasters/moist_anom/w.anom.20210523.tif"
            ):
                date_str = "/home/connor/Desktop/MassCrop/TimedRasters/moist_anom/w.anom.20210524.tif"
            with rasterio.open(date_str) as src:

                print("reading files from " + date_str)
                print("Input file CRS is {}".format(src.crs))
                print("Input file shape is {}".format(src.shape))
                print("Input file transform is {}".format(src.transform))

                with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as tmp:
                    tmp = (
                        rxr.open_rasterio(tmp, from_disk=True)
                        .rio.write_crs("EPSG:5070")
                        .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                        .rio.reproject("EPSG:5070", resolution=30)
                        .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                        .rio.set_nodata(-9999)
                    )

            # tmp = rxr.open_rasterio(date_str).rio.write_crs("EPSG:4326").rio.reproject('EPSG:5070', resolution = 30).rio.clip([clip_dict[k]['geometry']]).sel(time = date_clip_dict[k]['date'], method="nearest")

            rast_out[k]["anom"] = (
                xr.DataArray(tmp.squeeze(drop=True), dims=["y", "x"])
                .rio.reproject_match(rast_out[k])
                .rio.write_nodata(-9999)
            )
            # Windspeed/direction
            if date_clip_dict[k]["date"].year > 2020:

                with rasterio.open(
                    "/home/connor/Desktop/MassCrop/TimedRasters/winds/vwnd.10m.gauss.2021_clip.nc"
                ) as src:

                    print(
                        "reading files from "
                        + "/home/connor/Desktop/MassCrop/TimedRasters/winds/vwnd.10m.gauss.2021_clip.nc"
                    )
                    print("Input file CRS is {}".format(src.crs))
                    print("Input file shape is {}".format(src.shape))
                    print("Input file transform is {}".format(src.transform))

                    with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as tmp:
                        vwnd = (
                            rxr.open_rasterio(tmp, from_disk=True, masked=True)
                            .rio.write_crs("EPSG:5070")
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                            .sel(time=date_clip_dict[k]["date"], method="nearest")
                            .rio.reproject("EPSG:5070", resolution=30)
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                        )
                        # vwnd.loc[vwnd > 10000,'vwnd'] = -9999
                        vwnd = vwnd.rio.write_nodata(-9999)
                with rasterio.open(
                    "/home/connor/Desktop/MassCrop/TimedRasters/winds/uwnd.10m.gauss.2021_clip.nc"
                ) as src:

                    print(
                        "reading files from "
                        + "/home/connor/Desktop/MassCrop/TimedRasters/winds/uwnd.10m.gauss.2021_clip.nc"
                    )
                    print("Input file CRS is {}".format(src.crs))
                    print("Input file shape is {}".format(src.shape))
                    print("Input file transform is {}".format(src.transform))

                    with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as tmp:
                        uwnd = (
                            rxr.open_rasterio(tmp, from_disk=True, masked=True)
                            .rio.write_crs("EPSG:5070")
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                            .sel(time=date_clip_dict[k]["date"], method="nearest")
                            .rio.reproject("EPSG:5070", resolution=30)
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                        )
                        # uwnd.loc[uwnd > 10000,'uwnd'] = -9999
                        uwnd = uwnd["uwnd"].rio.write_nodata(-9999)

                # vwnd = rxr.open_rasterio('/home/connor/Desktop/MassCrop/TimedRasters/winds/vwnd.10m.gauss.2021_clip.nc').rio.write_crs("EPSG:4326").rio.reproject('EPSG:5070').rio.clip([clip_dict[k]['geometry']]).sel(time = date_clip_dict[k]['date'], method="nearest")
                # uwnd = rxr.open_rasterio('/home/connor/Desktop/MassCrop/TimedRasters/winds/uwnd.10m.gauss.2021_clip.nc').rio.write_crs("EPSG:4326").rio.reproject('EPSG:5070').rio.clip([clip_dict[k]['geometry']]).sel(time = date_clip_dict[k]['date'], method="nearest")
                # direction
                print(f"Our uwnd looks like - {uwnd}")
                print(f"Our vwnd looks like - {vwnd}")
                raster_wd = np.arctan2(uwnd["uwnd"], vwnd["vwnd"]) * 180 / np.pi
                raster_wd = xr.where(raster_wd >= 0, raster_wd, raster_wd + 360) / 360
                # speed
                raster_ws = np.sqrt(vwnd["vwnd"] ** 2 + uwnd["uwnd"] ** 2) / 100
                print(raster_ws)
                print(raster_wd)

                rast_out[k]["speed"] = (
                    xr.DataArray(raster_ws.squeeze(drop=True), dims=["y", "x"])
                    .rio.reproject_match(rast_out[k])
                    .rio.write_nodata(-9999)
                )
                rast_out[k]["direction"] = (
                    xr.DataArray(raster_wd.squeeze(drop=True), dims=["y", "x"])
                    .rio.write_crs("EPSG:5070")
                    .rio.reproject_match(rast_out[k])
                    .rio.write_nodata(-9999)
                )
            else:
                with rasterio.open(
                    "/home/connor/Desktop/MassCrop/TimedRasters/winds/vwnd.10m.gauss.2020_clip.nc"
                ) as src:

                    print(
                        "reading files from "
                        + "/home/connor/Desktop/MassCrop/TimedRasters/winds/vwnd.10m.gauss.2020_clip.nc"
                    )
                    print("Input file CRS is {}".format(src.crs))
                    print("Input file shape is {}".format(src.shape))
                    print("Input file transform is {}".format(src.transform))

                    with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as tmp:
                        vwnd = (
                            rxr.open_rasterio(tmp, from_disk=True, masked=True)
                            .rio.write_crs("EPSG:5070")
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                            .sel(time=date_clip_dict[k]["date"], method="nearest")
                            .rio.reproject("EPSG:5070", resolution=30)
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                        )
                        # vwnd.loc[vwnd > 10000,'vwnd'] = -9999
                        vwnd["vwnd"] = vwnd["vwnd"].rio.write_nodata(-9999)

                with rasterio.open(
                    "/home/connor/Desktop/MassCrop/TimedRasters/winds/uwnd.10m.gauss.2020_clip.nc"
                ) as src:

                    print(
                        "reading files from "
                        + "/home/connor/Desktop/MassCrop/TimedRasters/winds/uwnd.10m.gauss.2020_clip.nc"
                    )
                    print("Input file CRS is {}".format(src.crs))
                    print("Input file shape is {}".format(src.shape))
                    print("Input file transform is {}".format(src.transform))

                    with rasterio.vrt.WarpedVRT(src, crs="EPSG:5070") as tmp:
                        uwnd = (
                            rxr.open_rasterio(tmp, from_disk=True, masked=True)
                            .rio.write_crs("EPSG:5070")
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                            .sel(time=date_clip_dict[k]["date"], method="nearest")
                            .rio.reproject("EPSG:5070", resolution=30)
                            .rio.clip([clip_dict[k]["geometry"]], all_touched=True)
                        )
                        # uwnd.loc[uwnd > 10000,'uwnd'] = -9999
                        uwnd["uwnd"] = uwnd["uwnd"].rio.write_nodata(-9999)

                # vwnd = rxr.open_rasterio('/home/connor/Desktop/MassCrop/TimedRasters/winds/vwnd.10m.gauss.2020_clip.nc').rio.write_crs('EPSG:5070').rio.reproject('EPSG:5070', resolution = 30).rio.clip([clip_dict[k]['geometry']]).sel(time = date_clip_dict[k]['date'], method="nearest")
                # uwnd = rxr.open_rasterio('/home/connor/Desktop/MassCrop/TimedRasters/winds/uwnd.10m.gauss.2020_clip.nc').rio.write_crs('EPSG:5070').rio.reproject('EPSG:5070', resolution = 30).rio.clip([clip_dict[k]['geometry']]).sel(time = date_clip_dict[k]['date'], method="nearest")
                # direction
                print(f"Our uwnd looks like - {uwnd}")
                print(f"Our vwnd looks like - {vwnd}")
                raster_wd = np.arctan2(uwnd["uwnd"], vwnd["vwnd"]) * 180 / np.pi
                raster_wd = xr.where(raster_wd >= 0, raster_wd, raster_wd + 360) / 360
                # speed
                raster_ws = np.sqrt(vwnd["vwnd"] ** 2 + uwnd["uwnd"] ** 2) / 100

                rast_out[k]["speed"] = (
                    xr.DataArray(raster_ws.squeeze(drop=True), dims=["y", "x"])
                    .rio.reproject_match(rast_out[k])
                    .rio.write_nodata(-9999)
                )
                rast_out[k]["direction"] = (
                    xr.DataArray(raster_wd.squeeze(drop=True), dims=["y", "x"])
                    .rio.write_crs("EPSG:5070")
                    .rio.reproject_match(rast_out[k])
                    .rio.write_nodata(-9999)
                )

                # , coords ={'x':x, 'y':y}

            dir_path = self.traster

            extension = "*.shp"

            print(dir_path)

            files_shp = [
                str(path.absolute()) for path in Path(dir_path).rglob(extension)
            ]

            viirs_list = [i for i in files_shp if "fire_nrt" in i]

            # for file in viirs_list:
            #     trasterviirs = gpd.read_file(file)

            # trasterviirs_nt = trasterviirs[trasterviirs['DAYNIGHT'] == 'N']
            if "self.viirs_empty" in locals() or "self.viirs_empty" in globals():
                print(self.viirs_empty)
            for file in viirs_list:
                trasterviirs = gpd.read_file(file)
                trasterviirs_nt_tmp = trasterviirs
                if file == viirs_list[0]:
                    trasterviirs_nt = trasterviirs_nt_tmp
                else:
                    trasterviirs_nt = trasterviirs_nt.append(trasterviirs_nt_tmp)

            trasterviirs_nt = trasterviirs_nt.to_crs(5070)
            trasterviirs_nt["ACQ_DATE"] = pd.to_datetime(
                trasterviirs_nt["ACQ_DATE"]
            ).dt.date
            trasterviirs_nt_clipped = trasterviirs_nt[
                trasterviirs_nt.within(clip_dict[k]["geometry"])
            ]
            trasterviirs_nt_clipped = trasterviirs_nt_clipped[
                pd.to_datetime(trasterviirs_nt_clipped["ACQ_DATE"]).between(
                    pd.to_datetime(date_clip_dict[k]["date"]) - pd.DateOffset(days=1),
                    pd.to_datetime(date_clip_dict[k]["date"]) + pd.DateOffset(days=1),
                )
            ]
            trasterviirs_nt_clipped = trasterviirs_nt_clipped.fillna(0)
            trasterviirs_nt_clipped.geometry = trasterviirs_nt_clipped.buffer(
                190, cap_style=3
            )
            print(trasterviirs_nt_clipped)

            if trasterviirs_nt_clipped.empty:
                print("found no VIIRS detection for IrwinID " + str(k))
                # viirs_ds = make_geocube(trasterviirs_nt_clipped, measurements=['FRP','BRIGHT_T31', 'ACQ_DATE', 'LATITUDE', 'LONGITUDE'],
                #                     datetime_measurements=['ACQ_DATE'],
                #                     #resolution = (-375,375),
                #                     like = uwnd,
                #                     #geom = json.dumps(mapping(trasterviirs_nt_clipped)),
                #                     fill = 0).fillna(0)
                viirs_ds = xr.open_dataset("viirsempty.nc")

            else:
                viirs_ds = make_geocube(
                    trasterviirs_nt_clipped,
                    measurements=[
                        "FRP",
                        "BRIGHT_T31",
                        "ACQ_DATE",
                        "LATITUDE",
                        "LONGITUDE",
                    ],
                    datetime_measurements=["ACQ_DATE"],
                    # resolution = (-375,375),
                    like=uwnd,
                    # geom = json.dumps(mapping(trasterviirs_nt_clipped)),
                    fill=0,
                ).fillna(0)

                self.viirs_empty = viirs_ds * 0
                # try:
                # self.viirs_empty.to_netcdf('viirsempty.nc')
                # except PermissionError():
                #    print('Permission Error')
                # else:
                #    print("Other Error")
                print(self.viirs_empty)

            print(viirs_ds)
            # coords = {'x':viirs_ds['LONGITUDE'].values, 'y':viirs_ds['LATITUDE'].values}
            # viirs_ds = viirs_ds.assign_coords(coords)#.rio.set_spatial_dims(x_dim = viirs_ds['LONGITUDE'], y_dim = viirs_ds['LATITUDE'])
            print(
                "min: "
                + str(trasterviirs_nt_clipped["ACQ_DATE"].min())
                + "found for max:"
                + str(trasterviirs_nt_clipped["ACQ_DATE"].max())
            )
            print(viirs_ds)
            rast_out[k]["viirs"] = (
                xr.DataArray(viirs_ds["FRP"])
                .rio.reproject_match(rast_out[k])
                .rio.write_nodata(-9999)
            )  # .groupby('time').sum()

            rast_out[k].rio.reproject(dst_crs=rast.rio.crs, resolution=30)

            rast_out[k].rio.to_raster(filepath)

            rast = None

        slope_dict = None

        # self.raster_dict = rast_out
        # for k in self.raster_dict:
        #     filename = str(k) + '.tif'
        #     destination_folder = destination_folder + '/'
        #     filepath = os.path.join(destination_folder, filename)
        #     self.raster_dict[k].rio.to_raster(filepath)

        return (clip_dict, rast_out)

    def _t_raster_maker(
        self, destination_folder_viirs, destination_folder_noaa, buffer_meter
    ):

        """
        User function to create directory of time-attached 3d numpy arrays

        inputs: destination_folder - path of eventual folder where you want raster data to get written to

                time data: pandas df with columns 'date' 'geometry' (point) and 'IrwinID' (Irwin ID) (sent during init)

                buffer_meter: number of meters to a side that is considered study-area per fire, eg. 10000 -> 10kmx10km study area

        output: folder full of raster 4d (x, y, time, band) arrays

        NOTE: see documentation for how t_raster_maker searches for and finds files that match
        """

        print("building time-series spatial datasets")
        self.viirdict = {}
        self.narrdict = {}

        indices = (
            self.study_data[["geometry", "IrwinID", "date"]]
            .groupby(["IrwinID"])
            .idxmax()
            .reset_index()["date"]
            .tolist()
        )
        input_dat = self.study_data.iloc[indices, :][["geometry", "IrwinID"]]

        print(input_dat)

        # object to trim the data based on fire start/out times
        dateprocessor = self.study_data.groupby(["IrwinID"]).agg(
            {"date": [np.min, np.max]}
        )
        print(self.study_data)
        print(dateprocessor)
        # dateprocessor['date']['amax'] = self.study_data['FireOutDateTime']

        # to create a panel of 3x3 64x64 images for viirs

        geoprocessor = input_dat.buffer(buffer_meter, cap_style=3)
        study_buff = (
            self.study_data[["geometry", "IrwinID", "date"]]
            .set_crs(5070)
            .drop_duplicates(subset=["IrwinID", "date"])
            .set_index("IrwinID")
        )
        # study_buff = self.study_data[['IrwinID', 'date', 'geometry']]
        study_buff.geometry = study_buff.buffer(buffer_meter, cap_style=3)
        search_ranges = self.study_data.groupby(["IrwinID"]).agg(
            {"date": [np.min, np.max]}
        )

        dir_path = self.traster

        file_extension = ["*.shp", "*.nc"]

        print(dir_path)

        files_shp = [
            str(path.absolute()) for path in Path(dir_path).rglob(file_extension[0])
        ]
        # files_shp = [y for x in os.walk(dir_path) for y in glob.glob(os.path.join(x[0], '/**/.' + file_extension[0], recursive = True))]
        print(files_shp)
        # files_nc = [y for x in os.walk(dir_path) for y in glob.glob(os.path.join(x[0], '/**/*.' + file_extension[1], recursive = True))]
        files_nc = [
            str(path.absolute()) for path in Path(dir_path).rglob(file_extension[1])
        ]
        print(files_nc)
        files_shp = files_shp + files_nc

        print(files_shp)

        air_list = [i for i in files_shp if "air_nc" in i]

        for file in air_list:
            trastera = xr.open_dataset(file)
            trastera = trastera["air"][:, 0:2, :, :].mean(dim="level", skipna=True)
            trastera = trastera.resample(time="24H").mean()
            if file == air_list[0]:
                traster_a = trastera

            else:
                traster_a = xr.concat(
                    [traster_a, trastera.stack().squeeze()], dim="time"
                )

        shum_list = [i for i in files_shp if "shum_nc" in i]

        for file in shum_list:
            trasters = xr.open_dataset(file)
            trasters = trasters["shum"][:, 0:2, :, :].mean(dim="level", skipna=True)
            trasters = trasters.resample(time="24H").mean()
            if file == shum_list[0]:
                traster_s = trasters

            else:
                traster_s = xr.concat(
                    [traster_s, trasters.stack().squeeze()], dim="time"
                )

        vwnd_list = [i for i in files_shp if "vwnd_nc" in i]
        # vwnd_list = glob.glob('/home/connor/Desktop/MassCrop/traster/NOAAData/vwnd_nc/*')
        for file in vwnd_list:
            trasterv = xr.open_dataset(file)
            trasterv = trasterv["vwnd"][:, 0:2, :, :].mean(dim="level", skipna=True)

            if file == vwnd_list[0]:
                traster_v = trasterv

            else:
                traster_v = xr.concat(
                    [traster_v, trasterv.stack().squeeze()], dim="time"
                )

        uwnd_list = [i for i in files_shp if "uwnd_nc" in i]
        # uwnd_list =glob.glob('/home/connor/Desktop/MassCrop/traster/NOAAData/uwnd_nc/*')
        for file in uwnd_list:
            trasteru = xr.open_dataset(file)
            trasteru = trasteru["uwnd"][:, 0:2, :, :].mean(dim="level", skipna=True)

            if file == uwnd_list[0]:
                traster_u = trasteru

            else:
                traster_u = xr.concat(
                    [traster_u, trasteru.stack().squeeze()], dim="time"
                )

        traster_ws = (
            np.sqrt(traster_v.sortby("time") ** 2 + traster_u.sortby("time") ** 2) / 100
        )

        traster_ws.attrs["units"] = "m/s (pct of 100)"

        traster_ws = traster_ws.resample(time="24H").mean()
        traster_wd = (
            np.arctan2(traster_u.sortby("time"), traster_v.sortby("time")) * 180 / np.pi
        )
        traster_wd = xr.where(traster_wd >= 0, traster_wd, traster_wd + 360) / 360
        traster_wd.attrs["units"] = "degrees (pct of 360)"
        traster_wd = traster_wd.resample(time="24H").mean()

        narr_dataset = traster_ws.rename("windspeed").to_dataset()
        narr_dataset["winddirection"] = traster_wd.rename("wind-dir")
        narr_dataset["temp"] = (
            traster_a.rename("air_temp") / 316.483
        )  #% of 110 degrees F.
        narr_dataset["spec_humid"] = traster_s.rename("specific_humidity")

        # file_extension = "*shp"
        # files_shp = [y for x in os.walk(dir_path) for y in glob.glob(os.path.join(x[0], '*.' + file_extension))]
        viirs_list = [i for i in files_shp if "fire_nrt" in i]
        # for file in viirs_list:
        #     trasterviirs = gpd.read_file(file)

        # trasterviirs_nt = trasterviirs[trasterviirs['DAYNIGHT'] == 'N']

        for file in viirs_list:
            trasterviirs = gpd.read_file(file)
            trasterviirs_nt_tmp = trasterviirs
            if file == viirs_list[0]:
                trasterviirs_nt = trasterviirs_nt_tmp
            else:
                trasterviirs_nt = trasterviirs_nt.append(trasterviirs_nt_tmp)

        trasterviirs_nt.BRIGHT_T31 = trasterviirs_nt.BRIGHT_T31.fillna(
            trasterviirs_nt.BRIGHT_TI4
        )
        trasterviirs_nt_tmp = None

        for i in dateprocessor["date"].index.values:
            daterange = pd.date_range(
                dateprocessor.loc[i]["date"]["amin"],
                dateprocessor.loc[i]["date"]["amax"],
                freq="D",
            )
            print("selecting NARR data for IrwinID " + i)
            narrds = narr_dataset.sel(time=daterange, method="nearest")
            fpath = destination_folder_noaa + "/" + i + ".nc"
            print(fpath)

            if narrds.time.size == 0:
                print("Warning - found no NARR data in this range.")
            # narrds_clipped = narrds.rio.clip(, conus.crs, drop=False, invert=False)
            narrds = narrds.to_netcdf(path=fpath, format="NETCDF4")

            clipper = study_buff.loc[i].iloc[0]
            print(clipper)
            valid_buff = clipper["geometry"].is_valid
            if valid_buff:
                viirs_list = [i for i in files_shp if "fire_nrt" in i]

                trasterviirs_nt = trasterviirs_nt.to_crs(5070)
                trasterviirs_nt["ACQ_DATE"] = pd.to_datetime(
                    trasterviirs_nt["ACQ_DATE"]
                ).dt.date
                trasterviirs_nt_clipped = trasterviirs_nt[
                    trasterviirs_nt.within(clipper["geometry"])
                ]
                trasterviirs_nt_clipped = trasterviirs_nt_clipped[
                    pd.to_datetime(trasterviirs_nt_clipped["ACQ_DATE"]).between(
                        pd.to_datetime(dateprocessor.loc[i]["date"]["amin"]),
                        pd.to_datetime(dateprocessor.loc[i]["date"]["amax"]),
                    )
                ]
                # print(trasterviirs_nt_clipped)

                if trasterviirs_nt_clipped.empty:
                    print("found no VIIRS detection for IrwinID " + i)
                    self.viirdict[i] = "empty geodataframe - check source"

                else:
                    viirs_ds = make_geocube(
                        trasterviirs_nt_clipped,
                        measurements=[
                            "FRP",
                            "BRIGHT_T31",
                            "ACQ_DATE",
                            "LATITUDE",
                            "LONGITUDE",
                        ],
                        datetime_measurements=["ACQ_DATE"],
                        resolution=(-375, 375),
                        # geom = json.dumps(mapping(clipper)),
                        fill=0,
                        group_by="ACQ_DATE",
                    ).fillna(0)

                    print(
                        "min: "
                        + str(trasterviirs_nt_clipped["ACQ_DATE"].min())
                        + "found for max:"
                        + str(trasterviirs_nt_clipped["ACQ_DATE"].max())
                    )
                    # viirs_ds = viirs_ds.rio.reproject(viirs_ds.rio.crs)

                    # viirs_ds = viirs_ds.rolling(ACQ_DATE=2, center=True, min_periods=1).mean()
                    fpath = destination_folder_viirs + "/" + i + ".nc"
                    # print(fpath)
                    viirs_ds = viirs_ds.to_netcdf(path=fpath, format="NETCDF4")
                    # self.viirdict[i] = viirs_ds
            else:
                self.viirdict[i] = "invalid geometry"
        return None

    def time_series_data(self, destination_folder, requests_data):

        """
        Function to create directory of tabular time-series data

        inputs: destination_folder - path of eventual folder where you want tabular data (all) to get written to

                requests_data: filepath to folder with ALL requests_data

                self:

        output: folder full of 2d arrays with dimensions (time x variable) in destination_folder

        NOTE: see documentation for how _time_series_data searches for and finds files that match
        """

        # read data into file
        requests_data_path = os.path.join(requests_data, "IrwinResource_Request.csv")
        experience_data_path = os.path.join(
            requests_data, "IrwinResource_Experience.csv"
        )

        requests = pd.read_csv(requests_data_path)
        experience = pd.read_csv(experience_data_path)
        # print(requests)
        # strip prefixes - may be deprecated depending on final structure of files
        experience.columns = experience.columns.str.lstrip("properties.")
        requests.columns = requests.columns.str.lstrip("properties.")
        experience = experience.iloc[:, 2:]

        # drop any merge-artifacts
        requests = requests.drop_duplicates()
        experience = experience.drop_duplicates()

        # create base requests frame
        requests = pd.merge(
            experience[
                ["IrwinID", "IrwinRID", "ExperienceFromDate", "ExperienceToDate"]
            ],
            requests,
            how="right",
            left_on=["IrwinID", "IrwinRID"],
            right_on=["IrwinID", "IrwinRID"],
        )

        # fill with special code mapping to correct category for a synthetic IrocRequestID
        fillmapper = {
            "Engine": "E-55555",
            "Medical": "E-77777",
            "Position": "C-55555",
            "Fire": "C-66666",
            "Watertender": "E-66666",
        }
        # print(requests)
        requests.loc[requests["IrocRequestID"].isna(), "IrocRequestID"] = requests[
            requests["IrocRequestID"].isna()
        ]["Category"].map(fillmapper)

        # write estimated time of departure/arrival (for wait-times) into time stamp
        timeobjreq = requests.set_index(["IrwinID", "IrocRequestID", "IrwinRID"])

        timeobjreq[
            [
                "ExperienceFromDate",
                "ExperienceToDate",
                "CreatedOnDateTime_data",
                "CreatedOnDateTime",
                "ModifiedOnDateTime",
                "ETD",
                "ETA",
                "DemobETD",
                "DemobETA",
            ]
        ] = timeobjreq[
            [
                "ExperienceFromDate",
                "ExperienceToDate",
                "CreatedOnDateTime_data",
                "CreatedOnDateTime",
                "ModifiedOnDateTime",
                "ETD",
                "ETA",
                "DemobETD",
                "DemobETA",
            ]
        ].apply(
            pd.to_datetime, errors="coerce", unit="ms"
        )

        processeddat_req = timeobjreq[
            (
                timeobjreq["CreatedOnDateTime_data"].dt.date
                > pd.to_datetime("2020-03-04")
            )
            & (
                (timeobjreq["ETD"] >= pd.to_datetime("2020-03-04"))
                | (timeobjreq["ETA"] >= pd.to_datetime("2020-03-04"))
                | (timeobjreq["ExperienceFromDate"] >= pd.to_datetime("2020-03-04"))
            )
        ]
        processeddat_req.loc[:, "ETA"] = pd.to_datetime(processeddat_req["ETA"])
        processeddat_req = processeddat_req.reset_index()

        processeddat_req = processeddat_req.drop_duplicates()

        # set index
        processeddat_req.reset_index().set_index(
            ["IrwinID", "IrocRequestID", "IrwinRID", "DemobETD"]
        )
        processeddat_req[
            pd.to_datetime(processeddat_req["ETA"], unit="ms", errors="coerce").dt.date
            > pd.to_datetime("01-01-2040")
        ]

        processeddat_req[
            processeddat_req["CreatedOnDateTime_data"].dt.date
            > pd.to_datetime("01-01-2040")
        ] = pd.NaT

        # clean date stamps that are from 1700s or 2200s
        processeddat_req.loc[:, "CreatedOnDateTime_data"] = pd.to_datetime(
            processeddat_req["CreatedOnDateTime_data"]
        )
        processeddat_req.loc[:, "ETA"] = pd.to_datetime(processeddat_req["ETA"])
        processeddat_req.loc[
            (processeddat_req["ETA"] < pd.to_datetime("01-01-2020"))
            | (processeddat_req["ETA"] > pd.to_datetime("01-01-2040")),
            "ETA",
        ] = pd.NaT
        processeddat_req.loc[
            (processeddat_req["ETD"] < pd.to_datetime("01-01-2020"))
            | (processeddat_req["ETD"] > pd.to_datetime("01-01-2040")),
            "ETD",
        ] = pd.NaT

        # set creation date to timestamp(ns) form
        processeddat_req["CreatedOnDateTime_data"] = pd.to_datetime(
            processeddat_req["CreatedOnDateTime_data"]
        )
        processeddat_req.loc[:, "ETA"] = pd.to_datetime(processeddat_req["ETA"])
        # initialize variable for new request date marker
        processeddat_req["RequestCreation"] = pd.NaT

        processeddat_req.loc[
            (
                (
                    processeddat_req.CreatedOnDateTime_data - processeddat_req["ETD"]
                    > pd.Timedelta(0)
                )
                & (processeddat_req["ETD"] > pd.to_datetime("01-01-2020"))
            ),
            "RequestCreation",
        ] = processeddat_req.loc[
            (
                (
                    processeddat_req.CreatedOnDateTime_data - processeddat_req["ETD"]
                    > pd.Timedelta(0)
                )
                & (processeddat_req["ETD"] > pd.to_datetime("01-01-2020"))
            ),
            "ETD",
        ].dt.date

        processeddat_req.loc[
            (
                processeddat_req["CreatedOnDateTime_data"] - processeddat_req["ETD"]
                <= pd.Timedelta(0)
            ),
            "RequestCreation",
        ] = processeddat_req.loc[
            (
                processeddat_req["CreatedOnDateTime_data"] - processeddat_req["ETD"]
                <= pd.Timedelta(0)
            ),
            "CreatedOnDateTime_data",
        ].dt.date

        processeddat_req.loc[
            (processeddat_req["RequestCreation"]).isna(), "RequestCreation"
        ] = processeddat_req.loc[
            (processeddat_req["RequestCreation"]).isna(), "CreatedOnDateTime_data"
        ].dt.date

        # create wait-time variable
        processeddat_req["WaitTime"] = (
            pd.to_datetime(processeddat_req["ETA"]).dt.date
            - pd.to_datetime(processeddat_req["RequestCreation"]).dt.date
        )
        # create departure time variable
        processeddat_req.loc[
            processeddat_req["WaitTime"] > pd.Timedelta(1000, unit="D"), "WaitTime"
        ] = pd.NaT
        processeddat_req["DepartureTime"] = pd.NaT
        processeddat_req.loc[
            pd.isna(processeddat_req["DemobETD"]), "DepartureTime"
        ] = processeddat_req.loc[pd.isna(processeddat_req["DemobETD"]), "DemobETA"]
        processeddat_req.loc[
            ~pd.isna(processeddat_req["DemobETD"]), "DepartureTime"
        ] = processeddat_req.loc[~pd.isna(processeddat_req["DemobETD"]), "DemobETD"]

        # ---------------------------------------------------------------------------------------------------

        # PART II - Set up logic for creating type I, type II and type other firefighters, aircraft, helicopters etc.

        type_1_ff = [
            "Type 1",
            "Firefighter Type 1",
            "Smokejumper",
            "Crew Boss, Single Resource",
        ]
        type_2_ff = ["Type 2", "Firefighter Type 2", "Fuels"]
        type_faller_ff = [
            "Advanced Faller",
            "Faller",
            "Prescribed Fire Burn Boss Type 1",
        ]

        type_1_air = ["VLAT", "Type 1", "Type 1 Standard", "Type 1 Limited"]
        type_2_air = [
            "Type 2",
            "Type 3",
            "Airtanker",
            "Type 1 or 2",
            "Type 2 Standard",
            "Type 3 Standard",
            "Air Tactical",
            "Tactical",
            "Type 3 Multi-engine",
            "Type 3 Fixed Wing",
            "Type 3 Single, Multi or Type 4",
            "Type 3 Multi-engine",
        ]

        type_1_heli = ["Type 1", "Type 1 Standard", "Type 1 Limited"]
        type_2_heli = [
            "Type 2",
            "Type 3",
            "Type 3 Rotor Wing",
            "Type 4 Rotor Wing",
            "Type 2 Standard",
            "Type 3 Standard",
        ]

        type_1_truck = [
            "Type 1",
            "Type 2",
            "Type 3",
            "Type 4",
            "Type 2 Tactical",
            "Type 1 or 2",
            "Tender, Water - Type 1 (CAL FIRE HE Only)",
            "Tender, Water - Type Any (CAL FIRE HE Only)",
            "Type 3 Support",
        ]
        type_2_truck = [
            "Type 5",
            "Type 6",
            "Type 7",
            "Engine, Strike Team",
            "Type Any",
            "Type Any Tactical",
        ]

        heavy_equipment = [
            "Dozer",
            "Dozer, Strike Team",
            "Chipper",
            "Excavator",
            "Skidgine",
            "Skidder",
            "Masticator",
            "Road Grader",
            "Feller Buncher",
            "Pumper Cat",
            "Tractor",
        ]
        processeddat_req = processeddat_req.reset_index()
        processeddat_req.loc[
            processeddat_req["Type"].isin(["Helicopter Crewmember"]), "ReqType"
        ] = "Helicopter"
        processeddat_req.loc[
            processeddat_req["IrocRequestID"].str.contains("O-", na=False), "ReqType"
        ] = "Overhead"

        ## Write broad category

        processeddat_req["ReqType"] = np.NaN
        processeddat_req["ReqTypeCategory"] = "Other"

        processeddat_req.loc[
            processeddat_req["IrocRequestID"].str.contains("C-", na=False), "ReqType"
        ] = "Firefighter"
        processeddat_req.loc[
            processeddat_req["Type"].str.contains("Smokejumper", na=False), "ReqType"
        ] = "Smokejumper"
        processeddat_req.loc[
            processeddat_req["IrocRequestID"].str.contains("E-", na=False), "ReqType"
        ] = "Engine"
        processeddat_req.loc[
            processeddat_req["IrocRequestID"].str.contains("A-", na=False)
            & ~processeddat_req["Category"].isin(["Helicopter"]),
            "ReqType",
        ] = "Aircraft"
        processeddat_req.loc[
            (processeddat_req["IrocRequestID"].str.contains("A-", na=False))
            & processeddat_req["Category"].isin(["Helicopter"])
            | processeddat_req["Type"].isin(["Helicopter Crewmember"]),
            "ReqType",
        ] = "Helicopter"
        processeddat_req.loc[
            processeddat_req["Type"].isin(["Helicopter Crewmember"]), "ReqType"
        ] = "Helicopter"
        processeddat_req.loc[
            processeddat_req["ReqType"].isna(), "ReqType"
        ] = processeddat_req.loc[processeddat_req["ReqType"].isna(), "Category"]
        processeddat_req.loc[
            processeddat_req["ReqType"].isin(["Watertender", "Medical", "Engine"]),
            "ReqType",
        ] = "Engine"
        processeddat_req.loc[
            processeddat_req["ReqType"].isin(["Position", "Fire"]), "ReqType"
        ] = "Firefighter"

        processeddat_req.loc[
            processeddat_req["IrocRequestID"].str.contains("O-", na=False), "ReqType"
        ] = "Overhead"

        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Firefighter")
            & (processeddat_req["Type"].isin(type_1_ff)),
            "ReqTypeCategory",
        ] = "Type 1"
        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Firefighter")
            & (processeddat_req["Type"].isin(type_2_ff)),
            "ReqTypeCategory",
        ] = "Type 2"

        processeddat_req.loc[
            processeddat_req["ReqType"] == "Smokejumper", "ReqTypeCategory"
        ] = "Type 1"

        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Engine")
            & (processeddat_req["Type"].isin(type_1_truck)),
            "ReqTypeCategory",
        ] = "Type 1"
        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Engine")
            & (processeddat_req["Type"].isin(type_2_truck)),
            "ReqTypeCategory",
        ] = "Type 2"

        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Aircraft")
            & (processeddat_req["Type"].isin(type_1_air)),
            "ReqTypeCategory",
        ] = "Type 1"
        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Aircraft")
            & (processeddat_req["Type"].isin(type_2_air)),
            "ReqTypeCategory",
        ] = "Type 2"

        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Helicopter")
            & (processeddat_req["Type"].isin(type_1_heli)),
            "ReqTypeCategory",
        ] = "Type 1"
        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Helicopter")
            & (processeddat_req["Type"].isin(type_2_heli)),
            "ReqTypeCategory",
        ] = "Type 2"

        processeddat_req.loc[
            processeddat_req["Type"].str.contains("Smokejumper", na=False), "ReqType"
        ] = "Smokejumper"
        processeddat_req.loc[
            processeddat_req["IrocRequestID"].str.contains("E-", na=False), "ReqType"
        ] = "Engine"
        processeddat_req.loc[
            processeddat_req["IrocRequestID"].str.contains("A-", na=False)
            & ~processeddat_req["Category"].isin(["Helicopter"]),
            "ReqType",
        ] = "Aircraft"
        processeddat_req.loc[
            (processeddat_req["IrocRequestID"].str.contains("A-", na=False))
            & processeddat_req["Category"].isin(["Helicopter"])
            | processeddat_req["Type"].isin(["Helicopter Crewmember"]),
            "ReqType",
        ] = "Helicopter"
        processeddat_req.loc[
            processeddat_req["Type"].isin(["Helicopter Crewmember"]), "ReqType"
        ] = "Helicopter"
        processeddat_req.loc[
            processeddat_req["IrocRequestID"].str.contains("O-", na=False), "ReqType"
        ] = "Overhead"
        processeddat_req.loc[
            processeddat_req["Type"].str.contains("Smokejumper", na=False), "ReqType"
        ] = "Smokejumper"

        processeddat_req.loc[
            processeddat_req["ReqType"] == "Smokejumper", "ReqTypeCategory"
        ] = "Type 1"
        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Firefighter")
            & (processeddat_req["Type"].isin(type_1_ff)),
            "ReqTypeCategory",
        ] = "Type 1"
        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Firefighter")
            & (processeddat_req["Type"].isin(type_2_ff)),
            "ReqTypeCategory",
        ] = "Type 2"

        processeddat_req.loc[
            processeddat_req["ReqType"] == "Smokejumper", "ReqTypeCategory"
        ] = "Type 1"

        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Engine")
            & (processeddat_req["Type"].isin(type_1_truck)),
            "ReqTypeCategory",
        ] = "Type 1"
        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Engine")
            & (processeddat_req["Type"].isin(type_2_truck)),
            "ReqTypeCategory",
        ] = "Type 2"

        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Aircraft")
            & (processeddat_req["Type"].isin(type_1_air)),
            "ReqTypeCategory",
        ] = "Type 1"
        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Aircraft")
            & (processeddat_req["Type"].isin(type_2_air)),
            "ReqTypeCategory",
        ] = "Type 2"

        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Helicopter")
            & (processeddat_req["Type"].isin(type_1_heli)),
            "ReqTypeCategory",
        ] = "Type 1"
        processeddat_req.loc[
            (processeddat_req["ReqType"] == "Helicopter")
            & (processeddat_req["Type"].isin(type_2_heli)),
            "ReqTypeCategory",
        ] = "Type 2"

        smokejumper = processeddat_req["Type"].str.contains("Smokejumper", na=False)
        fft1 = processeddat_req["Kind"].isin(
            ["Crews", "Overhead Group", "Overhead"]
        ) & (processeddat_req["Type"].isin(type_1_ff))
        fft2 = processeddat_req["Kind"].isin(
            ["Crews", "Overhead Group", "Overhead"]
        ) & ~processeddat_req["Type"].isin(type_1_ff) & ~processeddat_req[
            "Category"
        ].isin(
            ["Non-Fire"]
        ) | (
            processeddat_req["Kind"].isin(["Overhead"])
            & processeddat_req["Type"].isin(["Firefighter Type 2", "Fuels"])
        )
        fireline = processeddat_req["Kind"].isin(["Overhead"]) & processeddat_req[
            "Type"
        ].isin(type_faller_ff)

        processeddat_req.loc[
            processeddat_req["ReqType"].isna(), "ReqType"
        ] = processeddat_req.loc[processeddat_req["ReqType"].isna(), "Category"]
        processeddat_req.loc[
            processeddat_req["ReqType"].isin(["Watertender", "Medical", "Engine"]),
            "ReqType",
        ] = "Engine"
        processeddat_req.loc[
            processeddat_req["ReqType"].isin(["Position", "Fire"]), "ReqType"
        ] = "Firefighter"

        ###### PART III - counter-version

        # Need to count the number of each subtype of resource and turn it into a sum by row - complicating this is a demobilization date that deducts resources as well.

        # force type to string
        processeddat_req[["IrwinID", "ReqType", "ReqTypeCategory"]] = processeddat_req[
            ["IrwinID", "ReqType", "ReqTypeCategory"]
        ].astype(str)
        processeddat_req = processeddat_req.sort_values(["RequestCreation"])
        processeddat_req["RequestCreation"] = pd.to_datetime(
            processeddat_req["RequestCreation"]
        )

        # write out any rows with unrealistic datetimes for departure
        processeddat_req.loc[
            (processeddat_req["DemobETA"] < pd.to_datetime("2020-01-01"))
            | (processeddat_req["DemobETD"] < pd.to_datetime("2020-01-01")),
            "DemobETA",
        ] = pd.NaT

        # begin using the data in object
        full_data = self.study_data.sort_values(["date"])
        processeddat_req.loc[
            processeddat_req["ReqTypeCategory"].isna(), "ReqTypeCategory"
        ] = "Other"
        lookup = (
            processeddat_req.groupby(
                ["IrwinID", "RequestCreation", "ReqType", "ReqTypeCategory"]
            )
            .size()
            .to_frame("GroupCounts")
        )

        # Now, melt data down into a lookup table that increments things up and down based on the event at hand

        # add a day to demobilization date to make sure we don't send someone home early
        processeddat_req.loc[
            processeddat_req["ReqTypeCategory"].isna(), "ReqTypeCategory"
        ] = "Other"

        processeddat_req.reset_index()[
            ["IrwinID", "RequestCreation", "ReqType", "ReqTypeCategory"]
        ].dtypes
        processeddat_req = processeddat_req.reset_index()

        processeddat_req[["IrwinID", "ReqType", "ReqTypeCategory"]] = processeddat_req[
            ["IrwinID", "ReqType", "ReqTypeCategory"]
        ].astype(str)
        processeddat_req.loc[
            processeddat_req["ReqTypeCategory"].isna(), "ReqTypeCategory"
        ] = "Other"
        processeddat_req["ReqTypeCategory"].fillna(inplace=True, value="Other")
        lookup = (
            processeddat_req.groupby(
                ["IrwinID", "RequestCreation", "ReqType", "ReqTypeCategory"]
            )
            .size()
            .to_frame("GroupCounts")
        )

        countervers = processeddat_req.set_index(
            ["IrwinID", "RequestCreation", "ReqType", "ReqTypeCategory"]
        ).join(lookup)
        # print(countervers)
        countervers.DemobETA += pd.Timedelta(days=1)

        # create temp dataframe that is a melted version of counter-version allowing a rollable sum grouped by IrwinID
        df = pd.melt(
            countervers.reset_index()[
                [
                    "IrwinID",
                    "IrwinRID",
                    "RequestCreation",
                    "DemobETD",
                    "ReqType",
                    "ReqTypeCategory",
                    "IrocRequestID",
                ]
            ],
            id_vars=[
                "IrwinID",
                "IrwinRID",
                "ReqType",
                "ReqTypeCategory",
                "IrocRequestID",
            ],
            var_name="change",
            value_name="date",
        )
        df.loc[df["date"] < pd.to_datetime("2020-03-05")] = pd.NaT
        df = df.sort_values("date")
        # rm(countervers)
        df["change"] = df["change"].replace({"RequestCreation": 1, "DemobETD": -1})

        df = df.loc[~df.date.isna()]

        df = df.loc[~df["ReqType"].isna()]

        df["ReqFullType"] = df[["ReqType", "ReqTypeCategory"]].agg("-".join, axis=1)
        df["change"] = pd.to_numeric(df["change"])
        df["count"] = df.groupby(["IrwinID", "ReqFullType"])["change"].cumsum()
        df_parts = []
        df.date = df.date.dt.date
        # start by looping over all IrwinIDs
        for IrwinID, group in df.groupby("IrwinID"):
            new_time = pd.date_range(group.date.min(), group.date.max())
            full_count = pd.DataFrame()
            full_count = full_count.reindex(new_time)

            # now loop over every request type
            for ReqType, subgroup in group.groupby("ReqFullType"):
                if ~subgroup.empty:
                    # print(subgroup[["date", "count"]].set_index("date"))
                    full_count[ReqType] = (
                        subgroup[["date", "count"]]
                        .groupby("date")
                        .max()
                        .reset_index()
                        .set_index("date")
                    )
                    full_count = full_count.ffill().fillna(0)

            # label the subset df with IrwinID
            full_count["IrwinID"] = IrwinID
            df_parts.append(full_count)

        df_full = pd.concat(df_parts)
        df_full = df_full.fillna(0)

        time_series2 = full_data[
            [
                "IrwinID",
                "date",
                "Fatalities",
                "Injuries",
                "PercentContained",
                "OtherStructuresThreatened",
                "OtherStructuresDestroyed",
                "IsFireCauseInvestigated",
                "IsFSAssisted",
                "IsInitialFireStrategyMet",
                "IsUnifiedCommand",
                "ResidencesThreatened",
                "ResidencesDestroyed",
                "EstimatedFinalCost",
                "EstimatedCostToDate",
                "CalculatedAcres",
                "DailyAcres",
                "TotalIncidentPersonnel",
            ]
        ]

        time_series2.loc["date"] = pd.to_datetime(time_series2["date"])
        time_series2 = time_series2.groupby(["IrwinID", "date"]).agg("mean")

        time_series2 = time_series2.reset_index()

        time_series_out = time_series2.merge(
            df_full.reset_index(),
            how="left",
            left_on=["IrwinID", "date"],
            right_on=["IrwinID", "index"],
        )
        # print(time_series_out.columns.values)
        total_frame = (
            time_series_out[
                [
                    "Overhead-Other",
                    "Aircraft-Type 2",
                    "Engine-Type 1",
                    "Engine-Type 2",
                    "Firefighter-Other",
                    "Helicopter-Type 2",
                    "Firefighter-Type 1",
                    "Aircraft-Other",
                    "Engine-Other",
                    "Firefighter-Type 2",
                    "Helicopter-Type 1",
                    "Smokejumper-Type 1",
                    "Helicopter-Other",
                    "Aircraft-Type 1",
                    "TotalIncidentPersonnel",
                    "IrwinID",
                    "date",
                ]
            ]
            .groupby("date")
            .agg("sum")
        )

        out2 = []
        for IrwinID, group in time_series_out[
            [
                "Overhead-Other",
                "Aircraft-Type 2",
                "Engine-Type 1",
                "Engine-Type 2",
                "Firefighter-Other",
                "Helicopter-Type 2",
                "Firefighter-Type 1",
                "Aircraft-Other",
                "Engine-Other",
                "Firefighter-Type 2",
                "Helicopter-Type 1",
                "Smokejumper-Type 1",
                "Helicopter-Other",
                "Aircraft-Type 1",
                "IrwinID",
                "date",
                "TotalIncidentPersonnel",
            ]
        ].groupby("IrwinID"):
            colnames = group.columns.values
            colnames_i = [col + "-i" for col in colnames]
            colnames_i.remove("date-i")
            colnames_i.remove("IrwinID-i")
            daterange = pd.date_range(
                start=group.date.min(),
                end=group.date.max() + pd.Timedelta(days=1),
                freq="D",
            )

            diff = total_frame.loc[
                (total_frame.index.isin(daterange)),
                ~total_frame.columns.isin(["IrwinID"]),
            ].subtract(
                group.loc[:, ~group.columns.isin(["IrwinID"])].set_index("date"),
                fill_value=0,
            )
            # print(diff.columns.values)
            diff.columns = colnames_i
            diff = diff.reset_index()
            diff["IrwinID"] = IrwinID

            group = group.merge(diff[~diff.isna()], how="left", on=["date"])
            group[["IrwinID"]] = IrwinID
            out2.append(diff)

        out_frames = pd.concat(out2)
        time_series_out2 = (
            time_series_out.set_index(["IrwinID", "date"])
            .join(out_frames.set_index(["IrwinID", "date"]), how="left", lsuffix="_x")
            .sort_values("date")
        )

        self.time_series = time_series_out2
        # rm(time_series_out2)
        self.tabular_fixed_effects = full_data[
            [
                "AdditionalFuelModel",
                "POOLandownerKind",
                "POOState",
                "IrwinID",
                "FinalAcres",
                "WFDSSStrategySliderValue",
                "PredominantFuelGroup",
                "POOProtectingAgency",
                "GACC",
                "InitialFireStrategy",
                "InitialResponseAcres",
                "EstimatedFinalCost",
            ]
        ].drop_duplicates(subset="IrwinID", keep="last")

        to_dummies = [
            "InitialFireStrategy",
            "GACC",
            "POOProtectingAgency",
            "PredominantFuelGroup",
            "POOState",
            "POOLandownerKind",
            "AdditionalFuelModel",
        ]

        dummies_tabular_fixed_effects = pd.get_dummies(
            self.tabular_fixed_effects[to_dummies], dummy_na=True
        )
        self.tabular_fixed_effects.drop(columns=to_dummies, inplace=True)

        self.tabular_fixed_effects = pd.concat(
            [self.tabular_fixed_effects, dummies_tabular_fixed_effects], axis=1
        )

        # rm(full_data)
        fpath = os.path.join(destination_folder, "time_series.pckl")
        self.time_series.to_pickle(fpath, compression=None)

        fpath = os.path.join(destination_folder, "tabular_fixed_effects.pckl")
        self.tabular_fixed_effects.to_pickle(fpath, compression=None)


if __name__ == "__main__":

    import argparse

    Irwin_FP = "/IrwinHistoricData/IrwinHistoricData.csv"

    RasterIn = "/MassCrop/InputRasters"

    RasterOut = "/MassCrop/OutputFolder/raster"

    TrasterIn = "/MassCrop/traster"

    TrasterOut_viirs = "/MassCrop/OutputFolder/traster/viirs/"

    TrasterOut_noaa = "/MassCrop/OutputFolder/traster/noaa/"

    IrwinReq_FP = "home/IrwinResource_Request.csv"

    IrwinTS = "/MassCrop/OutputFolder/tabular"

    StartDate = "2019-12-15"

    parser = argparse.ArgumentParser()

    parser.add_argument("--Irwin_FP", type=str, default=Irwin_FP)
    parser.add_argument("--RasterIn", type=str, default=RasterIn)
    parser.add_argument("--TrasterIn", type=str, default=TrasterIn)
    parser.add_argument("--destination_folder", type=str, default=RasterOut)

    Irwin_FP = parser.parse_args(["--Irwin_FP"])
    RasterIn = parser.parse_args(["--RasterIn"])
    TrasterIn = parser.parse_args(["--TrasterIn"])
    destination_folder = parser.parse_args(["--destination_folder"])

    study_data_input = initial_data_processor(
        Irwin_FP, pretraining=True, from_disk=True
    )

    print(study_data_input[1, :])

    databuilder = DatasetBuilder(
        raster=RasterIn, traster=TrasterIn, timeseries="", study_data=study_data_input
    )
    databuilder._t_raster_maker(
        destination_folder_viirs="/home/connor/Desktop/MassCrop/OutputFolder/traster/viirs",
        destination_folder_noaa="/home/connor/Desktop/MassCrop/OutputFolder/traster/noaa",
        buffer_meter=15000,
    )
    #
    databuilder.time_series_data(
        destination_folder="/home/connor/Desktop/MassCrop/OutputFolder/tabular",
        requests_data="",
    )
    databuilder._raster_maker(
        destination_folder=destination_folder, buffer_meter=15000, exclude=True
    )

    print("You made it!")
