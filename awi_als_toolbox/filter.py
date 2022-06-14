# -*- coding: utf-8 -*-

"""
A module containing filter algorithm for the ALS point cloud data.
"""

__author__ = "Stefan Hendricks"

import os
import numpy as np
import bottleneck as bn

from datetime import datetime, timedelta

import pyproj
from loguru import logger
from scipy.interpolate import interp1d
from pathlib import Path
import pandas as pd
from typing import Union, Tuple, Dict

from . import ALSPointCloudData

from icedrift import GeoReferenceStation, IceCoordinateSystem, GeoPositionData


class ALSPointCloudFilter(object):
    """ Base class for point cloud filters """

    def __init__(self, **kwargs) -> None:
        """
        Init the ALSPointCloudFilter base class.

        :param kwargs: Keyword arguments that will be stored in a config dictionary
        """
        self.cfg = kwargs

    def apply(self, als: "ALSPointCloudData") -> None:
        """
        This mandatory methods must be overwritten by inheriting classes

        :param als: The als point cloud data, will be changed in-place
        :return: None
        """
        cls_name = self.__class__.__name__
        msg = (
            f"{cls_name} does not implement mandatory method `apply()`"
            if cls_name != "ALSPointCloudData" else
            "ALSPointCloudData should not be called directly"
        )
        raise NotImplementedError(msg)


class AtmosphericBackscatterFilter(ALSPointCloudFilter):
    """
    Identifies and removes target echoes that are presumably within the atmosphere
    based on elevation statistics for each line.
    """

    def __init__(self, filter_threshold_m: float = 5.) -> None:
        """
        Initialize the filter.

        :param filter_threshold_m:
        """
        super(AtmosphericBackscatterFilter, self).__init__(filter_threshold_m=filter_threshold_m)

    def apply(self, als: "ALSPointCloudData") -> None:
        """
        Apply the filter for all lines in the ALS data container

        :param als:
        :return:
        """

        # Filter points outside the [-threshold, threshold] interval around the 
        # first mode of elevations
        elevations = als.get('elevation')

        # Determine elevation of first mode
        hist, bins = np.histogram(elevations[np.isfinite(elevations)], bins=100)
        diff = np.diff(np.append(np.zeros((1,)), hist))
        if np.any(np.all([diff[1:] < 0, diff[:-1] > 0], axis=0)):
            ind_peak = np.where(np.all([diff[1:] < 0, diff[:-1] > 0], axis=0))[0][0]
            min_mode_elev = np.mean(bins[ind_peak:ind_peak + 2])
        else:
            min_mode_elev = bn.nanmean(elevations)
        threshold = 20

        # Mask points outside the interval
        mask = np.where(np.any([elevations > min_mode_elev + threshold,
                                elevations < min_mode_elev - threshold], axis=0))
        elevations[mask] = np.nan
        als.set("elevation", elevations)

        for line_index in np.arange(als.n_lines):
            # 1  Compute the median elevation of a line
            elevation = als.get("elevation")
            elevations = elevation[line_index, :]
            line_median = np.nanmedian(elevations)

            # 2. Fill nan values with median elevation
            # This is needed for spike detection
            elevations_nonan = np.copy(elevations)
            elevations_nonan[np.isnan(elevations_nonan)] = line_median

            # Search for sudden changes (spikes)
            spike_indices = self._get_filter_indices(elevations_nonan, self.cfg["filter_threshold_m"])

            # Remove spiky elevations
            elevation[line_index, spike_indices] = np.nan
            als.set("elevation", elevation)

    @staticmethod
    def _get_filter_indices(vector: np.ndarray, filter_treshold: float) -> np.ndarray:
        """
        Compute the indices of potential spikes and return a list of indices

        :param vector:
        :param filter_treshold:
        :return:
        """

        # Compute index-wise change in data
        diff = vector[1:] - vector[:-1]

        # Compute change of data point to both directions
        diff_right = np.full(vector.shape, np.nan)
        diff_left = np.full(vector.shape, np.nan)
        diff_right[1:] = diff
        diff_left[:-1] = diff

        # Check for data change exceeds the filter threshold
        right_threshold = np.abs(diff_right) > filter_treshold
        left_threshold = np.abs(diff_left) > filter_treshold

        # Check where data point is local extrema
        is_local_extrema = np.not_equal(diff_right > 0, diff_left > 0)
        condition1 = np.logical_and(right_threshold, left_threshold)

        # point is spike: if change on both sides exceeds threshold and is local
        # extrema
        is_spike = np.logical_and(condition1, is_local_extrema)
        return np.where(is_spike)[0]


# TODO: This class needs to be refactored (different sources for reference stations)
# TODO: Add option to directly supply GeoReferencestation data as csv file
# TODO: Update icedrift API
# TODO: Ensure that icedrift and floenavi are not required imports
# TODO: Move functionality to get floenavi master solutions into the floenavi package
class IceDriftCorrection(ALSPointCloudFilter):
    """
    Corrects for ice drift during data aquisition, using floenavi or Polarstern position
    """

    def __init__(self,
                 refstation_file: str = None,
                 proj4: str = "auto",
                 reference_time: Union[str, datetime] = "auto",
                 ):
        """
        This filter applies a coordinate transformation to account for sea ice drift during the survey.
        The coordinate transformation is handled by the python package `icedrift`
        (https://gitlab.awi.de/floenavi-crs/icedrift).

        Applying the filter will add (x, y) coordinates in a translating & rotating cartesian reference
        frame (ice coordinate system). The (longitude, latitude) values are also drift corrected as if
        the entire ALS survey was done instantenously.

        The underlying projection for (x, y) coordinates is chosen automatically by the `icedrift`
        module, but can be overwritten with a proj4 string.

        This filter also chooses the reference time for the ice drift correction by the reference time
        from the temporal coverage of the reference station data. It can also be overwritten with
        a datetime object. In this case the user needs to take care that the custom reference time
        is within the bounds of the reference station data.

        :param refstation_file: absolute file path to icedrift reference station file.
            The data for the ice drift correction needs to be a csv file compatible with `
            icedrift.GeoReferenceStation.from_csv()`. At a minimum, the following info needs to be included:

                time,longitude,latitude,heading
                2019-12-29 23:53:31,116.15182,86.62111,202.3
                ...

        :param proj4: (default: auto) The projection used for the icedrift correction. If auto, then
            the projection will be chosen by `icedrift`
        :param reference_time: (defaut: auto) A datetime object. If default value, then the reference time
            will be set to the reference time of the first segment processed with this filter.
        """

        # TODO: Needs validation
        kwargs = {
            "refstation_file": refstation_file,
            "proj4": proj4,
            "reference_time": reference_time
        }

        super(IceDriftCorrection, self).__init__(**kwargs)

        # Init the ice coordinate system
        projection = None if proj4 == "auto" else pyproj.Proj(proj4)
        refstat = GeoReferenceStation.from_csv(refstation_file)
        self.icecs = IceCoordinateSystem(refstat, projection=projection)

        # Reference time
        self.reference_time = reference_time

    def apply(self, als: "ALSPointCloudData") -> None:
        """
        Apply the ice drift correction for all echos.

        :param als: ALS point-cloud data object

        :return: None, als data is changed in place
        """

        logger.info("IceDriftCorrection is applied")

        # Determine reference time
        # NOTE: This method is called for a segment of a single file, but this class remains
        #       instanced for all segments for a batch run. Thus, if no specific reference
        #       time for the ice drift correction is specified, the reference time of the
        #       first segment is stored in this instance and used for all other following
        #       filter applications.
        if self.reference_time is None:
            self.reference_time = als.ref_time

        # Convert the als echoes into an `icedrift.GeoPositionData` instance.
        # NOTE: Not all entries in the ALS point cloud data have valid longitude, latitude values.
        #       The GeoPositionData contains only valid entries and
        als_geo_pos, als_idx = self.get_als_geopos(als)

        # Init (x, y) coordinates
        x = np.full(als.dims, np.nan)
        y = x.copy()
        lon = x.copy()
        lat = x.copy()

        # Compute (x, y) ice coordinates
        icepos, icecs_transform = self.icecs.get_xy_coordinates(als_geo_pos, transform_output=True)
        x[als_idx], y[als_idx] = icepos.xc, icepos.yc

        # Update longitude, latitude values to reflect the (x, y) positions in the
        # ice coordinate system at the shared reference time
        reference_time = self.reference_time if self.reference_time not in ["auto", None] else als.ref_time_dt
        geopos = self.icecs.get_latlon_coordinates(icepos.xc, icepos.yc, reference_time)
        lon[als_idx], lat[als_idx] = geopos.longitude, geopos.latitude

        # Get the projection parameters
        projection_attrs = self.get_projection_attrs(icecs_transform)
        projection_dict = dict(name=projection_attrs['grid_mapping_name'], attrs=projection_attrs)

        # Update the ALS point cloud dataset
        als.set_icedrift_correction(x, y, lon, lat, projection_dict, self.reference_time)

    @staticmethod
    def get_als_geopos(als: "ALSPointCloudData") -> Tuple["GeoPositionData", np.ndarray]:
        """
        Return an icedrift.GeoPositionData as well as the list of indices of valid echoes

        :param als: The point cloud data

        :return icedrift.GeoPositionData, indices that maps geoposition data to als indices)
        """
        # Transform only valid positions
        nonan = np.where(np.logical_or(np.isfinite(als.get("longitude")), np.isfinite(als.get("latitude"))))

        # Generate GeoPositionData object from als
        epoch = datetime(1970, 1, 1, 0, 0, 0)
        time_als = np.array([epoch + timedelta(0, isec) for isec in als.get("timestamp")[nonan]])
        als_geo_pos = GeoPositionData(time_als, als.get("longitude")[nonan], als.get("latitude")[nonan])
        return als_geo_pos, nonan

    @staticmethod
    def get_projection_attrs(icecs_transform) -> Dict:
        """
        Extract and return the projection parameters used for the icedrift
        coordinate transformation as a dictionary
        TODO: This could be moved to the icedrift package
        """
        ikeys = [ik for ik in icecs_transform.prj.crs.to_cf().keys() if ik != 'crs_wkt']
        attrs = {ikey: icecs_transform.prj.crs.to_cf()[ikey] for ikey in ikeys}
        attrs['proj4_string'] = icecs_transform.prj.srs
        return attrs


class OffsetCorrectionFilter(ALSPointCloudFilter):
    """
    Reads in offset terms (for elevation) computed while gridding the floe
    grid and subtracts them from the variable field
    """

    def __init__(self, export_file='_correction.csv'):
        """
        Initialize the filter.

        :param export_file:
        """
        super(OffsetCorrectionFilter, self).__init__(export_file='_correction.csv')
        self.corr_files = None

    def apply(self, als):
        """
        Apply the filter for all lines in the ALS data container
        :param als:
        :return:
        """

        # Check for correction files stored
        self.corr_files = [ifile for ifile in os.listdir('./') if ifile.endswith(self.cfg['export_file'])]

        if not self.corr_files:
            logger.info('ELEVCOR: Warning - OffsetCorrectionFilter called without providing correction files')

        # Apply correction to als object
        for icor in self.corr_files:
            fpath = Path(icor).absolute()
            variable = icor.split(self.cfg['export_file'])[0]
            logger.info(f"Apply offset correction for: {variable} with file:{fpath}")

            # Read offset correction file
            df = pd.read_csv(fpath)
            t = np.array(df['timestamp'])
            c = np.array(df[f'{variable}_offset'])

            # Set-up interpolation function
            func = interp1d(t - t[0], c, kind='linear', bounds_error=False,
                            fill_value=(c[0], c[-1]))

            # Apply ALS binary file
            data = als.get(variable)
            cor_data = data - func(als.get("timestamp") - t[0])
            logger.info("    mean correction: %.05f" % np.nanmean(data - cor_data))
            als.set(variable, cor_data)
