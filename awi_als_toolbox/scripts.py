# -*- coding: utf-8 -*-

"""
Module that contains functions for standardized ALS processing workflows. These are
meant to be called by more specific scripts.
"""

import sys
import multiprocessing
from pathlib import Path

import matplotlib.pylab as plt

# This matplotlib setting is necessary if the script
# is run in a shell via ssh and no window manager
# import matplotlib
# matplotlib.use("agg")

from loguru import logger

from . import AirborneLaserScannerFile, AirborneLaserScannerFileV2, AlsDEM
from .export import AlsDEMNetCDF
import awi_als_toolbox.freeboard as freeboard
from awi_als_toolbox.filter import OffsetCorrectionFilter


def als_l1b2dem(als_filepath,
                dem_cfg,
                output_cfg,
                file_version=1,
                use_multiprocessing=False,
                mp_reserve_cpus=2):
    """
    Grid a binary point cloud file with given grid specification and in segments of
    a given temporal coverage
    :param als_filepath: (str, pathlib.Path): The full filepath of the binary ALS point cloud file
    :param dem_cfg: (awi_als_toolbox.demgen.AlsDEMCfg):
    :param output_cfg:
    :param file_version:
    :param use_multiprocessing:
    :param mp_reserve_cpus:
    :return:
    """

    # Get ALS file
    als_filepath = Path(als_filepath)
    alsfile = get_als_file(als_filepath, file_version, dem_cfg)

    # --- Step 3: loop over the defined segments ---
    # Get a segment list based on the suggested segment lengths for the gridding preset
    # TODO: Evaluate the use of multi-processing for the individual segments.
    segments = alsfile.get_segment_list(dem_cfg.segment_len_secs)
    n_segments = len(segments)
    logger.info("Split file in %d segments" % n_segments)

    # Substep (Only valid if multi-processing should be used
    process_pool = None
    if use_multiprocessing:
        # Estimate how much workers can be added to the pool
        # without overloading the CPU
        n_processes = multiprocessing.cpu_count()
        n_processes -= mp_reserve_cpus
        n_processes = max(n_processes, 1)
        # Create process pool
        logger.info(f"Use multi-processing with {n_processes} workers")
        process_pool = multiprocessing.Pool(n_processes)

    # Grid the data and write the output in a netCDF file
    # This can either be run in parallel or
    if use_multiprocessing:
        # Parallel processing of all segments
        results = [process_pool.apply_async(read_grid_wrapper, args=(als_filepath, dem_cfg, 
                                                                     output_cfg, file_version,
                                                                     start_sec, stop_sec, 
                                                                     i, n_segments)) 
                   for i, (start_sec, stop_sec) in enumerate(segments)]
        result = [iresult.get() for iresult in results]
    else:
        # Loop over all segments
        for i, (start_sec, stop_sec) in enumerate(segments):
            read_grid_wrapper(als_filepath, dem_cfg, output_cfg, file_version,
                              start_sec, stop_sec, i, n_segments)

    if use_multiprocessing:
        process_pool.close()
        process_pool.join()

    
def get_als_segments(als_filepaths, dem_cfg, file_version=1):
    """
    Function to return segements of all binary ALS point cloud files provided

    :param als_filepaths: list of full filepaths of all binary ALS point could files
    :param dem_cfg: (awi_als_toolbox.demgen.AlsDEMCfg)
    :param file_version:
    :return:"
    """
    output = {'als_filepath': [],
              'start_sec': [],
              'stop_sec': [],
              'i': [],
              'n_segments': []}
    
    for ifile in als_filepaths:
        als_filepath = Path(ifile)
        alsfile = get_als_file(als_filepath, file_version, dem_cfg)

        # --- Step 3: loop over the defined segments ---
        # Get a segment list based on the suggested segment lengths for the gridding preset
        # TODO: Evaluate the use of multi-processing for the individual segments.
        segments = alsfile.get_segment_list(dem_cfg.segment_len_secs)
        n_segments = len(segments)
        logger.info("Split file in %d segments" % n_segments)

        for i, (start_sec, stop_sec) in enumerate(segments):
            output['als_filepath'].append(als_filepath)
            output['start_sec'].append(start_sec)
            output['stop_sec'].append(stop_sec)
            output['i'].append(i)
            output['n_segments'].append(n_segments)
    
    logger.info("Overall number of segments: %i" % len(output['i']))
    
    return output
    
        
def get_als_file(als_filepath, file_version, dem_cfg):
    """
    Open a binary point cloud file with given grid specification and slice in segments of
    a given temporal coverage

    :param als_filepath: (str, pathlib.Path): The full filepath of the binary ALS point cloud file
    :param dem_cfg: (awi_als_toolbox.demgen.AlsDEMCfg):
    :param file_version:
    :return: awi_als_toolbox.ALSPointCloudData
    """

    # --- Step 1: connect to the ALS binary point cloud file ---
    #
    # At the moment there are two options:
    #
    #   1) The binary point cloud data from the "als_level1b" IDL project.
    #      The output is designated as file version 1
    #
    #   2) The binary point cloud data from the "als_level1b_seaice" IDL project.
    #      The output is designated as file version 2 and can be identified
    #      by the .alsbin2 file extension

    # Input validation
    als_filepath = Path(als_filepath)
    if not als_filepath.is_file():
        logger.error(f"File does not exist: {str(als_filepath)}")
        sys.exit(1)

    # Connect to the input file
    # NOTE: This step will not read the data, but read the header metadata information
    #       and open the file for sequential reading.
    logger.info(f"Open ALS binary file: {als_filepath.name} (file version: {file_version})")

    if file_version == 1:
        alsfile = AirborneLaserScannerFile(als_filepath, **dem_cfg.connect_keyw)
    elif file_version == 2:
        alsfile = AirborneLaserScannerFileV2(als_filepath)
    else:
        logger.error(f"Unknown file format: {dem_cfg.input.file_version}")
        sys.exit(1)

    return alsfile
        
    
def read_grid_wrapper(als_filepath, dem_cfg, output_cfg, file_version, start_sec, stop_sec, i, n_segments):
    """
    Wrapper of reading and gridding_workflow. May be joined with gridding_workflow in the future
    """
    
    # Get ALS file
    alsfile = get_als_file(als_filepath, file_version, dem_cfg)
    
    # Extract the segment
    # NOTE: This includes file I/O
    logger.info("Processing %s [%g:%g] (%g/%g)" % (als_filepath.name, start_sec, stop_sec, i+1, n_segments))
    als = alsfile.get_data(start_sec, stop_sec)

    # TODO: Replace with try/except with actual Exception
    # except BaseException:
    #     msg = "Unhandled exception while reading %s:%g-%g -> Skip segment"
    #     logger.error(msg % (als_filepath.name, start_sec, stop_sec))
    #     print(sys.exc_info()[1])
    #     continue

    # Apply any filter defined
    for input_filter in dem_cfg.get_input_filter():
        input_filter.apply(als)
        
    # Apply freeboard conversion
    if 'freeboard' in output_cfg.variable_attributes.keys():
        # Apply offset correction
        ocf = OffsetCorrectionFilter()
        ocf.apply(als)
        # Apply freeboard computation
        als_freeboard = freeboard.AlsFreeboardConversion(cfg=dem_cfg.freeboard)
        als_freeboard.freeboard_computation(als,dem_cfg=dem_cfg)
        
        # fig,ax = plt.subplots(1,1)
        # ax.pcolormesh(als.get('elevation'),vmin=-3,vmax=3)

    # Validate segment
    # -> Do not try to grid a segment that has no valid elevations
    if not als.has_valid_data:
        msg = "... No valid data in {}:{}-{} -> skipping segment"
        msg = msg.format(als_filepath.name, start_sec, stop_sec)
        logger.warning(msg)
    else:
        # Grid the data and write the output in a netCDF file
        gridding_workflow(als, dem_cfg, output_cfg)

        
def gridding_workflow(als, dem_cfg, output_cfg):
    """
    Single function gridding and plot creation that can be passed to a multiprocessing process
    :param als: (ALSData) ALS point cloud data
    :param dem_cfg:
    :param output_cfg:
    :return: None
    """

    # Grid the data
    logger.info("... Start gridding")

    dem = AlsDEM(als, cfg=dem_cfg)
    dem.create()

    # try:
    #     dem = AlsDEM(als, cfg=dem_cfg)
    #     dem.create()
    # except:
    #     logger.error("Unhandled exception while gridding -> skip gridding")
    #     print(sys.exc_info()[1])
    #     return
    logger.info("... done")

    # create
    nc = AlsDEMNetCDF(dem, output_cfg)
    nc.export()
    logger.info(f"... exported to: {nc.path}")
