# Copyright (c) 2020 Scott Moisik and Pertti Palo.
#
# When making use of this code, please cite (TODO There might be a more appropriate place for these. Placed here temporarily):
#   [1] Esling, J. H., & Moisik, S. R. (2012). Laryngeal aperture in relation to larynx height change: An analysis using simultaneous laryngoscopy and laryngeal ultrasound. In D. Gibbon, D. Hirst, & N. Campbell (Eds.), Rhythm, melody and harmony in speech: Studies in honour of Wiktor Jassem: Vol. 14/15 (pp. 117–127). Polskie Towarzystwo Fonetyczne.
#   [2] Moisik, S. R., Lin, H., & Esling, J. H. (2014). A study of laryngeal gestures in Mandarin citation tones using simultaneous laryngoscopy and laryngeal ultrasound (SLLUS). Journal of the International Phonetic Association, 44(01), 21–58. https://doi.org/10.1017/S0025100313000327
#   [3] Poh, D. P. Z., & Moisik, S. R. (2019). An acoustic and articulatory investigation of citation tones in Singaporean Mandarin using laryngeal ultrasound. In S. Calhoun, P. Escudero, M. Tabain, & P. Warren (Eds.), Proceedings of the 19th International Congress of the Phonetic Sciences.
#
# This file is part of Pixel Difference toolkit
# (see https://github.com/giuthas/pd/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# The example data packaged with this program is licensed under the
# Creative Commons Attribulton-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License. You should have received a
# copy of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License along with the data. If not,
# see <https://creativecommons.org/licenses/by-nc-sa/4.0/> for details.
#


# TODO : List of non-rigid registration algorithms http://pyimreg.github.io/

from contextlib import closing
from datetime import datetime

# built in packages
import csv
import math
import glob
import logging
import os
import os.path
import pickle
import re
import struct
import sys
import time
import warnings
import wave

from multiprocessing import Process, Manager

# optical flow algorithm implemented in openCV
import cv2

# demons algorithim implemented in the PIRT package with visualization support
#import pirt
#import visvis as vv

# diffeomorphic demons algorithm implemented in python in the DIPY package
#from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
#from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
# from dipy.viz import regtools

# numpy and scipy
import numpy as np
import scipy.io as sio
import scipy.io.wavfile as sio_wavfile
import scipy.ndimage as ndi
# from scipy.signal import butter, filtfilt, kaiser, sosfilt

from scipy import interpolate
#from skimage import transform
#import plotly.graph_objects as go
#import plotly.offline as pyo

# scientific plotting
import matplotlib.image as image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages

# create module logger
ofreg_logger = logging.getLogger('of.ofreg')


def read_prompt(filebase):
    with closing(open(filebase, 'r')) as promptfile:
        lines = promptfile.read().splitlines()
        prompt = lines[0]
        date = datetime.strptime(lines[1], '%d/%m/%Y %I:%M:%S %p')
        participant = lines[2].split(',')[0]

        return prompt, date, participant


def read_wav(filebase):
    samplerate, frames = sio_wavfile.read(filebase)
    # duration = frames.shape[0] / samplerate

    return frames, samplerate


def _parse_ult_meta(filebase):
    """Return all metadata from AAA txt as dictionary."""
    with closing(open(filebase, 'r')) as metafile:
        meta = {}
        for line in metafile:
            (key, value_str) = line.split("=")
            try:
                value = int(value_str)
            except ValueError:
                value = float(value_str)
            meta[key] = value

        return meta


def read_ult_meta(filebase):
    """Convenience fcn for output of targeted metadata."""
    meta = _parse_ult_meta(filebase)

    return (meta["NumVectors"],
            meta["PixPerVector"],
            meta["ZeroOffset"],
            meta["Angle"],
            meta["PixelsPerMm"],
            meta["FramesPerSec"],
            meta["TimeInSecsOfFirstFrame"])


def get_data_from_dir(directory):
    # this is equivalent with the following: sorted(glob.glob(directory + '/.' +  '/*US.txt'))
    ult_meta_files = sorted(glob.glob(directory + '/*US.txt'))

    # this takes care of *.txt and *US.txt overlapping.
    ult_prompt_files = [prompt_file
                        for prompt_file in glob.glob(directory + '/*.txt')
                        if not prompt_file in ult_meta_files
                        ]

    ult_prompt_files = sorted(ult_prompt_files)
    filebases = [os.path.splitext(pf)[0] for pf in ult_prompt_files]
    meta = [{'filebase': filebase} for filebase in filebases]

    # iterate over file base names and check for required files
    for i,fb in enumerate(filebases): 
        # Prompt file should always exist and correspond to the filebase because
        # the filebase list is generated from the directory listing of prompt files.
        meta[i]['ult_prompt_file'] = ult_prompt_files[i]
        (prompt, date, participant) = read_prompt(ult_prompt_files[i])
        meta[i]['prompt'] = prompt
        meta[i]['date'] = date
        meta[i]['participant'] = participant

        # generate candidates for file names
        ult_meta_file = os.path.join(fb + "US.txt")
        ult_wav_file = os.path.join(fb + ".wav")
        ult_file = os.path.join(fb + ".ult")

        # check if assumed files exist, and arrange to skip them if any do not
        if os.path.isfile(ult_meta_file):
            meta[i]['ult_meta_file'] = ult_meta_file
            meta[i]['ult_meta_exists'] = True
        else:
            notice = 'Note: ' + ult_meta_file + " does not exist."
            ofreg_logger.warning(notice)
            meta[i]['ult_meta_exists'] = False
            meta[i]['excluded'] = True

        if os.path.isfile(ult_wav_file):
            meta[i]['ult_wav_file'] = ult_wav_file
            meta[i]['ult_wav_exists'] = True
        else:
            notice = 'Note: ' + ult_wav_file + " does not exist."
            ofreg_logger.warning(notice)
            meta[i]['ult_wav_exists'] = False
            meta[i]['excluded'] = True

        if os.path.isfile(ult_file):
            meta[i]['ult_file'] = ult_file
            meta[i]['ult_exists'] = True
        else:
            notice = 'Note: ' + ult_file + " does not exist."
            ofreg_logger.warning(notice)
            meta[i]['ult_exists'] = False
            meta[i]['excluded'] = True

    meta = sorted(meta, key=lambda item: item['date'])

    return meta


#def parallel_register(ns, index, num_frames, storage):
def parallel_register(im1, im2, index, num_frames, storage):
    sys.stdout.write("Working on frame pair %d of %d\n" % (index, num_frames - 1))
    #current_im = ns.ultra_interp[index]
    #next_im = ns.ultra_interp[index + 1]

    # suppress warnings (from dipy package encouraging us to install "Fury")
    #warnings.filterwarnings("ignore")

    # execute and store the optimization
    #storage[index] = {'of': pirt_reg(current_im, next_im), 'current frame': index, 'next frame': index + 1}
    storage[index] = {'of': pirt_reg(im1, im2), 'current frame': index, 'next frame': index + 1}
    # revert back to always displaying warnings
    #warnings.filterwarnings("always")


#def pirt_reg(im1, im2):
#    # Init registration
#    reg = pirt.OriginalDemonsRegistration(im1, im2)
#    reg.params.scale_levels = 2 #Good: 2
#    reg.params.speed_factor = 3.0 #Good: 3.0
#    reg.params.noise_factor = 0.5 #Good: 0.5
#    reg.params.mapping = 'forward'
#    reg.params.scale_sampling = 5 #Good: 5
##    reg.params.final_grid_sampling = 2 #Good: 2#

    # Register (non-verbose)
#    reg.register(0)

#    return reg.get_deform(0)._fields


def opencv_opticalflow(im1, im2, flow_estimate=None):
    """Computes the dense optical flow between frame pairs using the OpenCV framework"""
    flow_estimate = None # Initial estimate of the flow
    pyr_scale = 0.5 # Image pyramid scaling for multiresolution estimation of optical flow. Specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
    pyr_levels = 3 # Number of levels in the image pyramid
    win_size  = 7 # Averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
    iterations = 100 # Number of iterations at each pyramid level
    poly_n = 5 # Size of pixel neighbourhood to find polynomial expansion in each pixel (larger values mean image is approximated with smoother surfaces, giving more robust algorithm)
    poly_sigma = 1.1 # Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion (examples of good values: for poly_n = 5, use 1.1; for poly_n = 7, use 1.5)

    flags = cv2.OPTFLOW_USE_INITIAL_FLOW if flow_estimate is not None else 0

    return cv2.calcOpticalFlowFarneback(im1, im2, flow_estimate,
                                        pyr_scale, pyr_levels, win_size, iterations,
                                        poly_n, poly_sigma, flags)


def linear_probe_interpolation(ultra, ult_pixels_per_vector, ult_pixels_per_mm, ult_num_vectors, ult_no_frames):
    # interpolate the data to correct the axis scaling for purposes of image registration
    probe_array_length_mm = 40.0  # TODO 40 mm long linear probe assumed!!!
    probe_depth_mm = ult_pixels_per_vector / ult_pixels_per_mm
    length_depth_ratio = probe_depth_mm / probe_array_length_mm

    x = np.linspace(1, ult_num_vectors, ult_num_vectors)
    y = np.linspace(1, ult_pixels_per_vector, ult_pixels_per_vector)

    xnew = np.linspace(1, ult_num_vectors, ult_num_vectors)
    ynew = np.linspace(1, ult_pixels_per_vector, math.ceil(ult_num_vectors * length_depth_ratio))

    ultra_interp = []
    probe_data = {probe_depth_mm, probe_array_length_mm}
    for fIdx in range(0, ult_no_frames):
        f = interpolate.interp2d(x, y, np.transpose(ultra[fIdx, :, :]), kind='linear')
        ultra_interp.append(f(xnew, ynew))

    return ultra_interp, probe_data


def radial_probe_interpolation(ultra, ult_pixels_per_vector, ult_pixels_per_mm, ult_num_vectors, ult_angle, ult_zero_offset, ult_no_frames):
    # First compute the forward mapping from source to destination
    half_fan_rads = ult_num_vectors * ult_angle * 0.5
    right_edge_rads = math.pi * 0.5 - half_fan_rads
    angle_increments = right_edge_rads + np.arange(0, float(ult_num_vectors)) * ult_angle
    pixel_increments = np.arange(0, ult_pixels_per_vector)
    locs_x = np.outer(np.cos(angle_increments), (ult_zero_offset + pixel_increments)) / ult_pixels_per_mm
    locs_y = np.outer(np.sin(angle_increments), (ult_zero_offset + pixel_increments)) / ult_pixels_per_mm

    probe_view_angle = ult_num_vectors * ult_angle * 180 / math.pi
    probe_array_radius_mm = ult_zero_offset / ult_pixels_per_mm
    probe_array_depth_mm = ult_pixels_per_vector / ult_pixels_per_mm
    #TODO Resolution of some US videos can be too high (causing OutOfMemory error when remapping below); this can be solved by downsampling the remapped image
    xdst, ydst = np.meshgrid(np.arange(np.min(locs_x), np.max(locs_x), 1.0), #10.0),
                             np.arange(np.min(locs_y), np.max(locs_y), 1.0)) #10.0))

    # Now compute the inverse mapping of pixels from source to destination (since cv2.remap requires the mapping from dst -> src)
    xdstMm = (xdst * ult_pixels_per_mm)
    ydstMm = (ydst * ult_pixels_per_mm)
    theta = np.arctan2(ydstMm, xdstMm)
    minAngle = np.min(angle_increments)
    maxAngle = np.max(angle_increments)
    thetaScaled = (theta - minAngle) / (maxAngle - minAngle)
    thetaScaled *= ult_num_vectors #TODO: The ultrasound may be a bit off here - perhaps because of how angle_increments is defined
    r = np.sqrt((xdstMm*xdstMm) + (ydstMm*ydstMm)) - ult_zero_offset

    #im = plt.imshow(thetaScaled)
    #plt.colorbar()
    #plt.show()

    ultra_interp = []
    probe_data = {probe_view_angle, probe_array_radius_mm, probe_array_depth_mm}
    for fIdx in range(0, ult_no_frames):

        f = np.flipud(cv2.remap(ultra[fIdx, :, :], r.astype(np.float32), thetaScaled.astype(np.float32), cv2.INTER_CUBIC))
        #plt.imshow(f)
        #plt.show()
        ultra_interp.append(f)

    return ultra_interp, probe_data






def compute(data_list, results_directory):
    # inputs: elements in data dictionary generated by get_data_from_dir
    # i.e. all_data[i] is an item

    toolbar_width = 40
    file_count = 1

    print("Started working on optical flow for " + str(len(data_list)) + " files... Please wait...")

    for item in data_list:
        ofreg_logger.info("PD: " + item['filebase'] + " " + item['prompt'] + '. item processed.')

        (ult_wav_frames, ult_wav_fs) = read_wav(item['ult_wav_file'])
        (ult_num_vectors, ult_pixels_per_vector, ult_zero_offset, ult_angle, ult_pixels_per_mm, ult_fps, ult_time_in_sec_of_first_frame) = read_ult_meta(item['ult_meta_file'])

        with closing(open(item['ult_file'], 'rb')) as ult_file:
            ult_data = ult_file.read()
            ultra = np.fromstring(ult_data, dtype=np.uint8)
            ultra = ultra.astype("float32")

            ult_no_frames = int(len(ultra) / (ult_num_vectors * ult_pixels_per_vector))

            # reshape into vectors containing a frame each
            ultra = ultra.reshape((ult_no_frames, ult_num_vectors, ult_pixels_per_vector))
            ultra_interp = []
            probe_data = []

            # TODO: Determine whether we're working with linear or radial ultrasound (This is a hack and probably could be done better)
            # Radial ultrasound has pixPerVector != zeroOffset; linear ultrasound probe has pixPerVector == zeroOffset
            if ult_pixels_per_vector == ult_zero_offset:
                ultra_interp, probe_data = linear_probe_interpolation(ultra, ult_pixels_per_vector, ult_pixels_per_mm, ult_num_vectors, ult_no_frames)
            elif ult_pixels_per_mm != ult_zero_offset:
                ultra_interp, probe_data = radial_probe_interpolation(ultra, ult_pixels_per_vector, ult_pixels_per_mm, ult_num_vectors, ult_angle, ult_zero_offset, ult_no_frames)

            # Simple progress bar borrowed from here: https://stackoverflow.com/questions/3160699/python-progress-bar
            sys.stdout.write("[%s]" % (" " * toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

            # DO REGISTRATION (CHECK FOR PARALLELISM)
            # TODO: Get parallelism working with the PIRT routine (causes paging file error on my machine)
            useParallelFlag = False
            if useParallelFlag:
                # setup parallelism for running the registration
                mgr = Manager()

                storage = mgr.dict()  # create the storage for the optical flow
                #ns = mgr.Namespace()
                #ns.ultra_interp = ultra_interp

                procs = []

                # run the parallel processes
                for fIdx in range(0, ult_no_frames-1):
                    #proc = Process(target=parallel_register, args=(ns, fIdx, ult_no_frames, storage))
                    proc = Process(target=parallel_register, args=(ultra_interp[fIdx], ultra_interp[fIdx + 1], fIdx, ult_no_frames, storage))
                    procs.append(proc)
                    proc.start()

                # finalize the parallel processes
                for proc in procs:
                    proc.join()

                # retrieve the output
                ofdisp = storage.values()
            else:
                # do registration without parallel computation support
                ofdisp = []
                prog_increment = int(ult_no_frames / 100)

                for fIdx in range(ult_no_frames - 1):
                    #sys.stdout.write("Working on frame pair %d of %d\n" % (fIdx, ult_no_frames - 1))
                    current_im = ultra_interp[fIdx]
                    next_im = ultra_interp[fIdx + 1]

                    # execute and store the optimization
                    # ofdisp.append({'of': pirt_reg(current_im, next_im), 'current frame': fIdx, 'next frame': fIdx + 1})
                    flow = opencv_opticalflow(current_im, next_im, None)
                    ofdisp.append({'of': (flow[..., 0], flow[..., 1]), 'current frame': fIdx, 'next frame': fIdx + 1})

                    # update the bar
                    if fIdx % prog_increment is 0:
                        sys.stdout.write("-")
                        sys.stdout.flush()

            sys.stdout.write("]\n")  # this ends the progress bar

            print("Finished computing optical flow for %s" % (item['filebase']) + " (File " + str(file_count) + " of " + str(len(data_list)) + ")")
            file_count = file_count + 1

            # compute the ultrasound time vector
            ultra_time = np.linspace(0, ult_no_frames, ult_no_frames, endpoint=False) / ult_fps
            ultra_time = ultra_time + ult_time_in_sec_of_first_frame + .5 / ult_fps

            # Compute the audio time vector
            ult_wav_time = np.linspace(0, len(ult_wav_frames), len(ult_wav_frames), endpoint=False) / ult_wav_fs

            # Store the data
            data = {}
            data['ofdisp'] = ofdisp
            data['ultra_interp'] = ultra_interp
            data['ult_time'] = ultra_time
            data['wav_time'] = ult_wav_time
            data['wav_data'] = ult_wav_frames
            data['wav_fs'] = ult_wav_fs
            data['ult_no_frames'] = ult_no_frames
            data['probe_data'] = probe_data
            #data['probe_array_length_mm'] = probe_array_length_mm
            #data['probe_depth_mm'] = probe_depth_mm

            # TODO the registration takes a long time to compute and each should be saved but this is probably not the best way; also, assumes 'results' folder exists
            save_file = results_directory + os.sep + os.path.basename(item['filebase']) + "_OF.pickle" #item['filebase'].replace('data', 'results') + "_OF.pickle"
            pickle.dump(data, open(save_file, "wb"))
            print("Saving results to %s" % save_file)


