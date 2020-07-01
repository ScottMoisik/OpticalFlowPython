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

from multiprocessing import Process, Manager

# diffeomorphic demons algorithm implemented in python in the DIPY package
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
from dipy.viz import regtools

# numpy and scipy
import numpy as np
import scipy.io as sio
import scipy.io.wavfile as sio_wavfile
from scipy.signal import butter, filtfilt, kaiser, sosfilt

from scipy import interpolate

# scientific plotting
import matplotlib.pyplot as plt
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


def parallel_register(ns, index, num_frames, storage):
    sys.stdout.write("Working on frame pair %d of %d\n" % (index, num_frames - 1))
    current_im = ns.ultra_interp[index]
    next_im = ns.ultra_interp[index + 1]

    # suppress warnings (from dipy package encouraging us to install "Fury")
    warnings.filterwarnings("ignore")

    # execute the optimization
    storage[index] = {'of': ns.sdr.optimize(next_im, current_im), 'current frame': index, 'next frame': index + 1}

    # revert back to always displaying warnings
    warnings.filterwarnings("always")


def compute(data_list):
    # inputs: elements in data dictionary generated by get_data_from_dir
    # i.e. all_data[i] is an item

    for item in data_list:
        ofreg_logger.info("PD: " + item['filebase'] + " " + item['prompt'] + '. item processed.')

        (ult_wav_frames, ult_wav_fs) = read_wav(item['ult_wav_file'])
        (ult_NumVectors, ult_PixPerVector, ult_PixelsPerMm, ult_fps, ult_TimeInSecOfFirstFrame) = read_ult_meta(item['ult_meta_file'])

        with closing(open(item['ult_file'], 'rb')) as ult_file:
            ult_data = ult_file.read()
            ultra = np.fromstring(ult_data, dtype=np.uint8)
            ultra = ultra.astype("float32")

            ult_no_frames = int(len(ultra) / (ult_NumVectors * ult_PixPerVector))

            # reshape into vectors containing a frame each
            ultra = ultra.reshape((ult_no_frames, ult_NumVectors, ult_PixPerVector))

            # interpolate the data to correct the axis scaling for purposes of image registration
            probe_array_length_mm = 40.0  #TODO 40 mm long linear probe assumed!!!
            probe_depth_mm = ult_PixPerVector/ult_PixelsPerMm
            length_depth_ratio = probe_depth_mm/probe_array_length_mm

            x = np.linspace(1, ult_NumVectors, ult_NumVectors)
            y = np.linspace(1, ult_PixPerVector, ult_PixPerVector)

            xnew = np.linspace(1, ult_NumVectors, ult_NumVectors)
            ynew = np.linspace(1, ult_PixPerVector, math.ceil(ult_NumVectors * length_depth_ratio))
            f = interpolate.interp2d(x, y, np.transpose(ultra[1, :, :]), kind='linear')

            ultra_interp = []

            # debug plotting
            debug_plot_interpolation = False
            if debug_plot_interpolation:
                fig, ax = plt.subplots(1, 1)
                im = ax.imshow(f(xnew, ynew))

            for fIdx in range(0, ult_no_frames):
                f = interpolate.interp2d(x, y, np.transpose(ultra[fIdx, :, :]), kind='linear')
                ultra_interp.append(f(xnew, ynew))

                # debug plotting
                if debug_plot_interpolation:
                    im.set_data(ultra_interp[fIdx])
                    ax.set_title(str(fIdx))
                    fig.canvas.draw_idle()
                    plt.pause(0.01)

            # perform registration using diffeomorphic demons algorithm (from DIPY package)
            # https://dipy.org/documentation/1.1.1./examples_built/syn_registration_2d/#example-syn-registration-2d

            # specify the number of iterations in each of the levels in the multiresolution pyramid
            level_iters = [100, 50, 25]
            level_iters = [10, 5]
            metric = SSDMetric(2, smooth=2, inner_iter=10)

            # create a registration metric
            sigma_diff = 1.0 #3.0
            radius = 3
            metric = CCMetric(2, sigma_diff, radius)

            # create the registration object
            # TODO it may not be necessary to get the inverse displacement field... setting inv_iter to 0 in an attempt to speed up computation
            sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=0)

            skipDebugAreaFlag = True
            if skipDebugAreaFlag:
                #TODO Debugging area
                fIdx = 134
                current_im = ultra_interp[fIdx]
                next_im = ultra_interp[fIdx + 1]
                current_imHR = ultra[fIdx, :, :]
                next_imHR = ultra[fIdx + 1, :, :]

                # execute the optimization
                level_iters = [10, 5]
                #metric = SSDMetric(2, smooth=2, inner_iter=10)#CCMetric(2, 1.5, 1)

                sdr = SymmetricDiffeomorphicRegistration(metric, level_iters, inv_iter=0)

                mapping = sdr.optimize(next_im, current_im)
                regtools.overlay_images(next_im, current_im, 'Next', 'Overlay', 'Current', 'testOverlay.png')
                regtools.plot_2d_diffeomorphic_map(mapping, 5, 'testMaps.png')

                #mappingHR = sdr.optimize(next_imHR, current_imHR)
                #regtools.overlay_images(next_imHR, current_imHR, 'Current', 'Overlay', 'Next', 'testOverlayHR.png')
                #regtools.plot_2d_diffeomorphic_map(mappingHR, 5, 'testMapsHR.png')

                skipIncrement = 5
                xx, yy = np.meshgrid(range(1, ultra_interp[fIdx].shape[0]),
                                               range(1, ultra_interp[fIdx].shape[1]))
                xIndices, yIndices = np.meshgrid(np.arange(0, xx.shape[0], skipIncrement),
                                                           np.arange(0, xx.shape[1], skipIncrement))

                # TODO set the default figure size to be some sensible proportion of the screen real estate
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_axes([0.1, 0.6, 0.4, 0.4])
                im = ax.imshow(ultra_interp[0])
                quiver = plt.quiver(yy[xIndices, yIndices],
                                         xx[xIndices, yIndices],
                                         mapping.forward[yIndices, xIndices, 0],
                                         mapping.forward[yIndices, xIndices, 1],
                                         scale_units='xy', scale=0.2, color='r', angles='xy')
                plt.show(block=False)



            # iterate through the frame pairs and perform the registration each
            #ult_no_frames = 25 # just for debugging purposes
            debug_plot_ofreg = False

            # DO REGISTRATION (CHECK FOR PARALLELISM)
            ofoutput = []
            useParallelFlag = False
            if useParallelFlag:
                # setup parallelism for running the registration
                mgr = Manager()
                storage = mgr.dict()  # create the storage for the optical flow
                ns = mgr.Namespace()
                ns.ultra_interp = ultra_interp
                ns.sdr = sdr

                procs = []

                # run the parallel processes
                for fIdx in range(0, ult_no_frames-1):
                    proc = Process(target=parallel_register, args=(ns, fIdx, ult_no_frames, storage))
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
                for fIdx in range(ult_no_frames - 1):
                    sys.stdout.write("Working on frame pair %d of %d\n" % (fIdx, ult_no_frames - 1))
                    current_im = ultra_interp[fIdx]
                    next_im = ultra_interp[fIdx + 1]

                    # execute the optimization
                    ofdisp.append({'of': sdr.optimize(next_im, current_im), 'current frame': fIdx, 'next frame': fIdx + 1})

                    # debug plotting
                    if debug_plot_ofreg:
                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(current_im)
                        ax[1].imshow(next_im)
                        plt.show()
                        plt.pause(0.05)

            print("Finished computing optical flow for %s" % (item['filebase']))


            # debug plotting
            debug_plot_quiver = True
            if debug_plot_quiver:
                # visualize registration as quiver plot
                xx, yy = np.meshgrid(range(1, ultra_interp[0].shape[0]), range(1, ultra_interp[0].shape[1]))

                fig, ax = plt.subplots(1, 1)
                im = ax.imshow(ultra_interp[0])
                quiver = plt.quiver(yy, xx, ofdisp[1]['of'].forward[:, :, 0], ofdisp[1]['of'].forward[:, :, 1], color = 'r')
                debug_frame = 1

                # create mousewheel callback function for updating the plot to a new frame
                def update_plot(event):
                    nonlocal debug_frame
                    # detect the mouse wheel action
                    if event.button == 'up':
                        debug_frame = min(debug_frame + 1, ult_no_frames - 2)
                    elif event.button == 'down':
                        debug_frame = max(debug_frame - 1, 1)
                    else:
                        print("oops")

                    # update the plot
                    im.set_data(ultra_interp[debug_frame])
                    quiver.set_UVC(ofdisp[debug_frame]['of'].forward[:, :, 0], ofdisp[debug_frame]['of'].forward[:, :, 1])
                    ax.set_title(str(debug_frame))
                    fig.canvas.draw_idle()

                # register the callback function with the figure
                cid = fig.canvas.mpl_connect('scroll_event', update_plot)

            #plt.show()

            # compute the ultrasound time vector
            ultra_time = np.linspace(0, ult_no_frames, ult_no_frames, endpoint=False) / ult_fps
            ultra_time = ultra_time + ult_TimeInSecOfFirstFrame + .5 / ult_fps

            ult_wav_time = np.linspace(0, len(ult_wav_frames), len(ult_wav_frames), endpoint=False) / ult_wav_fs

            data = {}
            data['ofdisp'] = ofdisp
            data['ultra_interp'] = ultra_interp
            data['ult_time'] = ultra_time
            data['wav_time'] = ult_wav_time
            data['ult_no_frames'] = ult_no_frames
            data['probe_array_length_mm'] = probe_array_length_mm
            data['probe_depth_mm'] = probe_depth_mm

            # TODO the registration takes a long time to compute and each should be saved but this is probably not the best way; also, assumes 'results' folder exists
            save_file = item['filebase'].replace('data', 'results') + "_OF.pickle"
            pickle.dump(data, open(save_file, "wb"))
            print("Saving results to %s" % save_file)

