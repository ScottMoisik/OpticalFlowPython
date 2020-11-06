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

import numpy as np
from scipy import integrate


def compute_kinematics(ofdisp, ult_time, ult_no_frames, x_indices, y_indices, ult_period, scaling):
    """ compute the velocity and position from the displacement field """
    vel = np.empty((ult_no_frames - 1, 2))
    pos = np.empty((ult_no_frames - 2, 2))  # integration of velocity leaves us with N - 1, we pad afterwards

    # obtain the consensus velocity vector for each frame(pair)
    for fIdx in range(0, ult_no_frames - 1):
        disp_comp_h = ofdisp[fIdx]['of'][0][y_indices, x_indices]  # ofdisp[fIdx]['of'].forward[:, :, 0]
        disp_comp_v = ofdisp[fIdx]['of'][1][y_indices, x_indices]  # ofdisp[fIdx]['of'].forward[:, :, 1]

        # vel[fIdx, 0] = trim_mean(disp_comp_h.flatten(), 0.25) / ult_period * scaling
        # vel[fIdx, 1] = trim_mean(disp_comp_v.flatten(), 0.25) / ult_period * scaling
        vel[fIdx, 0] = np.mean(disp_comp_h.flatten()) / ult_period * scaling
        vel[fIdx, 1] = np.mean(disp_comp_v.flatten()) / ult_period * scaling

    # perform numerical integration
    pos = np.empty((ult_no_frames - 2, 2))
    pos[:, 0] = integrate.cumtrapz(vel[:, 0], ult_time[0:ult_no_frames - 1])
    pos[:, 1] = integrate.cumtrapz(vel[:, 1], ult_time[0:ult_no_frames - 1])

    # pad the position with zeros to match the length of the velocity
    pos = np.concatenate((pos, np.array([[0.0, 0.0]])), axis=0)

    return pos, vel