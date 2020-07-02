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


import pickle
import os
import enum
import math

# numpy and scipy
import numpy as np

# scientific plotting
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.stats import trim_mean


class FlowDir(enum.Enum):
    Horizontal = 0
    Vertical = 1


class GUI:
    def __init__(self, data):
        """ class initializer """
        # store relevant dictionary entries into convenient local variables
        self.ultra_interp = data['ultra_interp']
        self.ofdisp = data['ofdisp']
        self.ult_no_frames = data['ult_no_frames']
        self.ult_time = data['ult_time']
        self.ult_period = self.ult_time[1] - self.ult_time[0]
        self.wav_time = data['wav_time']
        self.wav_data = data['wav_data'] / max(abs(data['wav_data']))
        self.wav_fs = data['wav_fs']
        self.wav_num_samples = len(self.wav_data)

        # Visualization options
        self.flow_dir = FlowDir.Horizontal
        self.flow_polarity = 1
        self.flow_dir_label = 'Horizontal in Video'
        self.flow_pol_label = ''
        self.skip_increment = 7
        self.pos_ylim_del = 1
        self.vel_ylim_del = 10
        self.pos_ylim_min = 1
        self.pos_ylim_max = 200
        self.vel_ylim_min = 10
        self.vel_ylim_max = 2000

        # TODO implement vector scaling as a user parameter
        self.scaling = 1.0

        # visualize registration as quiver plot
        self.xx, self.yy = np.meshgrid(range(1, self.ultra_interp[0].shape[0]),
                                       range(1, self.ultra_interp[0].shape[1]))
        self.x_indices, self.y_indices = np.meshgrid(np.arange(0, self.xx.shape[0], self.skip_increment),
                                                   np.arange(0, self.xx.shape[1], self.skip_increment))

        #TODO set the default figure size to be some sensible proportion of the screen real estate
        self.fig = plt.figure(figsize=(10, 8))
        self.ax_quiver = self.fig.add_axes([0.1, 0.6, 0.4, 0.4])
        plt.sca(self.ax_quiver)
        self.im = self.ax_quiver.imshow(self.ultra_interp[0])
        self.quiver = plt.quiver(self.yy[self.x_indices, self.y_indices],
                                 self.xx[self.x_indices, self.y_indices],
                                 self.ofdisp[0]['of'][1][self.y_indices, self.x_indices],
                                 self.ofdisp[0]['of'][0][self.y_indices, self.x_indices],
                                 scale_units='xy', scale=0.5, color='r', angles='xy')


        # compute the velocity and position using the trimmed mean approach
        self.vel = np.empty((self.ult_no_frames - 1, 2))
        self.pos = np.empty((self.ult_no_frames - 2, 2))
        self.compute_kinematics()

        # Add a compass plot to visualize the consensus vector
        angles, radii = cart2pol(self.vel[:, 0], self.vel[:, 1])
        angle, radius = cart2pol(self.vel[0, 0], self.vel[0, 1])
        self.ax_compass = self.fig.add_axes([0.7, 0.7, 0.2, 0.2], projection='polar')
        self.ax_compass.set_ylim(0, max(radii)*0.5)

        # create velocity plot
        self.ax_vel = self.fig.add_axes([0.1, 0.5, 0.8, 0.1])
        self.line_vel, = plt.plot(self.ult_time[0:self.ult_no_frames - 1], self.vel[:, self.flow_dir.value])
        plt.axhline(linewidth=1, color='k')
        # TODO: Setting the xlim changes the tick scaling but not the data scale
        self.ax_vel.set_xlim([0.0, self.ult_time[-1]])
        self.ax_vel.set_ylim([-1e2, 1e2])
        self.ax_vel.set_title("Velocity (" + self.flow_dir_label + ")")
        self.ax_vel.set_ylabel("velocity (mm/s)")
        self.ax_vel.tick_params(bottom=False)

        # create position plot
        self.ax_pos = self.fig.add_axes([0.1, 0.3, 0.8, 0.1])
        self.line_pos, = plt.plot(self.ult_time[1:self.ult_no_frames - 1], self.pos[:, self.flow_dir.value])
        plt.axhline(linewidth=1, color='k')
        # TODO: Setting the xlim changes the tick scaling but not the data scale
        self.ax_pos.set_xlim([0.0, self.ult_time[-1]])
        self.ax_pos.set_ylim([-2e1, 2e1])
        self.ax_pos.set_title("Position (" + self.flow_dir_label + ")")
        self.ax_pos.set_ylabel("relative position (mm)")
        self.ax_pos.set_xlabel("time (s)")
        # cache the axis for faster rendering

        # create audio plot
        self.ax_audio = self.fig.add_axes([0.1, 0.1, 0.8, 0.1])
        self.line_audio, = plt.plot(self.wav_time, self.wav_data)

        self.ax_audio.set_xlim([0.0, self.ult_time[-1]])
        self.ax_audio.set_ylim([-1, 1])
        plt.axhline(linewidth=1, color='k')
        self.ax_audio.set_title("Audio")

        self.frame_index = 0

        # connect the callbacks
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.mouse_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)

        # cache the axis for faster rendering
        self.fig.canvas.draw()
        self.ax_quiver_bg = self.fig.canvas.copy_from_bbox(self.ax_quiver.bbox)
        self.ax_vel_bg = self.fig.canvas.copy_from_bbox(self.ax_vel.bbox)
        self.ax_vel_bg = self.fig.canvas.copy_from_bbox(self.ax_vel.bbox)
        self.ax_pos_bg = self.fig.canvas.copy_from_bbox(self.ax_pos.bbox)
        self.ax_audio_bg = self.fig.canvas.copy_from_bbox(self.ax_audio.bbox)
        self.ax_compass_bg = self.fig.canvas.copy_from_bbox(self.ax_compass.bbox)

        # now that we have cached the background add the varying plot features (image, quiver, time markers)
        plt.sca(self.ax_quiver)
        self.quiver_frame_label = plt.text(1, 10, "1", fontsize=48, color="white")
        plt.sca(self.ax_vel)
        self.point_vel, = plt.plot(self.ult_time[0], self.vel[0, self.flow_dir.value], marker="o", ls="", color="r")
        plt.sca(self.ax_pos)
        self.point_pos, = plt.plot(self.ult_time[0], self.pos[0, self.flow_dir.value], marker="o", ls="", color="r")
        plt.sca(self.ax_audio)
        self.marker_line_audio, = plt.plot([self.wav_time[0], self.wav_time[0]], [-1, 1], color="r")
        plt.sca(self.ax_compass)
        self.marker_line_compass, = plt.plot([0, angle], [0, radius], color="r") #("", xy=(angle, radius), xytext=(0, 0), arrowprops=dict(arrowstyle="<-", color='k'))

        plt.show()

    def key_press(self, event):
        # print('press', event.key)

        if event.key == 'n':
            self.flow_polarity = -self.flow_polarity
            self.flow_pol_label = ', Negated' if self.flow_polarity is -1 else ''
            self.line_vel.set_ydata(self.flow_polarity*self.vel[:, self.flow_dir.value])
            self.line_pos.set_ydata(self.flow_polarity*self.pos[:, self.flow_dir.value])
            self.ax_vel.set_title("Velocity (" + self.flow_dir_label + " in Video" + self.flow_pol_label + ")")
            self.ax_pos.set_title("Position (" + self.flow_dir_label + " in Video" +self.flow_pol_label + ")")
        elif event.key == 'd':
            self.flow_dir = FlowDir.Vertical if self.flow_dir is FlowDir.Horizontal else FlowDir.Horizontal
            self.flow_dir_label = 'Vertical' if self.flow_dir is FlowDir.Vertical else 'Horizontal'

            self.line_vel.set_ydata(self.flow_polarity*self.vel[:, self.flow_dir.value])
            self.line_pos.set_ydata(self.flow_polarity*self.pos[:, self.flow_dir.value])
            self.ax_vel.set_title("Velocity (" + self.flow_dir_label + " in Video" + self.flow_pol_label + ")")
            self.ax_pos.set_title("Position (" + self.flow_dir_label + " in Video" + self.flow_pol_label + ")")
        elif (event.key == 'up') | (event.key == 'down'):
            shift_dir = -1 if event.key == 'up' else 1
            ylim_pos = self.ax_pos.get_ylim()
            ylim_vel = self.ax_vel.get_ylim()
            ylim_pos_val = max(min(ylim_pos[1] + shift_dir * self.pos_ylim_del, self.pos_ylim_max), self.pos_ylim_min)
            ylim_vel_val = max(min(ylim_vel[1] + shift_dir * self.vel_ylim_del, self.vel_ylim_max), self.vel_ylim_min)
            self.ax_pos.set_ylim(-ylim_pos_val, ylim_pos_val)
            self.ax_vel.set_ylim(-ylim_vel_val, ylim_vel_val)

        # re-cache the plot backgrounds
        self.fig.canvas.draw()
        self.ax_vel_bg = self.fig.canvas.copy_from_bbox(self.ax_vel.bbox)
        self.ax_pos_bg = self.fig.canvas.copy_from_bbox(self.ax_pos.bbox)


        # update the gui
        self.update_gui()

    def mouse_scroll(self, event):
        """ create mousewheel callback function for updating the plot to a new frame """
        # detect the mouse wheel action
        if event.button == 'up':
            self.frame_index = min(self.frame_index + 1, self.ult_no_frames - 3)
        elif event.button == 'down':
            self.frame_index = max(self.frame_index - 1, 0)
        else:
            print("oops")
            
        # update the gui
        self.update_gui()

    def update_gui(self):
        """ update the gui by changing the state according to the current frame """
        self.fig.canvas.restore_region(self.ax_quiver_bg)
        self.fig.canvas.restore_region(self.ax_pos_bg)
        self.fig.canvas.restore_region(self.ax_vel_bg)
        self.fig.canvas.restore_region(self.ax_audio_bg)
        self.fig.canvas.restore_region(self.ax_compass_bg)

        # update the plots
        self.im.set_data(self.ultra_interp[self.frame_index])
        self.quiver.set_UVC(self.ofdisp[self.frame_index]['of'][1][self.y_indices, self.x_indices],
                            self.ofdisp[self.frame_index]['of'][0][self.y_indices, self.x_indices])

        #self.ofdisp[self.frame_index]['of'].forward[self.y_indices, self.x_indices, 0],
        #self.ofdisp[self.frame_index]['of'].forward[self.y_indices, self.x_indices, 1])

        angle, radius = cart2pol(self.vel[self.frame_index, 0], self.vel[self.frame_index, 1])
        self.marker_line_compass.set_xdata([0, angle])
        self.marker_line_compass.set_ydata([0, radius])

        self.point_vel.set_xdata(self.ult_time[self.frame_index])
        self.point_vel.set_ydata(self.flow_polarity*self.vel[self.frame_index, self.flow_dir.value])

        self.point_pos.set_xdata(self.ult_time[self.frame_index])
        self.point_pos.set_ydata(self.flow_polarity*self.pos[self.frame_index, self.flow_dir.value])

        # Find closest audio sample to current frame
        self.marker_line_audio.set_xdata([self.ult_time[self.frame_index], self.ult_time[self.frame_index]])

        self.quiver_frame_label.set_text(str(self.frame_index))
        #self.fig.canvas.draw_idle()

        # Optimizations for Matplotlib
        # https://bastibe.de/2013-05-30-speeding-up-matplotlib.html
        # https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot
        self.ax_quiver.draw_artist(self.im)
        self.ax_quiver.draw_artist(self.quiver)
        self.ax_quiver.draw_artist(self.quiver_frame_label)
        self.ax_pos.draw_artist(self.point_pos)
        self.ax_vel.draw_artist(self.point_vel)
        self.ax_audio.draw_artist(self.marker_line_audio)
        self.ax_compass.draw_artist(self.marker_line_compass)

        # Use blitting to quickly refresh the canvas
        self.fig.canvas.blit(self.ax_quiver.bbox)
        self.fig.canvas.blit(self.ax_pos.bbox)
        self.fig.canvas.blit(self.ax_vel.bbox)
        self.fig.canvas.blit(self.ax_audio.bbox)
        self.fig.canvas.blit(self.ax_compass.bbox)

        #self.fig.canvas.flush_events()

    def compute_kinematics(self):
        """ compute the velocity and position from the displacement field """

        # TODO implement alternative means of obtaining the consensus vector
        # TODO this is working differently from the Matlab implementation (may need padding of the signals, e.g., following integration)
        # obtain the consensus velocity vector for each frame(pair)
        for fIdx in range(0, self.ult_no_frames - 1):
            disp_comp_h = self.ofdisp[fIdx]['of'][1][self.y_indices, self.x_indices] #self.ofdisp[fIdx]['of'].forward[:, :, 0]
            disp_comp_v = self.ofdisp[fIdx]['of'][0][self.y_indices, self.x_indices] #self.ofdisp[fIdx]['of'].forward[:, :, 1]

            self.vel[fIdx, 0] = trim_mean(disp_comp_h.flatten(), 0.25) / self.ult_period * self.scaling
            self.vel[fIdx, 1] = trim_mean(disp_comp_v.flatten(), 0.25) / self.ult_period * self.scaling

        # perform numerical integration
        self.pos[:, 0] = integrate.cumtrapz(self.vel[:, 0], self.ult_time[0:self.ult_no_frames - 1])
        self.pos[:, 1] = integrate.cumtrapz(self.vel[:, 1], self.ult_time[0:self.ult_no_frames - 1])


def cart2pol(x, y):
    # Based on: https://ocefpaf.github.io/python4oceanographers/blog/2015/02/09/compass/
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return theta, radius



def main():
    # TODO hard coded path for convenience while developing code
    filename = "..\\results\\P1_01_OF.pickle"

    # unpickle an OF file produced by ofreg.py
    data = pickle.load(open(filename, "rb"))

    # TODO export first field to compare it with the Matlab results
    # ofdisp = data['ofdisp']
    # np.savetxt("P1_01_test.txt", ofdisp[0]['of'].forward[:, :, 0], fmt="%.4f")

    # create the ofbrowse gui (scaffolded on Matplotlib)
    gd = GUI(data)

if __name__ == '__main__':
    main()
