# Copyright (c) 2020 Scott Moisik.
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
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License. You should have received a
# copy of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License along with the data. If not,
# see <https://creativecommons.org/licenses/by-nc-sa/4.0/> for details.
#

import argparse
import logging
import sys
import os
import time
import datetime

import pickle
import csv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# local modules
from of import ofreg as of


def widen_help_formatter(formatter, total_width=140, syntax_width=35):
    """Return a wider HelpFormatter, if possible."""
    try:
        # https://stackoverflow.com/a/5464440
        # beware: "Only the name of this class is considered a public API."
        kwargs = {'width': total_width, 'max_help_position': syntax_width}
        formatter(None, **kwargs)
        return lambda prog: formatter(prog, **kwargs)
    except TypeError:
        warnings.warn("argparse help formatter failed. Falling back.")
        return formatter


def main():
    parser = argparse.ArgumentParser(description="OF test script",
                                     formatter_class=widen_help_formatter(argparse.HelpFormatter, total_width=100, syntax_width=35))

    # mutually exclusive with reading previous results from a file
    parser.add_argument("directory",
                        help="Directory containing the data to be read.")

    parser.add_argument("-e", "--exclusion_list", dest="exclusion_filename",
                        help="Exclusion list of data files that should be ignored.",
                        metavar="file")

    parser.add_argument("-l", "--log_file", dest="log_filename",
                        help="Write log to file.", metavar="file")

    helptext = "NOT IMPLEMENTED. Save results to file. Supported types are .pickle, .json and .csv."
    parser.add_argument("-o", "--output", dest="filename",
                        help=helptext, metavar="file")

    parser.add_argument("-v", "--verbose",
                        action="store_true", dest="verbose",
                        default=True,
                        help="NOT IMPLEMENTED. Print status messages to stdout.")

    args = parser.parse_args()

    directory = args.directory

    exclusion_list_name = None
    if args.exclusion_filename:
        exclusion_list_name = args.exclusion_filename

    if args.log_filename:
        log_filename = args.log_filename
    else:
        log_filename = directory.strip("/") + '.log'

    logging.basicConfig(filename=log_filename,
                        filemode='w',
                        level=logging.INFO)
    logging.info('Run started at ' + str(datetime.datetime.now()))

    # this is the actual list of items that gets processed including meta data contained outwith the ult file
    root = tk.Tk()
    root.withdraw()

    # Ask the user which folder to open
    while True:
        data_directory = filedialog.askdirectory(title="Select data directory...")

        if data_directory:
            data_list = of.get_data_from_dir(data_directory)

            answer = messagebox.askyesnocancel("Proceed with analysis",
                                   "Found " + str(len(data_list)) + " ultrasound data packages. Proceed with analysis and use default results folder (Yes), choose results folder (No), or cancel?",
                                   icon='warning')
            if answer:
                results_directory = os.path.join(data_directory, "results")
                while True:
                    if os.path.exists(results_directory):
                        if messagebox.askokcancel("Warning", "Use existing results folder?", icon="warning"):
                            break
                        else:
                            exit(0)
                    else:
                        #Make the folder
                        os.makedirs(results_directory)
                        break

                # run OF on each item
                ofdata_file_list = of.compute(data_list, results_directory)
                exit(0)
            elif answer is None:
                exit(0)
            else:
                results_directory = filedialog.askdirectory(title="Select results directory...")
                if results_directory:
                    # run OF on each item
                    data = of.compute(data_list, results_directory)
                    exit(0)
                else :
                    exit(0)
        else:
            answer = messagebox.askretrycancel("File select", "No folder was selected. Try again?")
            if not answer:
                break





if __name__ == '__main__':
    t = time.time()
    main()
    elapsed_time = time.time() - t
    logging.info('Elapsed time ' + str(elapsed_time))
