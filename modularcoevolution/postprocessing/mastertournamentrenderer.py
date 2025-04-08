#  Copyright 2025 BONSAI Lab at Auburn University
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2025, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

import PIL.Image as Image
import PIL.ImageColor as ImageColor

import numpy
import scipy.stats

import math
import sys


# TODO: Rework this old code to newer standards
# Takes a file, returns a Pillow image.
def create_image(file, ranked=False):
    file_lines = file.readlines()
    images = dict()

    line_number = 0
    while line_number < len(file_lines):
        objective = "".join(file_lines[line_number].strip().title().split())
        tournament_size = len(file_lines[line_number+1].split())
        objective_lines = file_lines[line_number+1:line_number+tournament_size+1]

        raw_data = list()
        for line in objective_lines:
            line_entries = list()
            entries = line.split()
            for entry in entries:
                if entry == "None":
                    numeric_entry = float("NaN")
                else:
                    numeric_entry = float(entry)
                    # numeric_entry = math.copysign(math.log(math.fabs(float(entry) + 0.0001)), float(entry) + 0.0001)
                line_entries.append(numeric_entry)
            raw_data.append(line_entries)
        tournament_values = numpy.array(raw_data)

        if ranked:
            ranked_tournament_values = numpy.array(scipy.stats.rankdata(tournament_values, method="dense").reshape(tournament_values.shape), numpy.float_)
            ranked_tournament_values[numpy.isnan(tournament_values)] = numpy.nan
            tournament_values = ranked_tournament_values

        objective_min = numpy.nanmin(tournament_values)
        objective_max = numpy.nanmax(tournament_values)
        objective_range = objective_max - objective_min
        if objective_range == 0:
            objective_range = 1
        objective_factor = 256 / objective_range
        objective_pixels = numpy.trunc((tournament_values - objective_min) * objective_factor)

        objective_image = Image.new("RGBA", (len(objective_pixels), len(objective_pixels)), (0, 0, 0, 0))
        for y, row in enumerate(objective_pixels):
            for x, pixel in enumerate(row):
                if math.isnan(pixel):
                    objective_image.putpixel((x, y), (0, 0, 0, 0))
                else:
                    # r = g = b = int(pixel)
                    r, g, b = colorize(pixel)
                    objective_image.putpixel((x, y), (r, g, b, 255))
        images[objective] = objective_image
        line_number += tournament_size + 1
    return images

def colorize(grey_pixel):
    # hue = grey_pixel / 256 * 120
    hue = (grey_pixel + 20) / 256 * 80
    return ImageColor.getrgb("hsv({hue},100%,100%)".format(hue=hue))


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        log_path = sys.argv[1]
    else:
        log_path = ""

    if len(sys.argv) >= 3:
        runs = int(sys.argv[2])
    else:
        runs = 30

    for i in range(runs):
        subfolder = "Run {}".format(i)
        try:
            attacker_data = open(f"{log_path}/{subfolder}/tournamentDataAttacker.txt", "r")
            defender_data = open(f"{log_path}/{subfolder}/tournamentDataDefender.txt", "r")
        except FileNotFoundError:
            continue

        attacker_images = create_image(attacker_data, False)
        defender_images = create_image(defender_data, False)

        for objective, image in attacker_images.items():
            image.save(f"{log_path}/{subfolder}/tournamentAttacker{objective.capitalize()}.png")
        for objective, image in defender_images.items():
            image.save(f"{log_path}/{subfolder}/tournamentDefender{objective.capitalize()}.png")
