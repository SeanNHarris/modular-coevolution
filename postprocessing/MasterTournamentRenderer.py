import PIL.Image as Image
import PIL.ImageColor as ImageColor

import math


# TODO: Rework this old code to newer standards
# Takes a file, returns a Pillow image.
def create_image(file):
    file_lines = file.readlines()
    images = dict()

    line_number = 0
    while line_number < len(file_lines):
        objective = "".join(file_lines[line_number].strip().title().split())
        tournament_size = len(file_lines[line_number+1].split())
        objective_lines = file_lines[line_number+1:line_number+tournament_size+1]
        objective_min = None
        objective_max = None
        objective_pixels = list()
        for line in objective_lines:
            line_pixels = list()
            entries = line.split()
            for entry in entries:
                if entry == "None":
                    numeric_entry = float("NaN")
                    line_pixels.append(numeric_entry)
                    continue
                numeric_entry = float(entry)
                line_pixels.append(numeric_entry)
                if objective_min is None or numeric_entry < objective_min:
                    objective_min = numeric_entry
                if objective_max is None or numeric_entry > objective_max:
                    objective_max = numeric_entry
            objective_pixels.append(line_pixels)
        objective_factor = 256 / (objective_max - objective_min)
        for row in objective_pixels:
            for i, value in enumerate(row):
                if math.isnan(value):
                    row[i] = value
                else:
                    row[i] = int((value - objective_min) * objective_factor)

        objective_image = Image.new("RGBA", (len(objective_pixels), len(objective_pixels)), (0, 0, 0, 0))
        for y, row in enumerate(objective_pixels):
            for x, pixel in enumerate(row):
                if math.isnan(pixel):
                    objective_image.putpixel((x, y), (0, 0, 0, 0))
                else:
                    r, g, b = colorize(pixel)
                    objective_image.putpixel((x, y), (r, g, b, 255))
        images[objective] = objective_image
        line_number += tournament_size + 1
    return images

def colorize(grey_pixel):
    hue = grey_pixel / 256 * 120
    return ImageColor.getrgb("hsv({hue},100%,100%)".format(hue=hue))


if __name__ == "__main__":
    attacker_data = open("../Logs/tournamentDataAttacker.txt", "r")
    defender_data = open("../Logs/tournamentDataDefender.txt", "r")

    attacker_images = create_image(attacker_data)
    defender_images = create_image(defender_data)

    for objective, image in attacker_images.items():
        image.save("../Logs/tournamentAttacker{0}.png".format(objective.capitalize()))
    for objective, image in defender_images.items():
        image.save("../Logs/tournamentDefender{0}.png".format(objective.capitalize()))
