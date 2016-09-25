import os
import numpy
import re
from collections import defaultdict
from PIL import Image


def character_image_to_vector(image_path):
    image = Image.open(image_path)
    return numpy.matrix([c for c in image.tobytes()])


def get_character_from_name(image_name):
    ALPHABET = list(map(chr, range(97, 123)))
    PATTERN = r'PA5-([0-9]*)\.png'
    index = int(re.match(PATTERN, image_name).group(1))
    return ALPHABET[(index-1)//2]


def get_characters_bitmap(path_to_data):
    from_char_to_images = defaultdict(list)
    for image_name in os.listdir(path_to_data):
        image_path = os.path.join(path_to_data, image_name)
        (from_char_to_images[get_character_from_name(image_name)].
         append(character_image_to_vector(image_path)))
    return from_char_to_images
