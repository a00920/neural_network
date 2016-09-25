import numpy
from get_data import get_characters_bitmap
from methods.hebb import hebb, get_error_number


PATH_TO_DATA = '../data/characters'


def transform_matrix(matrix):
    return (matrix * 2 - 1).T


def get_vector_from_letter(letters, letter):
    index = letters.index(letter)
    vector = numpy.matrix(numpy.full((len(letters), 1), -1))
    vector[index, 0] = 1
    return vector


def get_training_data(letters, example_numbers=1):
    characters_bitmap = get_characters_bitmap(PATH_TO_DATA)
    return [(transform_matrix(vector), get_vector_from_letter(letters, letter))
            for letter in letters for vector in characters_bitmap.get(letter, [])[:example_numbers]]


def get_test_data(letters):
    characters_bitmap = get_characters_bitmap(PATH_TO_DATA)
    return [(transform_matrix(vector), get_vector_from_letter(letters, letter))
            for letter in letters for vector in characters_bitmap.get(letter, [])]


def test_hebb(letters):
    training_data = get_training_data(letters)
    weights = hebb(examples=training_data)
    test_data = get_test_data(letters)
    print(weights)
    print(get_error_number(examples=test_data, weights=weights))


test_hebb(['a', 'b', 'c', 'd'])