import numpy
from get_data import get_characters_bitmap
from methods.hebb import hebb, get_error_number
from methods.perceptron import Perceptron


PATH_TO_DATA = '../data/characters'


def transform_matrix(matrix):
    return (matrix * 2 - 1).T


def get_vector_from_letter(letters, letter):
    index = letters.index(letter)
    vector = numpy.matrix(numpy.full((len(letters), 1), -1))
    vector[index, 0] = 1
    return vector


def get_training_data(letters, example_numbers=1, transform_matrix=lambda x:x.T):
    characters_bitmap = get_characters_bitmap(PATH_TO_DATA)
    return [(transform_matrix(vector), get_vector_from_letter(letters, letter))
            for letter in letters for vector in characters_bitmap.get(letter, [])[:example_numbers]]


def get_test_data(letters, transform_matrix=lambda x:x.T):
    characters_bitmap = get_characters_bitmap(PATH_TO_DATA)
    return [(transform_matrix(vector), get_vector_from_letter(letters, letter))
            for letter in letters for vector in characters_bitmap.get(letter, [])]


def test_hebb(letters):
    training_data = get_training_data(letters, transform_matrix=transform_matrix)
    weights = hebb(examples=training_data)
    test_data = get_test_data(letters, transform_matrix=transform_matrix)
    print(weights)
    print(get_error_number(examples=test_data, weights=weights))


def test_perceptron(first_latters, second_latters, example_numbers=1):

    characters_bitmap = get_characters_bitmap(PATH_TO_DATA)
    training_data = []
    training_data += [(vector.T, 0)
            for letter in first_latters for vector in characters_bitmap.get(letter, [])[:example_numbers]]
    training_data += [(vector.T, 1)
            for letter in second_latters for vector in characters_bitmap.get(letter, [])[:example_numbers]]
    p1 = Perceptron(25, 100, 5)
    p1.alpha_learn(examples=training_data)
    p2 = Perceptron(25, 100, 5)
    p2.gamma_learn(examples=training_data)

    test_data = []
    test_data += [(vector.T, 0)
            for letter in first_latters for vector in characters_bitmap.get(letter, [])]
    test_data += [(vector.T, 1)
            for letter in second_latters for vector in characters_bitmap.get(letter, [])]

    error1 = 0
    for input, output in test_data:
        if p1.get_result(input) != output:
            error1 += 1
    print(error1)

    error2 = 0
    for input, output in test_data:
        if p2.get_result(input) != output:
            error2 += 1
    print(error1)

test_perceptron(['d', 'b'], ['v', 'w'])