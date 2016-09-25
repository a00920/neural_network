import numpy
from utils import add_one_to_input


def is_correct_weight(input_matrix, output_matrix, weights):
    result = (weights.dot(input_matrix) > 0) * 2 - 1
    return numpy.array_equal(result, output_matrix)


def get_max_number_operations(number):
    return number * number + 5


@add_one_to_input('examples')
def hebb(*, examples):
    input, output = examples[0]
    input_matrix = numpy.concatenate([input for input, _ in examples], axis=1)
    output_matrix = numpy.concatenate([output for _, output in examples], axis=1)
    weights = numpy.matrix(numpy.zeros(shape=(output.shape[0], input.shape[0])))

    for _ in range(get_max_number_operations(len(examples))):
        for input, output in examples:
            if is_correct_weight(input_matrix, output_matrix, weights):
                return weights
            weights += output * input.T


@add_one_to_input('examples')
def get_error_number(*, examples, weights):
    return sum([is_correct_weight(input, output, weights) for input, output in examples])