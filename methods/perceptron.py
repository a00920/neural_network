import numpy
ALPHA = 'alpha'
GAMMA = 'gamma'


class Perceptron(object):
    def __init__(self, s_count, a_count, teta=0, w_range=(0, 1)):
        self.s_a_weights = numpy.matrix(numpy.random.rand(a_count, s_count))
        self.a_r_weights = numpy.matrix(numpy.zeros((1, a_count)))
        self.teta = teta

    def learn(self, examples, eta=1, method=ALPHA):
        if method == ALPHA:
            self.alpha_learn(examples, eta)
        elif method == GAMMA:
            self.gamma_learn(examples, eta)

    def alpha_learn(self, examples, eta=1):
        inputs = [input for input, _ in examples]
        outputs = [output for _, output in examples]
        while outputs != self.get_results(inputs):
            for input, output in examples:
                if self.get_result(input) != output:
                    sign = output - self.get_result(input)
                    self.a_r_weights += sign * eta * (self.s_a_weights.dot(input) >= self.teta).T

    def gamma_learn(self, examples, eta=1):
        inputs = [input for input, _ in examples]
        outputs = [output for _, output in examples]
        self.s_a_weights = numpy.matrix(numpy.random.rand(self.s_a_weights.shape[0], self.s_a_weights.shape[1]))
        while outputs != self.get_results(inputs):
            for input, output in examples:
                if self.get_result(input) != output:
                    sign = output - self.get_result(input)
                    active = (self.s_a_weights.dot(input) >= self.teta).sum()
                    delta = (active * sign * eta / self.s_a_weights.shape[0]) * numpy.ones((1, self.s_a_weights.shape[0]))
                    self.a_r_weights += sign * eta * (self.s_a_weights.dot(input) >= self.teta).T - delta

    def get_result(self, input):
        return int((self.a_r_weights.dot((self.s_a_weights.dot(input) >= self.teta)) >= self.teta)[0,0])

    def get_results(self, inputs):
        results = []
        for input in inputs:
            results.append(self.get_result(input))
        return results
