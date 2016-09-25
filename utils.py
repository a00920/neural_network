import numpy
def add_one_to_input(parametr_name):
    def decorator(func):
        def inner(*args, **kwargs):
            if parametr_name in kwargs:
                kwargs[parametr_name] = [(numpy.insert(input, 0, 1, axis=0), output) for
                            input, output in kwargs[parametr_name]]
            return func(*args, **kwargs)
        return inner
    return decorator