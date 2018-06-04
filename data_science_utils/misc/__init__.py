import inspect


def print_function_code(func):
    print("".join(inspect.getsourcelines(func)[0]))

