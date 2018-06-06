import inspect
import time

def print_function_code(func):
    print("".join(inspect.getsourcelines(func)[0]))


def get_timer():
    ctr=1
    pv = -1e8
    def timer(text=""):
        nonlocal ctr
        nonlocal pv
        tc = time.time()%10000
        diff = tc - pv
        diff = 0 if diff>1e6 else diff
        pv=tc
        print("%s: %.3f, %.3f, %s "%(ctr,tc,diff,text))
        ctr=ctr+1
    return timer