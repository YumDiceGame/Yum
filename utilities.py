import random
import numpy as np
def one_in_x_chances(x):
    """
    Will return True one in x times
    """
    rand_number = np.random.randint(0,x)
    if rand_number == x-1:
        print("yes")
    else:
        print("nop")
