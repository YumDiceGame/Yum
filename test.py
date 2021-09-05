from dice import *
from score import *
from do_keep_action_q_table import *
import numpy as np
import numpy.ma as ma
myDice = DiceSet()

myScore = Score()

myDice.roll_Heavy()
print(myDice)
print(myDice.sum())