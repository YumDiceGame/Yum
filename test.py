from dice import *
from score import *
from do_keep_action_q_table import *
import numpy as np
import numpy.ma as ma
myDice = DiceSet()

myScore = Score()


myDice.set([1, 1, 1, 1, 1])
# myDice.set([1, 4, 4, 4, 4])
# myDice.set([2, 2, 4, 4, 6])
print(myDice)


if myDice.is_almost_straight():
    print("almost straight")
if myDice.is_two_pairs():
    print("almost full")
if myDice.is_almost_yum():
    print("almost yum")

