from dice import *
from score import *
from do_keep_action_q_table import *
import numpy as np
import numpy.ma as ma
myDice = DiceSet()

myScore = Score()


myDice.set([1, 1, 2, 3, 3])
# myDice.set([1, 4, 4, 4, 4])
# myDice.set([2, 2, 4, 4, 6])
print(myDice)
print(myDice.get_dict())


if myDice.is_almost_straight():
    print("almost straight")
if myDice.is_two_pairs():
    print("almost full")
    singleton = myDice.find_face_not_two_pair()
    print("singleton = ", singleton)
    myDice.make_list_reroll_for_selected_die_face(singleton)
    print(myDice.get_list_reroll())
    myDice.flip_list_reroll()
    print(myDice.get_list_reroll())
if myDice.is_almost_yum():
    print("almost yum")

