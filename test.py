from dice import *
from score import *
from do_keep_action_q_table import *
import numpy as np
import numpy.ma as ma
myDice = DiceSet()
myDice.set([2, 2, 2, 4, 5])
print(myDice)
myScore = Score()

print(myDice.find_one_pair_face())

myDice.make_list_reroll_for_selected_die_faces({4,5})

print(myDice.get_list_reroll())
