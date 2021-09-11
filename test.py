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

myDice.set([1, 1, 4, 5, 6])

# if len(collections.Counter(myDice.dice())) == 4:
#     singletons = list(myDice.as_set())
#     mask = '0'
#     for i in range(len(singletons)-1):
#         print(singletons[i])
#         if singletons[i+1] - singletons[i] > 2:
#             mask = '1'
#     print("mask = ", mask)
#
#
# action_table = action_q_table()
# # keep actions plus masks:
# action_to_dice_to_keep, keeping_actions_masks = action_table.print_all_action_q_table()

print(3/2)

v = np.array([1, 0, 1, 0, 0])
u = np.array([0, 1, 1, 1, 0])
w = np.add(u,v)
print(f"{w}")
