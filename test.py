from dice import *
from score import *
from do_keep_action_q_table import *
from do_q_table import do_q_table_rows
import numpy as np
import numpy.ma as ma
import re
from utilities import *
import sys, select
dice = DiceSet()
# for _ in range(30):
#     one_in_x_chances(30)
# for _ in range(100):
#     dice.roll_special(10)
#     print(dice)

dice.set([1,2,3,4,5])
print(dice)

action_q_table_ = action_q_table()
full_dice_list = action_q_table_.do_list_of_dice_rolls()
print("hi")
# print("Hit <Enter> to abort shutdown --> ")
#
# i, o, e = select.select([sys.stdin], [], [], 600)
# if (i):
#     print("Shutdown aborted")
# else:
#     print("Shutting down ...")
