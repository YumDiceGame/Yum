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

score = Score()
# 1
dice.set([3, 3, 3, 3, 4])
score.score_a_category('Threes',dice)
# 2
dice.set([4, 5, 5, 5, 6])
score.score_a_category('Fives',dice)
# 3
dice.set([2, 3, 4, 5, 6])
score.score_a_category('Straight',dice)
# 4
dice.set([4, 5, 6, 6, 6])
score.score_a_category('Sixes',dice)
# 5
dice.set([3, 4, 6, 6, 6])
score.score_a_category('High',dice)
# 6
dice.set([1, 2, 4, 4, 4])
score.score_a_category('Fours',dice)
# 7
dice.set([3, 5, 5, 6, 6])
print("pot max = ", score.get_potential_max_score(dice))


print(score.print_scorecard())
