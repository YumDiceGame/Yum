from dice import *
from score import *
from do_keep_action_q_table import *
from do_q_table import do_q_table_rows
import numpy as np
import numpy.ma as ma
myDice = DiceSet()

myScore = Score()
myScore.reset_scores()
CompScore = Score()
CompScore.reset_scores()

# Action q table:
action_table = action_q_table()

# Map action to dice to keep and action masks
list_set_keep_actions, keep_action_mask_dict = action_table.print_all_action_q_table()

# my_scorecard_string = myScore.print_scorecard()
# comp_scorecard_string = CompScore.print_scorecard()
# both_scorecards = zip(my_scorecard_string, comp_scorecard_string)
#
# agent = "Agent"
# print(f"Player{agent.rjust(30)}")
# for line in both_scorecards:
#     print(f"{line[0]}{line[1].rjust(30)}")
#
# print("\n\n")
# for line in my_scorecard_string:
#     print(line)
#
# myDice.roll_Yum()
# print(myDice.is_yum())

action = 17
myDice.set([3, 3, 3, 5, 5])
myDice.make_list_reroll_for_selected_die_faces(list_set_keep_actions[action])
myDice.get_list_reroll()

print(myDice.get_list_reroll())
