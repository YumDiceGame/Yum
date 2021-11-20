from dice import *
from score import *
from do_keep_action_q_table import *
from do_q_table import do_q_table_rows
import numpy as np
import numpy.ma as ma
myDice = DiceSet()

myScore = Score()
myScore.reset_scores()
# CompScore = Score()
# CompScore.reset_scores()
#
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

# print(myDice.is_yum())
# print(myDice.as_dict())

# action_table = action_q_table()
# action_to_dice_to_keep, keeping_actions_masks = action_table.print_all_action_q_table()
#
# print("len 0 = ", len(action_to_dice_to_keep[0]) > 1)
# print("len 1 = ", len(action_to_dice_to_keep[1]) > 1)
# print("len 8 = ", len(action_to_dice_to_keep[8]) > 1)


myDice.roll_Full()
print(myDice)
dice_dict = myDice.get_dict()

print(myScore.is_components_full_available(myDice))

# print(list(dice_dict.keys())[list(dice_dict.values()).index(3)])
# print(list(dice_dict.keys())[list(dice_dict.values()).index(2)])

# int(max(self._dict, key=self._dict.get))
# myDice.roll()
# myDice.set_list_reroll([False, True, False, True, False])
# print(myDice.get_list_reroll())
# print(myDice.is_keep_all())
# print([False] * NUM_DICE)