# Playing Solo Yum games
# with user input


from score import *
from constants import *
import pickle
import numpy as np
from numpy import ma
from do_q_table import do_q_table_rows
from do_keep_action_q_table import action_q_table
import matplotlib.pyplot as plt

PRINT = False

# Will want to see what the scoring q table would do
with open("q_table_scoring.pickle", "rb") as score_q_table_file:
    q_table_scoring = pickle.load(score_q_table_file)

# Also the keeping
with open("q_table_keeping_LR=0.3_DIS=0.8_5M_action_60_better.pickle", "rb") as keep_q_table_file:
    q_table_keeping = pickle.load(keep_q_table_file)

# This is to get the q table rows, needed for scoring q table
# Create q_table_scoring_rows
q_table_scoring_rows, list_of_die_face_counts, list_scoreable_categories = do_q_table_rows()
ref_action_table = action_q_table()
list_all_dice_rolls = ref_action_table.do_list_of_dice_rolls()

# Action q table:
action_table = action_q_table()

# Map action to dice to keep and action masks
list_set_keep_actions, keep_action_mask_dict = action_table.print_all_action_q_table()

myDice = DiceSet()

# two score cards: one for me, one for computer
player_score = Score()
agent_score = Score()
player_score.reset_scores()
agent_score.reset_scores()
all_scored = False


all_scored = False
turn_done = False
turn = 0

while not all_scored:

    turn += 1
    myDice.reset()
    myDice.roll()

    print("initial roll")
    print(myDice)

    for roll in range(1, NUM_ROLLS):

        # What would agent do
        q_table_keeping_index_dice = list_all_dice_rolls.index(myDice.dice())
        q_table_keeping_index_score = list_scoreable_categories.index(agent_score.get_available_cat_vector())
        q_table_keeping_index = q_table_keeping_index_dice * TWO_TO_NUM_SCORE_CATEGORIES + q_table_keeping_index_score
        action = (ma.masked_array(q_table_keeping[q_table_keeping_index][0:NUM_KEEPING_ACTIONS],
                                  keep_action_mask_dict[myDice.as_short_string()])).argmax()
        print(f"agent would keep {list_set_keep_actions[action]} row {q_table_keeping_index}")
        if action == 60:
            print("action 60")

        # ask for list reroll
        input_valid = False
        while not input_valid:
            list_reroll_string = input("Enter dice to keep (r for reroll, k for keep) --> ")
            if all(c in "kr" for c in list_reroll_string) and len(list_reroll_string) == 5:
                input_valid = True
            else:
                print("bad input")
        list_reroll = []
        for c in list_reroll_string:
            if c == 'r':
                list_reroll.append(True)
            else:
                list_reroll.append(False)
        myDice.set_list_reroll(list_reroll)
        myDice.roll_list_reroll()
        print(f"roll number {myDice.get_num_rolls()}")
        print(f"dice = {myDice}")

    # What would the score q table say:
    q_table_scoring_index_dice = list_all_dice_rolls.index(myDice.dice())
    q_table_scoring_index_score = list_scoreable_categories.index(agent_score.get_available_cat_vector())
    q_table_scoring_index = q_table_scoring_index_dice * TWO_TO_NUM_SCORE_CATEGORIES + q_table_scoring_index_score
    q_table_would_score_category = agent_score.score_with_q_table(q_table_scoring, q_table_scoring_index, myDice)
    print("q table would score -> ", score_int_to_cat(q_table_would_score_category))

    # ask user to score
    print("Available categories are: ")
    player_score.print_available_cats()
    input_valid = False
    while not input_valid:
        score_cat_input = input("score (1 2 3 4 5 6 Y S F L H)? --> ")
        if (len(score_cat_input) == 1) and (all(c in "123456YSFLHysflh" for c in score_cat_input)):
            input_valid = True
            if player_score.is_category_available(score_char_to_cat(score_cat_input)):
                input_valid = True
                score_category = score_char_to_cat(score_cat_input)
            else:
                input_valid = False
                print("category already scored!  Select another")
                print("Available categories are: ")
                player_score.print_available_cats()
        else:
            print("bad input")

    player_score.score_a_category(score_category, myDice)
    print("category scored = ", score_category)
    my_scorecard_string = player_score.print_scorecard()
    for score_line in my_scorecard_string:
        print(score_line)

    all_scored = player_score.all_scored()

print("End of game #############")

my_scorecard_string = player_score.print_scorecard()
comp_scorecard_string = agent_score.print_scorecard()
both_scorecards = zip(my_scorecard_string, comp_scorecard_string)

agent = "Agent"
print(f"Player{agent.rjust(29)}")
for line in both_scorecards:
    print(f"{line[0]}{line[1].rjust(30)}")