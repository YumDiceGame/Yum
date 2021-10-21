# Playing Yum game above the line only (1's through 6's)
# with Q learning


from score import *
from constants import *
import pickle
import numpy as np
from numpy import ma
from do_q_table import do_q_table_rows
from do_keep_action_q_table import action_q_table
import matplotlib.pyplot as plt

PRINT = False

# Load q-tables:
# "q_table_2d_yum.pickle" for the dice keeping actions
# with open("q_table_2d_yum.pickle", "rb") as f:
#     q_table = pickle.load(f)
# And "" for the scoring actions
with open("q_table_scoring.pickle", "rb") as score_q_table_file:
    q_table_scoring = pickle.load(score_q_table_file)

with open("q_table_keeping.pickle", "rb") as keeping_q_table_file:
    q_table_keeping = pickle.load(keeping_q_table_file)

# Create q_table_scoring_rows
q_table_scoring_rows = do_q_table_rows()

# Action q table:
action_table = action_q_table()

# Map action to dice to keep and action masks
list_set_keep_actions, keep_action_mask_dict = action_table.print_all_action_q_table()


# For saving games
game_events_to_record = []
print_record_games = True
roll_seq = False

myDice = DiceSet()
score = Score()

scores = []
bonus_cnt = 0
yum_counter = 0
straight_counter = 0
full_counter = 0
lo_counter = 0
hi_counter = 0
action_60_counter = 0
empty_set = set()

all_scored = False

for game_number in range(NUM_GAMES):

    if print_record_games:
        game_events_to_record.append(f"Game number {game_number+1}\n")

    turn_done = False
    turn = 0

    score.reset_scores()
    all_scored = False

    while not all_scored:

        turn += 1
        myDice.reset()
        # if turn == 1:
        #     myDice.roll_Yum()
        # else:
        myDice.roll()

        if PRINT:
            print("initial roll ", myDice.get_num_rolls())
            print(myDice)
        if print_record_games:
            game_events_to_record.append(f"Turn {turn}\n")
            game_events_to_record.append(f"roll {myDice.get_num_rolls()} dice {myDice.dice()}")

        if not roll_seq:

            for roll in range(1, NUM_ROLLS):

                # This is what is used now for the state:
                # Rows are in the same order as score q table
                q_table_keeping_rows_index = q_table_scoring_rows[0].index(
                    myDice.get_dict_as_vector() + score.get_available_cat_vector())

                # Action
                action = (ma.masked_array(q_table_keeping[q_table_keeping_rows_index][0:NUM_KEEPING_ACTIONS],
                          keep_action_mask_dict[myDice.as_short_string()])).argmax()
                if action == 60:
                    print("action 60")
                    action_60_counter += 1

                # About to commit action to reroll
                myDice.make_list_reroll_for_selected_die_faces(list_set_keep_actions[action])

                dice_set_before_reroll = myDice.as_set()  # for knowing if it's Keep All action
                myDice.roll_list_reroll()

                if print_record_games:
                    game_events_to_record.append(f"row = {q_table_keeping_rows_index} ")
                    if list_set_keep_actions[action] == empty_set:
                        game_events_to_record.append(f" action {action} is keep none\n")
                    elif dice_set_before_reroll == list_set_keep_actions[action]:
                        game_events_to_record.append(f" dice set {myDice.as_set()} action set "
                                                     f"{list_set_keep_actions[action]} action {action} is keep all "
                                                     f"{myDice.get_list_reroll()}\n")
                    else:
                        game_events_to_record.append(f" action {action} is {list_set_keep_actions[action]}\n")
                    game_events_to_record.append(f"roll {myDice.get_num_rolls()} dice {myDice.dice()}")

                if PRINT:
                    print("at roll ", myDice.get_num_rolls())
                    print(myDice)

        else:  # Rolling from a pre-programmed sequence
            myDice.roll_seq(turn)

        # Score category

        # score.score_a_category(score_int_to_cat(state_face_max_die_count), myDice.dice())
        # Commented out the above "easy score", now we use the scoring q_table
        # need to compute q_table_score index
        # hey below whys q_table_scoring_rows[0] -> it's ok!
        q_table_scoring_rows_index = q_table_scoring_rows[0].index(myDice.get_dict_as_vector() + score.get_available_cat_vector())
        category_scored = score.score_with_q_table(q_table_scoring, q_table_scoring_rows_index, myDice)

        if roll_seq:
            print(f"{myDice} category scored = {score_int_to_cat(category_scored)} category score = "
                  f"{score.get_category_score(score_int_to_cat(category_scored))}")

        if print_record_games:
            game_events_to_record.append(f"\nScored {score.get_category_score(score_int_to_cat(category_scored))}"
                                         f" in category {score_int_to_cat(category_scored)}\n")

        all_scored = score.all_scored()

    if score.get_bonus() != 0:
        bonus_cnt = bonus_cnt+1

    if score.get_category_score('Yum') == 30:
        yum_counter += 1
    if score.get_category_score('Straight') == 25:
        straight_counter += 1
    if score.get_category_score('Full') == 25:
        full_counter += 1
    if score.get_category_score('Low') > 0:
        lo_counter += 1
    if score.get_category_score('High') > 0:
        hi_counter += 1

    game_score = score.get_total_score() + score.get_bonus()
    print(f"Total score for game {game_number + 1} is {game_score}\n")

    if print_record_games:
        game_events_to_record.append(f"Total score for game {game_number+1} is {game_score}"
                                     f" bonus is {score.get_bonus()}\n")
        game_events_to_record.append(f"-----\n")

    scores.append(game_score)

average_score = sum(scores) / NUM_GAMES
print("average score = ", average_score)
print("min score = ", min(scores))
print("max score = ", max(scores))
print(f"bonus happened {bonus_cnt} times")
print(f"yum count = {yum_counter}")
print(f"straight count = {straight_counter}")
print(f"full count = {full_counter}")
print(f"lo count = {lo_counter}")
print(f"hi count = {hi_counter}")
print(f"table action 60 happened {action_60_counter} times")

if print_record_games:
    with open("games.txt", "wt") as file_record_games:
        file_record_games.writelines(game_events_to_record)

# Histogram
np_scores = np.array(scores)
num_bins = 30
n, bins, patches = plt.hist(scores, density=True, bins=30)  #  num_bins, facecolor='blue', alpha=0.5)
plt.show()
