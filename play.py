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

myDice = DiceSet()
score = Score()

scores = []
bonus_cnt = 0

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
        myDice.roll()

        if PRINT:
            print("initial roll ", myDice.get_num_rolls())
            print(myDice)
        if print_record_games:
            game_events_to_record.append(f"Turn {turn}\n")
            game_events_to_record.append(f"roll {myDice.get_num_rolls()} dice {myDice.dice()}")

        for roll in range(1, NUM_ROLLS):

            # This is what is used now for the state:
            # Rows are in the same order as score q table
            q_table_keeping_rows_index = q_table_scoring_rows[0].index(
                myDice.get_dict_as_vector() + score.get_available_cat_vector())

            # Action
            action = (ma.masked_array(q_table_keeping[q_table_keeping_rows_index][0:NUM_KEEPING_ACTIONS],
                      keep_action_mask_dict[myDice.as_short_string()])).argmax()
            if PRINT:
                print("action = ", action)

            # About to commit action to reroll
            myDice.make_list_reroll_for_selected_die_faces(list_set_keep_actions[action])

            if PRINT:
                print(f"dice action {action} has list re-roll {myDice.get_list_reroll()}")
            myDice.roll_list_reroll()
            if print_record_games:
                game_events_to_record.append(f"row = {q_table_keeping_rows_index} ")
                game_events_to_record.append(f" action {action} has list re-roll {myDice.get_list_reroll()}\n")
                game_events_to_record.append(f"roll {myDice.get_num_rolls()} dice {myDice.dice()}")

            if PRINT:
                print("at roll ", myDice.get_num_rolls())
                print(myDice)

        # Score category

        # score.score_a_category(score_int_to_cat(state_face_max_die_count), myDice.dice())
        # Commented out the above "easy score", now we use the scoring q_table
        # need to compute q_table_score index
        # hey below whys q_table_scoring_rows[0] -> it's ok!
        q_table_scoring_rows_index = q_table_scoring_rows[0].index(myDice.get_dict_as_vector() + score.get_available_cat_vector())
        category_scored = score.score_with_q_table(q_table_scoring, q_table_scoring_rows_index, myDice)

        if PRINT:
            print("category scored = ", category_scored)
            print("current score = ", score.get_score_dict())
            print("avail cats = ", score.get_available_cat_vector())

        if print_record_games:
            game_events_to_record.append(f"\nScored {score.get_category_score(score_int_to_cat(category_scored))}"
                                         f" in category {score_int_to_cat(category_scored)}\n")

        all_scored = score.all_scored()

    game_score = score.get_total_score()
    print(f"Total score for game {game_number + 1} is {game_score + score.get_bonus()}\n")

    if score.get_bonus() != 0:
        bonus_cnt = bonus_cnt+1

    if print_record_games:
        game_events_to_record.append(f"Total score for game {game_number+1} is {game_score + score.get_bonus()}"
                                     f" bonus is {score.get_bonus()}\n")
        game_events_to_record.append(f"-----\n")

    scores.append(game_score)

average_score = sum(scores) / NUM_GAMES
print("average score = ", average_score)
print("min score = ", min(scores))
print("max score = ", max(scores))
print(f"bonus happened {bonus_cnt} times")

if print_record_games:
    with open("games.txt", "wt") as file_record_games:
        file_record_games.writelines(game_events_to_record)

# Histogram
np_scores = np.array(scores)
num_bins = 30
n, bins, patches = plt.hist(scores, density=True, bins=30)  #  num_bins, facecolor='blue', alpha=0.5)
plt.show()
