# Playing Yum game above the line only (1's through 6's)
# with Q learning


import numpy as np
from dice import DiceSet
from score import *
from constants import *
import pickle
import random
import os
from numpy import ma
from do_q_table import do_q_table_rows
from do_keep_action_q_table import action_q_table
import time

Train = True
if Train:
    do_epsilon = True
    Use_prior_q_table = False
    Save_q_table = True
    PRINT = False
else:
    do_epsilon = True
    Use_prior_q_table = True
    Save_q_table = False
    PRINT = False

Auto_shutdown = False


def calc_row_index():
    # We calculate the row index fairly often ...
    index_dice = list_all_dice_rolls.index(myDice.dice())
    index_score = list_scoreable_categories.index(score.get_available_cat_vector())
    row_index = index_dice * TWO_TO_NUM_SCORE_CATEGORIES + index_score
    return row_index

# Exploration settings
# Epsilon is not a constant, it will be decayed
# High epsilon means high random action
# START_EPSILON_DECAYING = 1
# END_EPSILON_DECAYING = NUM_EPISODES//2
START_EPSILON_DECAYING = NUM_EPISODES//2
END_EPSILON_DECAYING = NUM_EPISODES-1

if do_epsilon:
    epsilon = 1
    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
else:
    # Set epsilon to 0
    epsilon = 0

# Load q-tables:
# "q_table_2d_yum.pickle" for the dice keeping actions
# with open("q_table_2d_yum.pickle", "rb") as f:
#     q_table = pickle.load(f)
# And "" for the scoring actions
with open("q_table_scoring.pickle", "rb") as score_q_table_file:
    q_table_scoring = pickle.load(score_q_table_file)

# Create q_table_scoring_rows
q_table_scoring_rows, list_of_die_face_counts, list_scoreable_categories = do_q_table_rows()
q_table_height = len(q_table_scoring_rows)  #

##########################
# for the keeping actions
# rows are the same as scoring, except they are roll referred here for keeping
# (in scoring it's dictionary referred, list_of_die_face_counts is dict)
action_table = action_q_table()
# keep actions plus masks:
action_to_dice_to_keep, keeping_actions_masks = action_table.print_all_action_q_table()

# Finally the list of dice roll, why was it so hard to do this !!
list_all_dice_rolls = action_table.do_list_of_dice_rolls()

# Load q-table
# see "do q_table.py" for info
if Use_prior_q_table:
    with open("q_table_keeping.pickle", "rb") as f:
        q_table_keeping = pickle.load(f)
else:
    q_table_keeping = np.random.uniform(low=0, high=1, size=(q_table_height, NUM_KEEPING_ACTIONS))

# This is to try to follow the progress of the q_table
# if the maxes stop moving around, then we have converged (for better or for worse!)
q_table_track_max = np.zeros(q_table_height)
# collect all the maxes per row ... initially it's meaningless
for q_table_row in range(0, q_table_height):
    q_table_track_max[q_table_row] = q_table_keeping[q_table_row][0:NUM_KEEPING_ACTIONS].argmax()
with open("q_table_track_progress.txt", "w") as f:
    f.write(f"episode\tdiff\n")

# This is to follow the evolution of the scoring
with open("score_track_progress.txt", "w") as f:
    f.write(f"episode\tepsilon\tscore\tbonus\tstraight\tfull\tlow\thigh\tyum\n")

myDice = DiceSet()
score = Score()
scores = []

# for score tracking -->
track_score = True
track_average_score = 0
track_score_array = np.zeros(6)
# for score tracking <--

for episode in range(NUM_EPISODES+1):

    turn = 0
    score.reset_scores()
    all_scored = False

    while not all_scored:
        turn += 1
        myDice.reset()
        myDice.roll()
        # max_die_count: it's back
        if not score.is_above_the_line_all_scored():
            max_die_count_previous, face_max_die_count = myDice.max_die_count_for_available_category(
                score.get_available_cat_vector())

        # for tracking when you have almost straight, full or yum
        almost_straight_list_per_roll = []
        almost_full_list_per_roll = []
        almost_yum_list_per_roll = []

        almost_straight_list_per_roll.append(myDice.is_almost_straight())
        almost_full_list_per_roll.append(myDice.is_two_pairs())
        almost_yum_list_per_roll.append(myDice.is_almost_yum())

        potential_max_score_previous = score.get_potential_max_score(myDice)

        # print(f"max die count previous = {max_die_count_previous} face max die count = {face_max_die_count}")

        for roll in range(2, NUM_ROLLS+1):

            # This is the observation part of the state
            # old style when we were doing only max die count
            # max_die_count = face_max_die_count = 0
            # state_max_die_count_and_face = myDice.max_die_count_for_available_category(score.get_available_cat_vector())
            # state_max_die_count = state_max_die_count_and_face[0]
            # state_face_max_die_count = state_max_die_count_and_face[1]
            # max_die_count_alert = False
            # if PRINT:
            #     print("state_max_die_count = ", state_max_die_count)
            #     print("state_face_max_die_count = ", state_face_max_die_count)
            # if print_record_games:
            #     game_events_to_record.append(f"state_max_die_count = {state_max_die_count} state_face_max_die_count = {state_face_max_die_count}")

            # STATE DEFINITION --->
            # The state now is a concatenation of the dice dict and scored cats
            # state = myDice.dice() + score.get_available_cat_vector()
            # Get q_table_row number as "state_index"
            # segmented lookup
            state_index_dice_roll = list_all_dice_rolls.index(myDice.dice())
            state_index_score = list_scoreable_categories.index(score.get_available_cat_vector())
            state_index = state_index_dice_roll * TWO_TO_NUM_SCORE_CATEGORIES + state_index_score
            if PRINT:
                print("state_index_dice_roll = ", state_index_dice_roll)
                print("state_index_score = ", state_index_score)
                print("state_index = ", state_index)
                print("calc index = ", calc_row_index())
            # STATE DEFINITION <---

            # KEEPING ACTION --->
            if np.random.random() < epsilon:
                # Get random dice keeping action
                possible_random_actions = ma.masked_array([*range(0, NUM_KEEPING_ACTIONS)],
                                                          keeping_actions_masks[myDice.as_short_string()])
                masked_random_actions = possible_random_actions[possible_random_actions.mask == False]
                action = random.choice(masked_random_actions)
                if PRINT and episode % NUM_SHOW == 0:
                    print("possible random actions = ", possible_random_actions)
                    print("masked random actions = ", masked_random_actions)
                    print("action RANDOM")
                    print("action = ", action)
            else:
                # action = (ma.masked_array(q_table_keeping[state_index][0:NUM_KEEPING_ACTIONS],
                #                           keeping_actions_masks[myDice.as_short_string()])).argmax()
                action = (ma.masked_array(q_table_keeping[state_index][0:NUM_KEEPING_ACTIONS],
                                          keeping_actions_masks[myDice.as_short_string()])).argmax()
                if PRINT and episode % NUM_SHOW == 0:
                    print("action TABLE")

            myDice.make_list_reroll_for_selected_die_faces(action_to_dice_to_keep[action])
            myDice.roll_list_reroll()

            # for tracking when you have almost straight full or yum
            almost_straight_list_per_roll.append(myDice.is_almost_straight())
            almost_full_list_per_roll.append(myDice.is_two_pairs())
            almost_yum_list_per_roll.append(myDice.is_almost_yum())

            if PRINT and episode % NUM_SHOW == 0:
                print(f"action mask = {keeping_actions_masks[myDice.as_short_string()]}")
                print(f"roll {myDice.get_num_rolls()} action {action} list re-roll {myDice.get_list_reroll()}"
                      f" new dice {myDice} ")

            # KEEPING ACTION <---

            # REWARD --->
            # so here we are going to find out what would be the max potential score across available categories
            # if we were to score this roll right now.
            # We will record this quantity as some trailing max potential score
            #
            # If initial roll, the reward is whatever that potential max score is
            # In subsequent rolls, it gets more interesting:
            # is the potential max score went down, there is a penalty
            # if it went down by 20 or more, there is a large penalty
            # if it's stayed the same, a small reward.
            #   But if the potential max score was large and it stayed large, then we give a large reward
            # if the potential max score went up there is a reward, and a large one if it went up by 20 or more
            # So we need a function that will return potential max score

            if not score.is_above_the_line_all_scored():
                max_die_count, face_max_die_count = myDice.max_die_count_for_available_category(
                    score.get_available_cat_vector())

            reward = potential_max_score = score.get_potential_max_score(myDice)
            #
            # if PRINT:
            #     print(f"roll {myDice.get_num_rolls()} mdc = {max_die_count} fmdc = {face_max_die_count}")
            # shortcut for the reward:
            # right now it's max_die_count * face_max_die_count
            # but that will have to be generalized as explained at the top of this comment section
            # Experiment: normalize to max die count times, say, 3
            # reward = max_die_count * face_max_die_count
            # reward = max_die_count * 3 # face_max_die_count
            # so as before, if max die count went down ...
            # special_reward = 0
            # if PRINT:
            #     print(f"before mdc check  max_die_count_previous = {max_die_count_previous} max_die_count = {max_die_count} ")
            if potential_max_score >= 21:  # was 25: made 21 because want the low
                reward += 90  # this is great
            elif potential_max_score_previous >= (potential_max_score+25):  # if score went down a lot
                reward += -90
            elif reward == 0:  # reward = 0 probably you kept all dice wrongly
                reward += -60
            elif reward >= 15:  # could mean 3 out of 5
                reward += 40
            elif reward >= 10:  #
                reward += 20
            elif 0 <= reward < 10:
                reward -= 30

            potential_max_score_previous = potential_max_score

            # Stuff for encouraging to try straight,full or yum
            if score.is_category_available('Straight'):
                if myDice.is_straight():
                    pass
                elif almost_straight_list_per_roll[roll-2] and not almost_straight_list_per_roll[roll-1]:
                    # we got further away from straight
                    reward -= 50
            if score.is_category_available('Full'):
                if myDice.is_full():
                    pass
                elif almost_full_list_per_roll[roll-2] and not almost_full_list_per_roll[roll-1]:
                    # we got further away from full
                    reward -= 50
            if score.is_category_available('Yum'):
                if myDice.is_yum():
                    pass
                elif almost_yum_list_per_roll[roll-2] and not almost_yum_list_per_roll[roll-1]:
                    # we got further away from yum
                    reward -= 40

            if not score.is_above_the_line_all_scored():  # MDC only if anything left above the line:
                if max_die_count_previous > max_die_count:
                    reward += -60
                elif max_die_count > max_die_count_previous:
                    reward += 40
            max_die_count_previous = max_die_count

            # REWARD <---

            # NEW STATE --->
            # nothing special, we'll just get the new state as always
            # new_state = myDice.get_dict_as_vector() + score.get_available_cat_vector()
            # get q_table_row number as "state_index"
            new_state_index_dice = list_all_dice_rolls.index(myDice.dice())
            new_state_index_score = list_scoreable_categories.index(score.get_available_cat_vector())
            new_state_index = new_state_index_dice * TWO_TO_NUM_SCORE_CATEGORIES + new_state_index_score
            # if PRINT:
            #     print("new_state_index_dice = ", new_state_index_dice)
            #     print("new_state_index_score = ", new_state_index_score)
            #     print("new_state_index = ", new_state_index)
            # NEW STATE <---

            # Q UPDATE --->
            # should be similar to:
            max_future_q = np.max(q_table_keeping[new_state_index][action])
            # if PRINT:
            #     print("max future q = ", max_future_q)
            #
            # # Current Q value (for current state and performed action)
            current_q = q_table_keeping[state_index][action]
            # if PRINT:
            #     print("current q = ", current_q)
            #
            # # And here's our equation for a new Q value for current state and dice_action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            #
            # # Update Q table with new Q value
            q_table_keeping[state_index][action] = new_q
            # if PRINT:
            #     print("new_q = ", q_table_keeping[state_index][action])

            # print(f"Episode {episode} Turn {turn} roll {roll}time roll {time.time() - time_start_roll}")

        # SCORE CATEGORY --->
        # Scoring with previously trained score q table
        # need to compute q_table_score index
        q_table_scoring_index_dice = list_all_dice_rolls.index(myDice.dice())
        q_table_scoring_index_score = list_scoreable_categories.index(score.get_available_cat_vector())
        q_table_scoring_index = q_table_scoring_index_dice * TWO_TO_NUM_SCORE_CATEGORIES + q_table_scoring_index_score
        category_scored = score.score_with_q_table(q_table_scoring, q_table_scoring_index, myDice)

        # SCORE CATEGORY <---

        all_scored = score.all_scored()

    # Decaying is being done every episode if episode number is within decaying range
    if do_epsilon:
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

    if episode % NUM_SHOW == 0:
        print("episode = ", episode)
        print("score = ", score.get_total_score())
        score.print_scorecard()
        print("epsilon = ", epsilon)
        print("\n")

    # Track scoring: how does it evolve over time --->
    if (episode+NUM_GAMES) % NUM_TRACK_SCORE == 0:
        track_score = True
        track_average_score = 0
        track_score_array = np.zeros(6)
    elif episode % NUM_TRACK_SCORE == 0:
        track_score = False
    if track_score:
        # accumulation for average score
        track_average_score += score.get_total_score()
        # accumulation for tracking above the line items such as straight, full...
        track_score_array = np.add(track_score_array, np.array(score.get_above_the_line_success()))

    if episode != 0 and episode % NUM_TRACK_SCORE == 0:
        with open("score_track_progress.txt", "a") as f:
            f.write(f"{episode}\t{epsilon}\t{track_average_score/NUM_GAMES}")
            for item in track_score_array:
                f.write(f"\t{item / NUM_GAMES}")
            f.write("\n")
    # Track scoring: how does it evolve over time <---

    if episode != 0 and episode % EVAL_Q_TABLE == 0:
        # Evaluate q table
        # compare the new q_table maxes to the previous ones
        # and count the number of differences
        count_diff_maxes = 0
        for q_table_row in range (0, q_table_height):
            # identify differences
            if q_table_track_max[q_table_row] != q_table_keeping[q_table_row][0:NUM_KEEPING_ACTIONS].argmax():
                count_diff_maxes += 1
            # update q_table_track_max for comparison next EVAL_Q_TABLE
            q_table_track_max[q_table_row] = q_table_keeping[q_table_row][0:NUM_KEEPING_ACTIONS].argmax()
        print(f"q_table_diff = {count_diff_maxes}\n")
        with open("q_table_track_progress.txt", "a") as f:
            f.write(f"{episode}\t{count_diff_maxes}\n")

    # Save q table periodically
    if Save_q_table and episode != 0 and episode % Q_TABLE_SAVE_INTERVAL == 0:
        with open(f"q_table_keeping_{episode}.pickle", "wb") as f:
            pickle.dump(q_table_keeping, f)

# Save q table
if Save_q_table:
    with open("q_table_keeping.pickle", "wb") as f:
        pickle.dump(q_table_keeping, f)

# If you want to train a long one and want to shutdown unattended
if Auto_shutdown:
    os.system("shutdown -P now")


