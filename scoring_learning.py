# Training scoring with above the line, Yum, Straight, Full
# with Q learning

# Last training (which was not bad) on Oct 6:
# LEARNING_RATE = 0.2
# DISCOUNT = 0.4
# NUM_EPISODES = 12_000_000
# START_EPSILON_DECAYING = 500_000
# END_EPSILON_DECAYING = NUM_EPISODES-1

# Redoing it as above except NUM_EPISODES = 15_000_000
# Reason: my roll_Yum wasn't working !!

from dice import DiceSet
from score import *
from constants import *
import pickle
import random
import os
from do_q_table import do_q_table_rows
import numpy as np
import time

do_epsilon = True
Testing_Seq = False
Use_prior_q_table = False
Save_q_table = True
Auto_shutdown = True
PRINT = False
PRINT_L2 = False
track_diff = False
num_show = 5000

if do_epsilon:
    epsilon = 1
    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
else:
    # Set epsilon to 0
    epsilon = 0

# create q_table_rows
q_table_rows, list_of_die_face_counts, list_scoreable_categories = do_q_table_rows()
q_table_height = len(q_table_rows)  # this is 516096 !!!

# Load q-table
# see "do q_table.py" for info

if Use_prior_q_table:
    with open("q_table_scoring.pickle", "rb") as f:
        q_table_scoring = pickle.load(f)
else:
    q_table_scoring = np.random.uniform(low=0, high=1, size=(q_table_height, NUM_SCORE_CATEGORIES))

myDice = DiceSet()
score = Score()

# This is to try to follow the progress of the q_table
# if the maxes stop moving around, then we have converged (for better or for worse!)
if track_diff:
    q_table_track_max = np.zeros(q_table_height)
    # collect all the maxes per row ... initially it's meaningless
    for q_table_row in range(0, q_table_height):
        q_table_track_mask = number_to_bits_vector(q_table_row % TWO_TO_NUM_SCORE_CATEGORIES)
        q_table_track_max[q_table_row] = (ma.masked_array(q_table_scoring[q_table_row][0:NUM_SCORE_CATEGORIES],
                                                          q_table_track_mask)).argmax()
    with open("q_table_scoring_track_progress.txt", "w") as f:
        f.write(f"episode\tdiff\n")

scores = []
yum_scores = 0  # how many times it will score Yum

all_scored = False

for episode in range(1, NUM_EPISODES):

    if episode % num_show == 0:
        print("------")
        print("Episode = ", episode)

    # if it's time to inject a Yum, Straight or Full, do this
    # so as to not have it at every turn!
    Turn_for_injection = np.random.randint(1, NUM_SCORE_CATEGORIES+1)

    score.reset_scores()
    all_scored = False
    Turn = 1

    while not all_scored:

        myDice.reset_num_rolls()
        myDice.reset_list_reroll()

        if PRINT:
            print("***")

        # Inject some Yums, Straights and Fulls so that the algorithm can see enough of them
        if Turn == Turn_for_injection:
            if episode % 8 == 0:
                myDice.roll_Yum()
            elif episode % 7 == 0:
                myDice.roll_Straight()
            elif episode % 6 == 0:
                myDice.roll_Full()
            elif episode % 9 == 0:
                myDice.roll_Heavy()
        else:
            myDice.roll()

        # elif Testing_Seq:
        #     # Set dice to pre-programmed sequence for quick test
        #     myDice.seq(Turn)

        # The state now is a concatenation of the dice dict and scored cats
        state = myDice.get_dict_as_vector() + score.get_available_cat_vector()
        # Get q_table_row number as "state_index"

        # time_state_access = time.time()
        # segmented lookup
        # print(myDice)
        state_index_dice = list_of_die_face_counts.index(myDice.get_dict_as_vector())
        state_index_score = list_scoreable_categories.index(score.get_available_cat_vector())
        state_index_calc = state_index_dice * TWO_TO_NUM_SCORE_CATEGORIES + state_index_score
        state_index = state_index_calc  # q_table_rows.index(state) index() is slow!
        # print(f"STATE INDEX CALC = {state_index_calc}  STATE INDEX = {state_index}")
        # print(f"time elapsed for state index = {time.time() - time_state_access}")

        if np.random.random() > epsilon:
            action = (ma.masked_array(q_table_scoring[state_index][0:NUM_SCORE_CATEGORIES],
                                      score.get_available_cat_vector())).argmax()
            # action = np.argmax(q_table_scoring[state_index][0:NUM_SCORE_CATEGORIES])
        else:
            # Get random die face keeping action
            # But don't forget to consider category masking
            # action = np.random.randint(0, NUM_SCORE_CATEGORIES)
            possible_random_actions = ma.masked_array([*range(0, NUM_SCORE_CATEGORIES)], score.get_available_cat_vector())
            masked_random_actions = possible_random_actions[possible_random_actions.mask == False]
            action = random.choice(masked_random_actions)

        scored_cat = score_int_to_cat(action + 1)
        scored_amount = score.get_category_score(scored_cat)

        # Before scoring, save face_max_die_count
        # Doing this because don't want to mask for Yum with other categories (Yum is always ok)
        # Doesn't work since we added straight and full, so only consider "above line" plus Yum
        max_die_count = face_max_die_count = 0
        if not score.is_above_the_line_all_scored():
            max_die_count, face_max_die_count = myDice.max_die_count_for_available_category(score.get_available_cat_vector())

        # shorthand defs:
        # Have to do this before scoring !!
        can_yum = myDice.is_yum() and score.is_category_available('Yum')
        can_full = myDice.is_full() and score.is_category_available('Full')
        can_straight = myDice.is_straight() and score.is_category_available('Straight')

        # Score category
        scored_cat = score_int_to_cat(action + 1)
        score.score_a_category(scored_cat, myDice)
        scored_amount = score.get_category_score(scored_cat)

        # Reward
        reward = 0

        if can_yum:
            if scored_cat == 'Yum':
                reward += 200
            else:
                reward += -240
        else:  # anything other than Yum
            # prioritize above the line scoring
            if can_straight:
                if scored_cat == 'Straight':
                    reward += 150
                else:
                    reward += -200
            elif can_full:
                if scored_cat == 'Full':
                    reward += 150
                else:
                    reward += -250
            elif max_die_count >= 3:  #  and not can_full:  #
                if scored_cat != score_int_to_cat(face_max_die_count):
                    reward -= (90 + 30 * face_max_die_count)  # really need to score 3 of a kind above the line!
                    # also pro-rate
                else:  # max_die_count >= 3:  # Right category, and pretty good score, reward!
                    reward += 30 * face_max_die_count  # prorate according to face max die count

            # Above the line items
            elif scored_cat in ABOVE_THE_LINE_CATEGORIES:
                num_dice_scored = int(scored_amount / score_cat_to_int(scored_cat))
                if num_dice_scored <= 2:  # Right category, but too low score 2 is bad
                    reward += (-30*score_cat_to_int(scored_cat))  # prorate according to face max die count
                    if 0 < max_die_count <= 1:  # 1 is worse
                        reward += (-45*score_cat_to_int(scored_cat))  # prorate according to face max die count
                    # because low score in 6's not as bad as low score in 1's
                    # if episode % num_show == 0:
                    #     print(f"low mdc {reward}")

            # Hi Lo --->
            # Hi >= 22
            # 21 <= Lo < Hi
            elif scored_cat == 'High' or scored_cat == 'Low':
                # define shorthand quantities
                dice_sum = myDice.sum()
                hi_scored = not score.is_category_available('High')
                hi_score = score.get_category_score('High')
                lo_scored = not score.is_category_available('Low')
                lo_score = score.get_category_score('Low')
                if dice_sum >= 21:
                    if (hi_scored and hi_score > 0) and (lo_scored and lo_score > 0):
                        reward += 70  # we managed to score both high and low!
                        # if episode % num_show == 0:
                        #     print(f"hi and lo! {reward}")
                    elif scored_amount > 0:
                        # the below controls if you scored a too aggressive hi or low
                        # note that scored_amount isn't necessarily equal to dice sum!
                        # scored amount could be zero!
                        reward += score.assess_lo_hi_score(scored_amount, scored_cat)
                        # if episode % num_show == 0:
                        #     print(f"hi lo assess {reward}")
                    else:  # you were locked out!
                        reward -= 40
                        # if episode % num_show == 0:
                        #     print(f"hi lo locked out {reward}")
                else:  # dice sum too low
                    reward -= 40
                    # if episode % num_show == 0:
                    #     print(f"hi lo no dice {reward}")
                # Hi Lo <---

            # Scratching a category --->
            if scored_amount == 0:
                reward -= score.scratch_penalty(scored_cat)
                # if episode % num_show == 0:
                #     print(f"scratch {reward}")
            # Scratching a category <---

        # if episode % num_show == 0:
        #     print("reward end of turn = ", reward)

        # New state
        new_state = myDice.get_dict_as_vector() + score.get_available_cat_vector()

        # get q_table_row number as "state_index"
        new_state_index_dice = list_of_die_face_counts.index(myDice.get_dict_as_vector())
        new_state_index_score = list_scoreable_categories.index(score.get_available_cat_vector())
        new_state_index_calc = new_state_index_dice * TWO_TO_NUM_SCORE_CATEGORIES + new_state_index_score
        new_state_index = new_state_index_calc  # q_table_rows.index(new_state) index() is slow!

        # Maximum possible Q value in next step (for new state)
        max_future_q = np.max(q_table_scoring[new_state_index][action])
        if PRINT:
            print("max future q = ", max_future_q)

        # Current Q value (for current state and performed action)
        current_q = q_table_scoring[state_index][action]

        # And here's our equation for a new Q value for current state and dice_action
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # Update Q table with new Q value
        q_table_scoring[state_index][action] = new_q
        if PRINT:
            print("new_q = ", q_table_scoring[state_index][action])

        all_scored = score.all_scored()

        Turn += 1

    if episode % num_show == 0:
        print("score = ", score.get_total_score())
        # score.print_scorecard()

    if episode != 0 and episode % EVAL_Q_TABLE == 0 and track_diff:
        # Evaluate q table
        # compare the new q_table maxes to the previous ones
        # and count the number of differences
        count_diff_maxes = 0
        for q_table_row in range(0, q_table_height):
            # identify differences
            q_table_track_mask = number_to_bits_vector(q_table_row % TWO_TO_NUM_SCORE_CATEGORIES)
            qt_row_max = (ma.masked_array(q_table_scoring[q_table_row][0:NUM_SCORE_CATEGORIES], q_table_track_mask)).argmax()
            if q_table_track_max[q_table_row] != qt_row_max:
                # with open("q_table_scoring_diff.txt", "a") as f2:
                #     f2.write(f"---\n")
                #     f2.write(f"row = {q_table_row} was {q_table_track_max[q_table_row]} is {qt_row_max}\n")
                #     f2.write(f"qtable = {q_table_scoring[q_table_row]}\n")
                #     f2.write(f"mask = {q_table_track_mask}\n")
                prnt_qe_t_max = q_table_track_max[q_table_row]
                count_diff_maxes += 1
            # update q_table_track_max for comparison next EVAL_Q_TABLE
            q_table_track_max[q_table_row] = qt_row_max
        print(f"q_table_diff = {count_diff_maxes}\n")
        with open("q_table_scoring_track_progress.txt", "a") as f:
            f.write(f"{episode}\t{count_diff_maxes}\n")

    # Decaying is being done every episode if episode number is within decaying range
    if do_epsilon:
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value
        # Save q table regularly
        # if episode % 100000 == 0:
        #     with open(f"q_table_scoring_{episode}.pickle", "wb") as f:
        #         pickle.dump(q_table_scoring, f)

# Save q table
if Save_q_table:
    with open("q_table_scoring.pickle", "wb") as f:
        pickle.dump(q_table_scoring, f)

# If you want to train a long one and want to shutdown unattended
if Auto_shutdown:
    os.system("shutdown -P now")



