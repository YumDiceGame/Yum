# Training scoring with above the line, Yum, Straight, Full
# with Q learning


from dice import DiceSet
from score import *
from constants import *
import pickle
import random
import os
from do_q_table import do_q_table_rows
import numpy as np
import time

do_epsilon = False
Testing_Seq = False
Use_prior_q_table = True
Save_q_table = True
Auto_shutdown = True
PRINT = False
PRINT_L2 = False
num_show = 1000

# Exploration settings
# Epsilon is not a constant, it will be decayed
# High epsilon means high random action
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = NUM_EPISODES//2

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
        if episode % 100 == 0 and Turn == Turn_for_injection:
            myDice.roll_Yum()
        elif episode % 110 == 0 and Turn == Turn_for_injection:
            myDice.roll_Straight()
        elif episode % 120 == 0 and Turn == Turn_for_injection:
            myDice.roll_Full()
        else:
            myDice.roll()

        # elif Testing_Seq:
        #     # Set dice to pre-programmed sequence for quick test
        #     myDice.seq(Turn)

        if PRINT:
            print("roll = ", myDice)
            print("Turn = ", Turn)

        # The state now is a concatenation of the dice dict and scored cats
        state = myDice.get_dict_as_vector() + score.get_available_cat_vector()
        # Get q_table_row number as "state_index"

        # time_state_access = time.time()
        # segmented lookup
        state_index_dice = list_of_die_face_counts.index(myDice.get_dict_as_vector())
        state_index_score = list_scoreable_categories.index(score.get_available_cat_vector())
        state_index_calc = state_index_dice * TWO_TO_NUM_SCORE_CATEGORIES + state_index_score
        state_index = state_index_calc  # q_table_rows.index(state) index() is slow!
        # print(f"STATE INDEX CALC = {state_index_calc}  STATE INDEX = {state_index}")
        # print(f"time elapsed for state index = {time.time() - time_state_access}")

        if PRINT:
            print("state = ", state)

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

        if PRINT:
            print("action = ", action)

        # Before scoring, save face_max_die_count
        # Doing this because don't want to mask for Yum with other categories (Yum is always ok)
        # Doesn't work since we added straight and full, so only consider "above line" plus Yum
        max_die_count = face_max_die_count = 0
        if not score.is_above_the_line_all_scored():
            max_die_count, face_max_die_count = myDice.max_die_count_for_available_category(score.get_available_cat_vector())

        # Score category
        if PRINT:
            print(f"Trying to score in category {action+1}")
        # re_score_alert = score.score_category_allow_re_score(score_int_to_cat(action+1), myDice.dice())
        re_score_alert = score.score_a_category(score_int_to_cat(action + 1), myDice)

        # Reward
        reward = score.get_total_score()


        # First thing: assign "special punishments"
        # Gotta punish if it makes the classic wrong move of scoring in wrong category
        # special_reward = 0
        # If you have Yum and you are not scoring it, punish (if Yum is available of course)
        if score_int_to_cat(action+1) != 'Yum' and myDice.is_yum() and not score.get_score_dict()['Yum'][0]:
            reward -= 120
            # special_reward = reward
            yum_scores += 1
        elif (score_int_to_cat(action+1) != 'Straight') and myDice.is_straight() and not score.get_score_dict()['Straight'][0]:
            reward += -60
            # special_reward = reward
        # Should have scored full (if full is available)
        elif (score_int_to_cat(action+1) != 'Full') and myDice.is_full() and not score.get_score_dict()['Full'][0]:
            reward += -60
            # special_reward = reward


        # Above the line items
        elif (action+1) <= NUM_SCORE_CAT_ABOVE_LINE:
        # Score in the 1s - 6s if you have a good hand (if category available)
            if (action+1) != face_max_die_count:  # Scoring wrong category
                reward += -90
                # special_reward = reward
            else:
                if max_die_count <= 2:  # Right category, but too low score
                    reward += (-30) * face_max_die_count  # prorate according to face max die count
                    # because low score in 6's not as bad as low score in 1's
                else:  # max_die_count >= 3:  # Right category, and pretty good score, reward!
                    reward += 30 * face_max_die_count  # prorate according to face max die count

        ### Hi Lo --->
        # Hi >= 22
        # 21 <= Lo < Hi
        elif (dice_sum := myDice.sum()) > 21 and (scored_cat := score_int_to_cat(action + 1)) != 'Full'\
                and (scored_cat != 'Yum'):
            # define shorthand quantities
            hi_scored = score.get_score_dict()['High'][0]
            hi_score = score.get_score_dict()['High'][1]
            lo_scored = score.get_score_dict()['Low'][0]
            lo_score = score.get_score_dict()['Low'][1]

            # Next: missing scoring a good low
            if hi_scored and not lo_scored:
                if (scored_cat != 'Low') and (dice_sum < hi_score):
                    reward += -70
            # Next: don't want to miss out on scoring high (given low is scored):
            if lo_scored and not hi_scored:
                if (scored_cat != 'High') and (dice_sum > lo_score):
                    reward += -75
            # Give praise below:
            elif scored_cat == 'High':
                reward += 30
                if lo_scored:
                    reward += 40
            elif scored_cat == 'Low':
                reward += 30
                if hi_scored:
                    reward += 40

        ### Hi Lo <---

        if PRINT:
            print("get tot score = ", score.get_total_score())
            print("reward = ", reward)
            print("current score dict = ", score.get_score_dict())
            print("avail cats = ", score.get_available_cat_vector())

        # New state
        new_state = myDice.get_dict_as_vector() + score.get_available_cat_vector()

        # get q_table_row number as "state_index"
        new_state_index_dice = list_of_die_face_counts.index(myDice.get_dict_as_vector())
        new_state_index_score = list_scoreable_categories.index(score.get_available_cat_vector())
        new_state_index_calc = new_state_index_dice * TWO_TO_NUM_SCORE_CATEGORIES + new_state_index_score
        new_state_index = new_state_index_calc  # q_table_rows.index(new_state) index() is slow!

        if PRINT:
            print("new state = ", new_state)
            print("new_state_index = ", new_state_index)

        # Maximum possible Q value in next step (for new state)
        max_future_q = np.max(q_table_scoring[new_state_index][action])
        if PRINT:
            print("max future q = ", max_future_q)

        # Current Q value (for current state and performed action)
        current_q = q_table_scoring[state_index][action]
        if PRINT:
            print("current q = ", current_q)

        # And here's our equation for a new Q value for current state and dice_action
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        # if special_reward == 0:
        #     new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        # else:
        #     new_q = special_reward

        # Update Q table with new Q value
        q_table_scoring[state_index][action] = new_q
        if PRINT:
            print("new_q = ", q_table_scoring[state_index][action])

        all_scored = score.all_scored()

        Turn += 1

    if episode % num_show == 0:
        print("score = ", score.get_total_score())
        score.print_scorecard()

    # While not all_scored

    # game_score = score.get_total_score()
    # print("total score = ", game_score)
    # scores.append(game_score)

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

