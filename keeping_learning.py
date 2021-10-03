# Playing Yum game above the line only (1's through 6's)
# with Q learning


import os
import pickle
import random
import numpy as np
from numpy import ma

from constants import *
from do_keep_action_q_table import action_q_table
from do_q_table import do_q_table_rows
from score import *
from keeping_train import KeepingTrain

Train = True
if Train:
    do_epsilon = True
    Use_prior_q_table = False
    Save_q_table = False
    PRINT = False
else:
    do_epsilon = True
    Use_prior_q_table = True
    Save_q_table = False
    PRINT = False

Auto_shutdown = False


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
    q_table_keeping = None
    # q_table_keeping = np.random.uniform(low=0, high=1, size=(q_table_height, NUM_KEEPING_ACTIONS))


myDice = DiceSet()
score = Score()
scores = []

# for score tracking and other -->
track_score = True
track_average_score = 0
track_score_array = np.zeros(6)
# for score tracking <--

keeping_train = KeepingTrain()

discounts = learning_rates = list(range(20, 100, 20))
# LR=0.2 and DIS=0.4 seems the "best"

for learning_rate in learning_rates:
    for discount in discounts:
        q_table_keeping = np.random.uniform(low=0, high=1, size=(q_table_height, NUM_KEEPING_ACTIONS))
        print(f"LR = {learning_rate/100} DIS = {discount/100}")
        keeping_train.train(q_table_scoring, q_table_keeping, list_all_dice_rolls, list_scoreable_categories,
                            keeping_actions_masks, action_to_dice_to_keep, learning_rate/100, discount/100)

# Save q table
if Save_q_table:
    with open("q_table_keeping.pickle", "wb") as f:
        pickle.dump(q_table_keeping, f)

# If you want to train a long one and want to shutdown unattended
if Auto_shutdown:
    os.system("shutdown -P now")


