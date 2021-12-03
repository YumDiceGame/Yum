import pickle

from numpy import ma

from do_q_table import do_q_table_rows
from do_keep_action_q_table import action_q_table
from constants import *
from dice import DiceSet

dice = DiceSet()

def reduce_q_table(q_table, q_table_row_indexes, score_table):
    '''
    Transform the q_table to a row vector
    We only need the max action
    '''
    q_table_reduced = []
    q_table_rows_and_data = zip(q_table_row_indexes, q_table)  # q_table_rows is a tuple, want the first element [0]
    i = 0
    for row_index, q_table_row_data in q_table_rows_and_data:
        if score_table:  # we're doing the score_table
            masked_q_table_data = ma.masked_array(q_table_row_data, row_index[NUM_DIE_FACES::])
        else:  # we're doing the keeping table
            dice.list_to_dice_dict((row_index[0:NUM_DIE_FACES]))
            masked_q_table_data = ma.masked_array(q_table_row_data, keep_action_mask_dict[dice.as_short_string()])
        if len(masked_q_table_data) > 0:  # I have to do this because I never removed the rows with all 1's for score
            q_table_reduced.append(int(masked_q_table_data.argmax()))
        else:
            q_table_reduced.append(int(0))
    return q_table_reduced

# Create q_table_rows
q_table_rows = do_q_table_rows()
# Action q table:
action_table = action_q_table()
# Map action to dice to keep and action masks
list_set_keep_actions, keep_action_mask_dict = action_table.print_all_action_q_table()

# Uncomment if running q_table_reduction.py standalone
# and you want to reduce in size the keeping table
# But re-comment out immediately
# Because this operation is called at the end of the keeping training
#
# with open("q_table_keeping.pickle", "rb") as keeping_q_table_file:
#     q_table_keeping = pickle.load(keeping_q_table_file)
# q_table_keeping_reduced = reduce_q_table(q_table_keeping, q_table_rows[0], False)
# with open("q_table_keeping_reduced.pickle", "wb") as f:
#     pickle.dump(q_table_keeping_reduced, f)

# Uncomment if running q_table_reduction.py standalone
# and you want to reduce in size the scoring table
# But re-comment out immediately
# Because this operation is called at the end of the keeping training
#
# with open("q_table_scoring.pickle", "rb") as score_q_table_file:
#     q_table_scoring_straight = pickle.load(score_q_table_file)
# q_table_scoring_reduced = reduce_q_table(q_table_scoring_straight, q_table_rows[0], True)
# with open("q_table_scoring_reduced.pickle", "wb") as f:
#     pickle.dump(q_table_scoring_reduced, f)

