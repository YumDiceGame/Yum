# for printing a row from a q table

import pickle
from do_keep_action_q_table import *
from dice import DiceSet

Dice = DiceSet()
ActionQtable = action_q_table()

with open("q_table_keeping.pickle", "rb") as f:
    q_table = pickle.load(f)

row = 1024

max_q_table_index = q_table[row][0:NUM_KEEPING_ACTIONS].argmax()

for action, q_table_row_el in zip(ActionQtable.list_set_keep_actions, q_table[row]):
    print(f'{action} {q_table_row_el:.4}  ')
print(f'max index is {max_q_table_index} action is {ActionQtable.list_set_keep_actions[max_q_table_index]}')
