from constants import *
from dice import DiceSet
import numpy as np
import pickle

# See "do_q_table_detailed" for "verbose" version
# for example writing to files

myDice = DiceSet()
set_of_dice = set()

def do_q_table_rows():

    for d1 in range(1, NUM_DIE_FACES+1):
        for d2 in range(1, NUM_DIE_FACES + 1):
            for d3 in range(1, NUM_DIE_FACES + 1):
                for d4 in range(1, NUM_DIE_FACES + 1):
                    for d5 in range(1, NUM_DIE_FACES + 1):
                        myDice.set([d1, d2, d3, d4, d5])
                        myDice.get_dict()

                        # do a tuple, because we need a set
                        # because we need to count unique combinations
                        # of quantities
                        dice_tuple = tuple(myDice.get_dict_as_vector())
                        set_of_dice.add(dice_tuple)

    # the below gives only 252!
    # so there are only 252 separate combinations of dice "dictionaries"
    # recall that a "dictionary" has values that say how many of the same dice you have
    # example: 1 1 3 4 5
    # 1's: 2, 3's: 1, 4's: 1, 5's:1 and the rest is 0.
    # print("length of the set!! = ", len(set_of_dice))  # Gives 252
    tuple_list_of_die_face_counts = []
    for i in set_of_dice:
        tuple_list_of_die_face_counts.append(i)

    # sort list of die face counts
    # this is the "left" half of the observation table!
    tuple_list_of_die_face_counts.sort()

    # Convert back to list:
    list_of_die_face_counts = []
    for row in range(0, len(tuple_list_of_die_face_counts)):
        list_of_die_face_counts.append(list(tuple_list_of_die_face_counts[row]))

    # To complete the observation table,
    # We need to add the scoreable categories.
    # There are six categories, so 64 permutations
    # it's binary right ... we can score the category or we can't
    # probably we don't need the all zeros because that doesn't happen (so 63)
    cnt = 0
    scoreable_categories = []

    # '0' means the category is available
    # '1' means the category is not available, as in it has already been scored
    # Jul 23 2021: try to add Yum (c7) and straight (c8) for faster training and debug
    for c1 in range(0, 2):  # "c" for category
        for c2 in range(0, 2):
            for c3 in range(0, 2):
                for c4 in range(0, 2):
                    for c5 in range(0, 2):
                        for c6 in range(0, 2):
                            for c7 in range(0, 2):
                                for c8 in range(0, 2):
                                    for c9 in range(0, 2):
                                        for c10 in range(0, 2):
                                            for c11 in range(0, 2):
                                                scoreable_categories.append([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,
                                                                             c11])
                                                cnt += 1

    # remove the all-scored [1 1 1 1 1 1 1] (last row)
    # because when you get to all scored, the games is already over
    # scoreable_categories.pop()  # Leave for now

    q_table_rows = []
    for i in range(0, len(list_of_die_face_counts)):
        for j in range(0, len(scoreable_categories)):
            q_table_rows.append(list_of_die_face_counts[i] + scoreable_categories[j])

    # return more than just q table rows
    # the list_of_die_face_counts and scoreable_categories
    # will be used for lookups

    return q_table_rows, list_of_die_face_counts, scoreable_categories



