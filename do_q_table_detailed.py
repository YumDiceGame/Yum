from constants import *
from dice import DiceSet
import numpy as np

myDice = DiceSet()
set_of_dice = set()
# set_of_dice_rolls = set()


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
tuple_list_of_die_face_counts.sort()  # set (reverse=True) to get the same order as the action q table

# Convert back to list:
list_of_die_face_counts = []
for row in range(0, len(tuple_list_of_die_face_counts)):
    list_of_die_face_counts.append(list(tuple_list_of_die_face_counts[row]))
# for reference, I printed the table
with open("list_of_die_face_counts.txt", "w") as write_file:
    # header: how many 1's, 2's, ...
    write_file.write("1 2 3 4 5 6\n")
    write_file.write('-----------\n')
    for row in list_of_die_face_counts:
        # the below strips '[', ']' and also the comma
        # (can't use "strip" for the comma)
        write_file.write((str(row).strip('[]')).replace(',', ''))
        write_file.write('\n')

# To complete the observation table,
# We need to add the scoreable categories.
# There are six categories, so 64 permutations
# it's binary: category is available ("0" as in un-scored) or it is not ("1" means scored)
# probably we don't need the all zeros because that doesn't happen (so 63)
cnt = 0
scoreable_categories = []
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
# scoreable_categories.pop()  # yes makes sense but leave for now

# for reference, let's also print this one too
with open("scoreable_categories.txt", "w") as write_file:
    # header: score the 1's, 2's, ... (7 is Yum)
    write_file.write("1 2 3 4 5 6 Y S F L H\n")
    write_file.write('---------------------\n')
    for row in scoreable_categories:
        # the below strips '[', ']' and also the comma
        # (can't use "strip" for the comma)
        write_file.write((str(row).strip('[]')).replace(',', ''))
        write_file.write('\n')

# The number of rows in the table is unfortunately 252x63 = 15876
# q_table_rows: those aren't the actual q values, but just the rows

q_table_rows = []
for i in range(0, len(list_of_die_face_counts)):
    for j in range(0, len(scoreable_categories)):
        q_table_rows.append(list_of_die_face_counts[i] + scoreable_categories[j])

# We can access a particular row (this will be important):
# print("example accessed row 0 0 3 0 2 0 0 1 1 1 1 0 0 number = ",
#       q_table_rows.index([0, 0, 3, 0, 2, 0, 0, 1, 1, 1, 1, 0, 0]))

# for reference, let's also print this one too
# might be useful
row_number = 0
with open("q_table_rows.txt", "w") as write_file:
    for row in q_table_rows:
        if row_number % 45 == 0:
            if row_number != 0:
                write_file.write('\n')
            # header: how many 1's, 2's, ... AND score the 1's, 2's, (7 is Yum) ... and then row number
            write_file.write("1 2 3 4 5 6 1 2 3 4 5 6 Y S F L H row\n")
            write_file.write("-------------------------------------\n")
        # the below strips '[', ']' and also the comma
        # (can't use "strip" for the comma)
        row_to_write = (str(row).strip('[]')).replace(',', '') + " " + str(row_number)
        write_file.write(row_to_write)
        write_file.write('\n')
        row_number += 1



