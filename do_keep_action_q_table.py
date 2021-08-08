from dice import *
import numpy as np
myDice = DiceSet()

# This is for creating the action q table
# it's only 252 rows (all possible distinct rolls) by 32 columns (all possible keep actions)
# but this will print details
# Notice that it is based on "DiceSet" because we're using lots of stuff from there

# There are 53 keep actions for 6 dice:
#
# 1 2 3 4 5 6 12 13 14 15 16 23 24 25 26 34 35 36 45 46 56 123 124 125 126 134 135 136 145 146 156 234 235 236 245
# 246 256 345 346 356 456 1234 1235 1236 1345 1346 1356 1456 2345 2346 2356 2456 3456



class action_q_table(DiceSet):
    def __init__(self):
        super().__init__()

    # oh and the first one is the empty set for "keep none" -> always unmasked
    empty_set = set()
    list_set_keep_actions = [empty_set, {1}, {2}, {3}, {4}, {5}, {6}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 3},
                             {2, 4}, {2, 5}, {2, 6}, {3, 4}, {3, 5}, {3, 6}, {4, 5}, {4, 6}, {5, 6}, {1, 2, 3},
                             {1, 2, 4}, {1, 2, 5}, {1, 2, 6}, {1, 3, 4}, {1, 3, 5}, {1, 3, 6}, {1, 4, 5}, {1, 4, 6},
                             {1, 5, 6}, {2, 3, 4}, {2, 3, 5}, {2, 3, 6}, {2, 4, 5}, {2, 4, 6}, {2, 5, 6}, {3, 4, 5},
                             {3, 4, 6, }, {3, 5, 6}, {4, 5, 6}, {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 3, 6}, {1, 3, 4, 5},
                             {1, 3, 4, 6}, {1, 3, 5, 6}, {1, 4, 5, 6}, {2, 3, 4, 5}, {2, 3, 4, 6}, {2, 3, 5, 6},
                             {2, 4, 5, 6}, {3, 4, 5, 6}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 6}, {1, 2, 3, 5, 6},
                             {1, 2, 4, 5, 6}, {1, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {'K4'}]

    def print_all_action_q_table(self):
        # Prints the following in "list_of_die_rolls.txt":
        # - all possible rolls (w/o any duplicates) (section 1, cols 1-4)
        # - corresponding dictionary (section 2, cols 1-5)
        # - after that the wide columns 0 through F are the resulting (smaller) dictionaries after a keep action
        # -- for that see "def all_actions_to_list_reroll" below and "list_all_keep_actions.txt"
        keep_action_mask_dict = {}
        row_number = 0
        with open("list_of_die_rolls.txt", "w") as write_file:
            # Header
            #                      dice            dict
            write_file.write("  row 1 2 3 4 5 | 1 2 3 4 5 6 | action mask\n")
            write_file.write('  -----------------------------------------\n')
            action_mask = ""  # this will be for writing action masks to file
            # For removing duplicates rolls
            dice_set_for_dups = set()
            for d1 in range(1, NUM_DIE_FACES + 1):
                for d2 in range(1, NUM_DIE_FACES + 1):
                    for d3 in range(1, NUM_DIE_FACES + 1):
                        for d4 in range(1, NUM_DIE_FACES + 1):
                            for d5 in range(1, NUM_DIE_FACES + 1):
                                self.set([d1, d2, d3, d4, d5])
                                self._dice.sort()
                                # Remove duplicates
                                # not working quite yet
                                self.as_dict()
                                dice_str_for_dups = self.__str__()
                                if dice_str_for_dups not in dice_set_for_dups:
                                    dice_set_for_dups.add(dice_str_for_dups)
                                    write_file.write(
                                        str.rjust(str(row_number), 4) + "  " + (str(self._dice).strip('[]')).replace(',', ''))
                                    write_file.write("   ")
                                    write_file.write(str(self._dict.values()).strip('[]()dict_values').replace(',', ''))
                                    write_file.write("   ")
                                    # go thru all the possible keep actions
                                    # we mask the action if not all the dice are included in the keep action
                                    # One difference: action
                                    action_mask = ""
                                    for keep_action in self.list_set_keep_actions:
                                        if keep_action == {'K4'}:
                                            if len(collections.Counter(self._dice)) == 4:
                                                # This means that there is one pair and thee singletons
                                                # and we need an action specifically for this case
                                                action_mask += '0'
                                            else:
                                                action_mask += '1'
                                        elif keep_action.issubset(self.as_set()):
                                            action_mask += '0'
                                        else:
                                            action_mask += '1'
                                    # add to dict of masks per rolls:
                                    # some stuff to translate the action mask string to something useable:
                                    action_mask_int = np.fromstring(action_mask, np.int8) - 48
                                    keep_action_mask_dict[self.as_short_string()] = action_mask_int
                                    write_file.write(f" {action_mask}\n")
                                    row_number += 1

        return self.list_set_keep_actions, keep_action_mask_dict

    def do_list_of_dice_rolls(self):
        # Why this function:
        # it is for the state in keeping learning: it needs a list, not a string like what the other functions here do
        list_of_dice_rolls = []
        list_of_dice_rolls_no_dups = []
        for d1 in range(1, NUM_DIE_FACES + 1):
            for d2 in range(1, NUM_DIE_FACES + 1):
                for d3 in range(1, NUM_DIE_FACES + 1):
                    for d4 in range(1, NUM_DIE_FACES + 1):
                        for d5 in range(1, NUM_DIE_FACES + 1):
                            dice_rolls = [d1, d2, d3, d4, d5]
                            dice_rolls = sorted(dice_rolls)
                            list_of_dice_rolls.append(dice_rolls)
        # more fun way to remove duplicates:
        for roll in list_of_dice_rolls:
            if roll not in list_of_dice_rolls_no_dups:
                list_of_dice_rolls_no_dups.append(roll)
        # don't forget to revers it:
        return list_of_dice_rolls_no_dups[::-1]

    # def all_actions_to_dice_to_keep(self):
    #     # UPdate to 6 dice version!
    #     # Makes a corespondence between the action number and the dice to keep
    #     # Replaces "all_actions_to_list_reroll"
    #     # See "roll_masks.ods", look at the top row
    #     actions_dict = {0: [], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [1, 2], 7: [1, 3], 8: [1, 4], 9: [1, 5],
    #                     10: [2, 3], 11: [2, 4], 12: [2, 5], 13: [3, 4], 14: [3, 5], 15: [4, 5], 16: [1, 2, 3],
    #                     17: [1, 2, 4], 18: [1, 2, 5], 19: [1, 3, 4], 20: [1, 3, 5], 21: [1, 4, 5], 22: [2, 3, 4],
    #                     23: [2, 3, 5], 24: [2, 4, 5], 25: [3, 4, 5], 26: [1, 2, 3, 4], 27: [1, 2, 3, 5],
    #                     28: [1, 2, 4, 5], 29: [1, 3, 4, 5], 30: [2, 3, 4, 5]}
    #     return actions_dict

    # def get_roll_and_action_masks_dict(self):
    #     # Associate an action mast to a roll
    #     roll_to_mask_dict = {}
    #     with open("masks_n_rolls.txt", "r") as q_table_rows:
    #         lines = q_table_rows.readlines()
    #     for line in lines:
    #         line = line.strip("\n")
    #         line_split = line.split()
    #         # Action mask:
    #         line_action_mask = np.fromstring(line_split[0], np.int8) - 48
    #         # the roll
    #         line_roll = line_split[1]
    #         # Finally the long awaited simple dict of roll/mask!
    #         roll_to_mask_dict[line_roll] = line_action_mask
    #     return roll_to_mask_dict

    # def do_scoreable_categories(self):
    #     # as seen elsewhere, but now do a little function
    #     scoreable_categories = []
    #     for c1 in range(0, 2):  # "c" for category
    #         for c2 in range(0, 2):
    #             for c3 in range(0, 2):
    #                 for c4 in range(0, 2):
    #                     for c5 in range(0, 2):
    #                         for c6 in range(0, 2):
    #                             scoreable_categories.append([c1, c2, c3, c4, c5, c6])
    #     return scoreable_categories



