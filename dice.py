# Diffed with across_line_old 7/22/21
# 9/5: just for training with Git branches

import collections
import numpy as np
from constants import *

class DiceSet:
    ''' Class representing an ordered collection of dice. When unrolled the dice
    values are set to None.
    '''

    def __init__(self, numDice=NUM_DICE, dice=None):

        self._num_dice = numDice
        self._dice = [None for i in range(self._num_dice)] if dice is None else dice
        self._num_rolls = 0
        self._dict = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
        self._masked_dict = {}  # for masking with available categories
        self._list_reroll = [True] * NUM_DICE

    def num_dice(self):

        return self._num_dice

    def num_rolls(self):

        return self._num_rolls

    def dice(self):

        return self._dice

    # reset num_rolls to 0 and all dice to None
    def reset(self):

        self._dice = [None for i in range(self._num_dice)]
        self._num_rolls = 0

    # Roll all dice at the indices specified and increment our num_rolls count.
    # If no indices are specified re-roll all dice.
    def roll(self, indices=None):

        if indices is None:
            indices = [True] * NUM_DICE # range(self._num_dice)

        j = 0
        for i in indices:
            if (i == True) or (self._dice[j] is None):
                self._dice[j] = np.random.randint(1, NUM_DIE_FACES+1) # random.randint(1, 6)
            j += 1

        self._dice = sorted(self._dice)
        self._num_rolls += 1
        self.as_dict()  # So caller doesn't need to worry

    def roll_Yum(self, indices=None):
        # roll random Yum

        if indices is None:
            indices = [True] * NUM_DICE # range(self._num_dice)

        j = 0
        for i in indices:
            if (i == True) or (self._dice[j] is None):
                self._dice[j] = np.random.randint(1, NUM_DIE_FACES+1) # random.randint(1, 6)
            j += 1

        self._dice = sorted(self._dice)
        self._num_rolls += 1
        self.as_dict()  # So caller doesn't need to worry

    def roll_Straight(self):
        # generate a straight
        flip_a_coin = np.random.randint(0, 2)
        if flip_a_coin == 0:
            self._dice = [1, 2, 3 ,4 ,5]
        else:
            self._dice = [2, 3, 4, 5, 6]
        # already sorted
        self.as_dict()

    def roll_Full(self):
        # generate a full house
        rand_1 = np.random.randint(1, NUM_DIE_FACES+1)
        rand_2 = np.random.randint(1, NUM_DIE_FACES+1)
        self._dice = [rand_1, rand_1, rand_1, rand_2, rand_2]
        # if Yum gets generated instead, no big deal
        self._dice = sorted(self._dice)
        self.as_dict()

    def roll_Heavy(self):
        # slants the roll towards high values
        for i in range(0, NUM_DICE):
            rand_num = np.random.randint(1, 10)
            if 1 <= rand_num < 2:
                die = 1
            elif 2 <= rand_num < 3:
                die = 2
            elif 3 <= rand_num < 4:
                die = 3
            elif 4 <= rand_num < 6:
                die = 4
            elif 6 <= rand_num < 8:
                die = 5
            elif 8 <= rand_num < 10:
                die = 6
            self._dice[i] = die
        self._dice = sorted(self._dice)
        self.as_dict()

    def roll_list_reroll(self):  # uses list reroll to mask
        j = 0
        for i in self._list_reroll:
            if (i == True) or (self._dice[j] is None):
                self._dice[j] = np.random.randint(1, NUM_DIE_FACES+1) # random.randint(1, 6)
            j += 1

        self._dice = sorted(self._dice)
        self._num_rolls += 1
        self.as_dict()  # So caller doesn't need to worry

    def dice_to_keep_list(self, die_face):

        for k in range(0, NUM_DICE):
            if self.get_die_at_pos(k) == int(die_face):
                self._list_reroll[k] = False
            else:
                self._list_reroll[k] = True
        return

    def as_set(self):

        return set(self._dice)

    def as_count(self):

        return sorted(collections.Counter(self._dice))  # if len ==1 you have Yum

    def as_dict(self):

        self._dict = {}
        for i in range(0, NUM_DIE_FACES):
            self._dict[str(i+1)] = self._dice.count(i+1)
        return self._dict

    def get_dict(self):

        # Call "as_dict" first
        self.as_dict()
        return self._dict

    def get_dict_as_vector(self):

        # Returns the values in order
        # I mean how many 1's, how many two's ...
        # in the same order as the dict
        dict_as_vector = []
        for die in self._dict:
            dict_as_vector.append(self._dict[die])
        return dict_as_vector

    def __str__(self):

        return str(self._dice).strip('[').strip(']')

    def as_short_string(self):
        # str like 12235
        # in case not already sorted
        # self._dice.sort() actually it is
        short_string = ""
        for d in self._dice:
            short_string += str(d)
        return short_string

    def set(self, _dice):

        self._dice = _dice
        self.as_dict()

    def get_die_at_pos(self, pos):

        return self._dice[pos]

    def get_num_rolls(self):

        return self._num_rolls

    def get_list_reroll(self):

        return self._list_reroll

    def set_list_reroll(self, list_reroll):

        self._list_reroll = list_reroll

    def reset_num_rolls(self):

        self._num_rolls = 0

        return

    def sum(self):

        return sum(self._dice)

    def is_yum(self):

        return len(collections.Counter(self._dice)) == 1

    def is_almost_yum(self):
        self.as_dict()
        return 4 in self._dict.values()

    def is_straight(self):

        return self._dice == [1, 2, 3, 4, 5] or self._dice == [2, 3, 4, 5, 6]

    def is_all_singletons(self):

        return len(collections.Counter(self._dice)) == NUM_DICE

    def is_full(self):
        self.as_dict()
        return (2 in self._dict.values()) and (3 in self._dict.values())

    def find_one_pair_face(self):
        '''
        Find the face of the one pair in the hand
        intended for use when we know we have one pair and three singletons
        :return: the face of the pair
        '''
        self.as_dict()
        for die_face, count in self._dict.items():
            if count == 2:  # 2 for the pair
                return die_face

    def find_face_not_two_pair(self):
        '''
        Find the face of the singleton die in the hand
        intended for use when we know we have two pairs and one singleton
        :return: the face of the singleton
        '''
        self.as_dict()
        for die_face, count in self._dict.items():
            if count == 1:  # 12 for the pair
                return die_face

    def is_almost_straight(self):
        '''
        Are we within 1 die of a straight
        '''
        almost_straight = True
        one_chance = False  # it can be more that one of diff only once
        if self.is_straight():
            almost_straight = False
        else:
            if len(collections.Counter(self._dice)) < 4:
                almost_straight = False
            else:
                elements = list(self.as_set())
                # check that we are 1 die away from a straight
                for i in range(len(elements) - 1):
                    if elements[i + 1] - elements[i] > 2:
                        # we are not ... so mask at 1
                        almost_straight = False
                    elif elements[i + 1] - elements[i] > 1:
                        if one_chance:
                            almost_straight = False
                        one_chance = True
        return almost_straight

    def is_two_pairs(self):

        cnt_pairs = 0
        for die in self._dict:
            if self._dict[str(die)] == 2:
                cnt_pairs += 1
        return cnt_pairs == 2

    def is_two_pairs_masked(self):

        cnt_pairs = 0
        for die in self._masked_dict:
            if self._masked_dict[str(die)] == 2:
                cnt_pairs += 1
        return cnt_pairs == 2

    def two_pairs_choose(self):

        list_two_faces = []

        # find which two faces
        for i in self._masked_dict:
            if self._masked_dict[str(i)] == 2:
                #  list_two_faces.append(self._dict[str(i)].k)
                list_two_faces.append(int(i))
        return list_two_faces

    def reset_list_reroll(self):

        self._list_reroll = [True] * NUM_DICE

    def flip_list_reroll(self):

        self._list_reroll = [not element for element in self._list_reroll]

    def make_list_reroll_for_selected_die_face(self, die_face):

        for k in range(0, NUM_DICE):
            if self.get_die_at_pos(k) == int(die_face):
                self._list_reroll[k] = False
            else:
                self._list_reroll[k] = True
        return

    def make_list_reroll_for_selected_die_faces(self, die_faces):
        # similar to make_list_reroll_for_selected_die_face
        # except generalized for multiple die faces

        if die_faces == {'K4'}:  # This is the 'K4' case in which I want to keep 4 singletons
            # we are keeping all dice except one half of the pair
            one_pair_face = self.find_one_pair_face()
            found_half_pair = False
            self._list_reroll = [False] * NUM_DICE
            for k in range(0, NUM_DICE):
                if str(self.get_die_at_pos(k)) == one_pair_face:
                    if not found_half_pair:
                        self._list_reroll[k] = True
                        found_half_pair = True
        else:
            for k in range(0, NUM_DICE):
                if self.get_die_at_pos(k) in die_faces:
                    self._list_reroll[k] = False
                else:
                    self._list_reroll[k] = True
        return

    def max_die_count(self):

        return int(max(self._dict.values()))

    def max_die_count_for_available_category(self, avail_cat_vector):
        # Like max_die_count, but gated by available_category_vector

        # Special case if Yum is our only category left:
        only_yum_left = (sum(avail_cat_vector) == NUM_SCORE_CATEGORIES) & (avail_cat_vector[YUM_COL] == 0)
        # Must check Yum status
        if self.is_yum():
            max_die_count_local = NUM_DICE
            # Compact construct below, safe for use when we know that there are no two maxes
            max_die_face_local = int(max(self._dict, key=self._dict.get))

        else:
            # Special case if only category left is Yum
            if only_yum_left:
                max_die_face_local = self.face_max_die_count()
                max_die_count_local = self.max_die_count()
            else:
                # Reset masked dictionary:
                self._masked_dict = {}
                # need to choke down the avail cat vector to above the line
                avail_cat_vector_above_line = []
                for j in range(NUM_SCORE_CAT_ABOVE_LINE):
                    avail_cat_vector_above_line.append(avail_cat_vector[j])
                i = 0
                for die in self._dict:
                    if not avail_cat_vector_above_line[i]:
                        self._masked_dict[die] = self._dict[die]
                    i += 1
                # print("avail_cat_vector = ", avail_cat_vector_above_line)
                max_die_count_local = int(max(self._masked_dict.values()))

                # Let's also return the face in the same function
                # Copied from face_max_die_count
                max_die_face_local = 0
                # to avoid learning biasing
                # if max_die_count is one, return a random choice of the held die faces
                if max_die_count_local == 1:
                    avail_die_faces = []
                    # grab the available die faces:
                    for die in self._masked_dict:
                        if self._masked_dict[str(die)] == 1:
                            avail_die_faces.append(die)
                    max_die_face_local = int(avail_die_faces[(np.random.randint(0, len(avail_die_faces)))])

                # also for the case you have two pairs, you have to return one of the two faces
                # selected at random
                elif self.is_two_pairs_masked():
                    max_die_face_local = self.two_pairs_choose()[int(np.random.randint(0, 2))]

                else:
                    # Not singletons, not two pairs
                    for i in self._masked_dict:
                        if self._masked_dict[str(i)] == max_die_count_local:
                            max_die_face_local = int(i)

        return [max_die_count_local, max_die_face_local]

    def face_max_die_count(self):

        self._masked_dict = self._dict  # hack here for two pair choose in Yum case
        max_die_face = 0

        # to avoid learning biasing
        # if max_die_count is one, return a random choice of the held die faces
        if self.max_die_count() == 1:
            max_die_face = self._dice[int(np.random.randint(0, NUM_DICE))]
        # also for the case you have two pairs, you have to return on of the two faces
        # selected at random
        elif self.is_two_pairs():
            max_die_face = self.two_pairs_choose()[int(np.random.randint(0, 2))]
        else:
            # Not singletons, not two pairs
            for i in self._dict:
                if self._dict[str(i)] == self.max_die_count():
                    max_die_face = int(i)

        return max_die_face

    def get_potential_max_score(self, list_avail_categories):
        # Based on the dice in hand and the available categories (parameter)
        # Return the max potential score and category
        # it can return [0, 0] depending on available categories

        potential_max_score = 0
        potential_max_score_die_face = 0
        i = 0
        for die_face in self._dict:
            if not list_avail_categories[i]:
                if potential_max_score < int(die_face) * self._dict[die_face]:
                    potential_max_score = int(die_face) * self._dict[die_face]
                    potential_max_score_die_face = int(die_face)
            i = i+1
        return [potential_max_score, potential_max_score_die_face]

    def get_possible_max_score(self, list_avail_categories):
        # Based on the dice in hand and the available categories (parameter)
        # Return the max the score that is possible to record
        # in the case of a zero score, it will return the lowest category
        # WARNING: this chokes is there are no available categories
        # First find out if the potential max score is valid, i.e. not [0, 0]
        possible_max_score = self.get_potential_max_score(list_avail_categories)
        if possible_max_score[1] == 0:
            # No potential max score
            # Find the lowest avail category
            category_index_found = False
            i = 0
            while not category_index_found:
                if list_avail_categories[i] == 0:
                    category_index = i+1
                    category_index_found = True
                i += 1
            possible_max_score = [0, category_index]
        return possible_max_score

