from constants import *
from dice import DiceSet
from numpy import ma

class Score:
    ''' For scoring categories
     '''

    def __init__(self):
        self._num_score_categories = NUM_SCORE_CATEGORIES
        # _scores is a dict: Key is the category, Value is a list: [scored T/F and actual score]
        self._scores = self.reset_scores()
        self._total_score = 0
        self._re_score_alert = 0

    def reset_scores(self):
        # 'False' means that category has not been scored, thus it is available
        # For now training only above the line plus Yum and straight
        self._scores = {'Ones': [False, 0], 'Twos': [False, 0], 'Threes': [False, 0], 'Fours': [False, 0],
                        'Fives': [False, 0], 'Sixes': [False, 0], 'Yum': [False, 0], 'Straight': [False, 0],
                        'Full': [False, 0], 'Low': [False, 0], 'High': [False, 0]}
        self._re_score_alert = 0
        return self._scores

    def get_score_dict(self):

        return self._scores

    def get_total_score(self):

        self._total_score = 0
        for category in self._scores:
            self._total_score += self._scores[category][1]

        return self._total_score

    def get_above_the_line_success(self):
        '''
        return a vector to help the score tracking in keeping_learning
        "1" means the above line category was scored well
        Order is bonus, straight, full, low, high, yum
        '''
        success_scores = []
        if self.get_bonus() != 0:
            success_scores.append(1)
        else:
            success_scores.append(0)
        if self._scores['Straight'][1] != 0:
            success_scores.append(1)
        else:
            success_scores.append(0)
        if self._scores['Full'][1] != 0:
            success_scores.append(1)
        else:
            success_scores.append(0)
        if self._scores['Low'][1] != 0:
            success_scores.append(1)
        else:
            success_scores.append(0)
        if self._scores['High'][1] != 0:
            success_scores.append(1)
        else:
            success_scores.append(0)
        if self._scores['Yum'][1] != 0:
            success_scores.append(1)
        else:
            success_scores.append(0)

        return success_scores

    def get_bonus(self):
        # calculates total score above the line
        # return 25 if that score is 63 or more, 0 otherwise
        bonus = 0
        above_the_line_sum = self._scores['Ones'][1] + self._scores['Twos'][1] + self._scores['Threes'][1] \
                             + self._scores['Fours'][1] + self._scores['Fives'][1] + self._scores['Sixes'][1]
        if above_the_line_sum >= 63:
            bonus = 25
        return bonus

    def print_scorecard(self):
        # actually creates a string that the caller will have to print
        scorecard_string = []
        for score_key in self._scores:
            if self._scores[score_key][0] and self._scores[score_key][1] == 0:
                scorecard_string.append(f"{score_key} score X")
            else:
                scorecard_string.append(f"{score_key} score {self._scores[score_key][1]}")
        scorecard_string.append(f"Bonus {self.get_bonus()}")
        scorecard_string.append(f"Total score {self.get_total_score() + self.get_bonus()}")
        return scorecard_string

    def all_scored(self):

        all_cat_scored = True
        for score in self._scores:
            all_cat_scored &= self._scores[score][0]

        return all_cat_scored

    def get_potential_max_score(self, dice):
        # given avail categories, what is the best we can get
        potential_score = [0]
        for category in self._scores:
            # if not self._scores[category][0]:  # Category available
            if self.is_category_available(category):  # Category available
                if category == 'Yum':
                    if dice.is_yum(): potential_score.append(30)
                elif category == 'Straight':
                    if dice.is_straight(): potential_score.append(25)
                elif category == 'Full':
                    if dice.is_full(): potential_score.append(25)
                elif category == 'High' or category == 'Low':
                    potential_score.append(self.compute_score_hi_lo(category, dice))
                else:  # 1's thru 6's
                    # commented out below the plain score
                    # potential_score.append(score_cat_to_int(category) * dice.get_dict()[str(score_cat_to_int(category))])
                    # we want to a bit normalize wrt dice usage for the above the line categories
                    # boost it a bit -> 6x instead of 5x
                    potential_score.append(10 * dice.get_dict()[str(score_cat_to_int(category))])
        # print("pot max = ", max(potential_score))
        return max(potential_score)

    def assess_lo_hi_score(self, score_amount, category_scored):
        '''
        returns a "critique" in the form of a reward of the lo/hi scoring
        this should only be called if a non zero score was made
        '''
        assert score_amount >= 21  # score doesn't qualify
        reward = 0
        # Below: scoring too high of a low is bad.  You want a low that won't block your high
        if category_scored == 'Low':
            if self.is_category_available('High'):  # Must leave space for the high
                if 21 <= score_amount < 24:
                    reward = 40
                elif score_amount == 25:
                    reward = 15
                elif score_amount >= 26:
                    reward = -25
            else:  # high is scored, can do what you want
                reward = 40
        # Below: scoring too low of a high is bad.  You want a high that won't block your low
        if category_scored == 'High':
            if self.is_category_available('Low'):
                if 27 <= score_amount <= 30:
                    reward = 40
                elif score_amount == 26:
                    reward = 20
                elif score_amount == 25:
                    reward = 15
                elif score_amount < 25:
                    reward = -25
            else:
                reward = 40
        return reward

    def compute_score_hi_lo(self, category, dice):
        # computes the score for hi/lo with validation
        score_hi_lo = 0
        if category == 'Low':
            # If High has been scored, check if Low lower than High
            score_low = dice.sum()
            if score_low >= 21:  # has to be 21 at least
                if self._scores['High'][0] and self._scores['High'][1] != 0:  # High has been scored and not scratched!
                    if score_low < self._scores['High'][1]:  #
                        score_hi_lo = score_low
                    else:
                        score_hi_lo = 0
                else:
                    score_hi_lo = score_low
            else:
                score_hi_lo = 0
        elif category == 'High':
            # If Low has been scored, check if High higher than Low
            score_high = dice.sum()
            if score_high >= 22:  # has to be 22 at least
                if self._scores['Low'][0] and self._scores['Low'][1] != 0:  # Low has been scored and not scratched
                    if score_high > self._scores['Low'][1]:                 # (would work w/o that and)
                        score_hi_lo = score_high
                    else:
                        score_hi_lo = 0
                else:
                    score_hi_lo = score_high
            else:
                score_hi_lo = 0
        return score_hi_lo

    def score_a_category(self, category, dice):

        assert(self._scores[category][0] == False)  # Can't re-score category
        cat_score = 0

        # do the yum case fist
        if category == 'Yum':
            if dice.is_yum():
                self._scores[category][1] = 30
            else:
                self._scores[category][1] = 0
        elif category == 'Straight':
            if dice.is_straight():
                self._scores[category][1] = 25
            else:
                self._scores[category][1] = 0
        elif category == 'Full':
            if dice.is_full():
                self._scores[category][1] = 25
            else:
                self._scores[category][1] = 0
        # Try new more efficient for Hi and Lo categories
        elif category == 'Low' or category == 'High':
            self._scores[category][1] = self.compute_score_hi_lo(category, dice)

        else:  # these are the 1's thru 6's categories
            dice_dict = dice.get_dict()
            key = str(score_cat_to_int(category))
            self._scores[category][1] = score_cat_to_int(category) * dice_dict[str(score_cat_to_int(category))]
            # for die in dice_list:
            #     if die == score_cat_to_int(category):
            #         cat_score += score_cat_to_int(category)
            # self._scores[category][1] = cat_score
        self._scores[category][0] = True  # Category scored

        return

    def get_category_score(self, category):

        return self._scores[category][1]

    def is_category_available(self, category_name):

        return not self._scores[category_name][0]

    def print_available_cats(self):
        list_avail_cats = []
        for cat in self._scores:
            if not self._scores[cat][0]:
                list_avail_cats.append(score_cat_to_char(cat))
        print(list_avail_cats)

    def available_cat(self):
        available_categories = []
        for cat in self._scores:
            if not self._scores[cat][0]:
                available_categories.append(cat)
        return available_categories

    def get_available_cat_vector(self):
        available_categories_vector = []
        for cat in self._scores:
            if not self._scores[cat][0]:
                # This appends a "0" meaning available
                available_categories_vector.append(0)
            else:
                # This returns a "1" meaning scored (not available)
                available_categories_vector.append(1)
        return available_categories_vector

    def is_above_the_line_all_scored(self):
        available_categories_vector_above_line = self.get_available_cat_vector()[0:NUM_SCORE_CAT_ABOVE_LINE]
        # print("avail = ", available_categories_vector_above_line)
        # print("sum = ", sum(available_categories_vector_above_line))
        return (sum(available_categories_vector_above_line) == NUM_SCORE_CAT_ABOVE_LINE)

    # Ok so we have been struggling to return the action corresponding to the max q
    # When categories start becoming unavailable.
    # BTW "q_vector" is that portion of the q_table in the score section
    # Check it out:
    def max_action_q_table(self, q_vector):
        remaining_q_actions = {}
        i = 0
        for cat in self._scores:
            if not self._scores[cat][0]: # if category avail
                remaining_q_actions[cat] = q_vector[i]
            i += 1
        return score_cat_to_int(max(remaining_q_actions, key=remaining_q_actions.get))

    def get_score_status(self, cat_name):
        # for the obs space: need to know if current category has been scored
        # and if yum has been scored
        # see more details at the top of "main"
        score_status = 0
        if self._scores[cat_name][0]:
            score_status = 1
            # If Yum scored bundle this state of fact in score status
        if self._scores['Yum'][0]:
            score_status += 2

        return score_status

    def score_with_q_table(self, q_table_scoring, q_table_scoring_index, dice):
        # Uses prior built q table to score dice in non-masked category
        # Caller needs to open table
        #
        action = (ma.masked_array(q_table_scoring[q_table_scoring_index][0:NUM_SCORE_CATEGORIES],
                                  self.get_available_cat_vector())).argmax()
        # Score away
        self.score_a_category(score_int_to_cat(action+1), dice)

        return action+1  # Info for caller: what category

    def scratch_penalty(self, scratched_cat):
        '''
        Trying to impose an order in which you will scratch categories
        for example, probably Yum should always be scratched first.
        Also, don't scratch the Full too soon because you might get it later
        "scratched_cat" is the category that has just been scored, or scratched I should say
        Also, caller calls this function only when a zero has just been scored
        '''
        assert self.get_category_score(scratched_cat) == 0 and \
               not self.is_category_available(scratched_cat)  # category not scratched!

        penalty = 0
        if scratched_cat == 'Yum' or scratched_cat == 'Ones':
            penalty = 35
        elif scratched_cat == 'Straight' or scratched_cat == 'Twos':
            penalty = 40
        elif scratched_cat == 'High':
            penalty = 45
        elif scratched_cat == 'Full' or scratched_cat == 'Low':
            penalty = 50
        elif scratched_cat == 'Threes' or scratched_cat == 'Fours':
            penalty = 60
        elif scratched_cat == 'Fives' or scratched_cat == 'Sixes':
            penalty = 75
        return penalty


def score_cat_to_int(category_name):

    cat_to_int_value = None
    if category_name == 'Ones':
        cat_to_int_value = 1
    elif category_name == 'Twos':
        cat_to_int_value = 2
    elif category_name == 'Threes':
        cat_to_int_value = 3
    elif category_name == 'Fours':
        cat_to_int_value = 4
    elif category_name == 'Fives':
        cat_to_int_value = 5
    elif category_name == 'Sixes':
        cat_to_int_value = 6
    elif category_name == 'Yum':
        cat_to_int_value = 7
    elif category_name == 'Straight':
        cat_to_int_value = 8
    elif category_name == 'Full':
        cat_to_int_value = 9
    elif category_name == 'Low':
        cat_to_int_value = 10
    elif category_name == 'High':
        cat_to_int_value = 11

    return cat_to_int_value


def score_int_to_cat(category_number):

    category_name = ""

    if category_number == 1:
        category_name = 'Ones'
    elif category_number == 2:
        category_name = 'Twos'
    elif category_number == 3:
        category_name = 'Threes'
    elif category_number == 4:
        category_name = 'Fours'
    elif category_number == 5:
        category_name = 'Fives'
    elif category_number == 6:
        category_name = 'Sixes'
    elif category_number == 7:
        category_name = 'Yum'
    elif category_number == 8:
        category_name = 'Straight'
    elif category_number == 9:
        category_name = 'Full'
    elif category_number == 10:
        category_name = 'Low'
    elif category_number == 11:
        category_name = 'High'
    return category_name

def score_char_to_cat(category_char):

    if category_char == '1':
        category_name = 'Ones'
    elif category_char == '2':
        category_name = 'Twos'
    elif category_char == '3':
        category_name = 'Threes'
    elif category_char == '4':
        category_name = 'Fours'
    elif category_char == '5':
        category_name = 'Fives'
    elif category_char == '6':
        category_name = 'Sixes'
    elif category_char == 'Y' or category_char == 'y':
        category_name = 'Yum'
    elif category_char == 'S' or category_char == 's':
        category_name = 'Straight'
    elif category_char == 'F' or category_char == 'f':
        category_name = 'Full'
    elif category_char == 'L' or category_char == 'l':
        category_name = 'Low'
    elif category_char == 'H' or category_char == 'h':
        category_name = 'High'
    return category_name

def score_cat_to_char(category_name):
    score_char = None
    if category_name == 'Ones':
        score_char = '1'
    elif category_name == 'Twos':
        score_char = '2'
    elif category_name == 'Threes':
        score_char = '3'
    elif category_name == 'Fours':
        score_char = '4'
    elif category_name == 'Fives':
        score_char = '5'
    elif category_name == 'Sixes':
        score_char = '6'
    elif category_name == 'Yum':
        score_char = 'Y'
    elif category_name == 'Straight':
        score_char = 'S'
    elif category_name == 'Full':
        score_char = 'F'
    elif category_name == 'Low':
        score_char = 'L'
    elif category_name == 'High':
        score_char = 'H'
    return score_char

def get_perfect_score(category_name):

    perfect_score = 0
    if category_name == 'Ones':
        perfect_score = 5
    elif category_name == 'Twos':
        perfect_score = 10
    elif category_name == 'Threes':
        perfect_score = 15
    elif category_name == 'Fours':
        perfect_score = 20
    elif category_name == 'Fives':
        perfect_score = 25
    elif category_name == 'Sixes':
        perfect_score = 30
    elif category_name == 'Yum':
        perfect_score = 30
    elif category_name == 'Straight':
        perfect_score = 25
    elif category_name == 'Full':
        perfect_score = 25

    return perfect_score


# function to translate number to bits
# useful for the q_table_tracking mask
# this is sort of related to score because the input is a q table row number and we want to know the score status
# and use that as a mask
def number_to_bits_vector(input_number):
    as_bits = [0] * NUM_SCORE_CATEGORIES  # Meant for 11 bits
    bit_pos = NUM_SCORE_CATEGORIES-1
    while input_number > 0:
        remainder = input_number % 2
        as_bits[bit_pos] = remainder
        input_number = input_number // 2
        bit_pos -= 1
    return as_bits
