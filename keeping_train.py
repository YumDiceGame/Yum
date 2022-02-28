# Core of the keeping training

from score import *
from utilities import *

def calc_row_index(dice, available_categories, list_all_dice_rolls, list_scoreable_categories):
    # We calculate the row index fairly often ...
    index_dice = list_all_dice_rolls.index(dice)
    index_score = list_scoreable_categories.index(available_categories)
    row_index = index_dice * TWO_TO_NUM_SCORE_CATEGORIES + index_score
    return row_index


class KeepingTrain:

    def __init__(self):
        self.dice = DiceSet()
        self.score = Score()
        self.epsilon_decay_value = 0
        self.learning_rate_decay_value = 0
        self.track_score = True
        self.print = False
        self.trace_reward = False

    def print_cond(self, condition, item):
        if condition:
            print(item)

    def train(self, q_table_scoring, q_table_keeping, list_all_dice_rolls, list_scoreable_categories,
              keeping_actions_masks, action_to_dice_to_keep, learning_rate_init, discount,
              do_epsilon, decrease_learning_rate):

        if do_epsilon:
            epsilon = 1
            self.epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
        else:
            # Set epsilon to 0
            epsilon = 0

        # We are going to decrease the LR:
        # Initial LR is the one passed in as "learning_rate"
        learning_rate = learning_rate_init
        if decrease_learning_rate:
            learning_rate_final = 0.1  # Somewhat arbitrary
            self.learning_rate_decay_value = (learning_rate_init-learning_rate_final) / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
        else:
            self.learning_rate_decay_value = 0

        track_average_score = 0
        track_score_array = np.zeros(6)
        q_table_height = len(q_table_keeping)
        q_table_track_max = np.zeros(q_table_height)
        tracking_score = False  # will become True during the eval window

        # We want to track how many times each row is visited
        q_table_row_visit = np.zeros(q_table_height, dtype=int)
        # And see how often it changed
        q_table_row_max_changes = np.zeros(q_table_height, dtype=int)
        # And the below will say how long it went w/o a change
        q_table_row_max_streak = np.zeros(q_table_height, dtype=int)

        # This is to follow the evolution of the scoring
        filename = f"score_track_progress_episodes_{NUM_EPISODES/1e6}M_LR_{learning_rate_init}_DIS_{discount}.txt"
        with open(filename, "w") as f:
            f.write(f"episode\teps\tlr\tscore\tbonus\tstraight\tfull\tlow\thigh\tyum\n")

        len_all_dice_rolls = len(list_all_dice_rolls)

        for episode in range(NUM_EPISODES + 1):

            turn = 0
            self.score.reset_scores()
            # Starting with "random" score table (unless you're evaluating)
            if not tracking_score:
                num_starting_cats = self.score.seed_score_table()
            all_scored = self.score.all_scored()

            while not all_scored:
                turn += 1
                self.dice.reset()

                # Roll with uniform probability
                # meaning 55555 is equally probable as 11245
                if not tracking_score:
                    self.dice.set(list_all_dice_rolls[np.random.randint(0, len_all_dice_rolls)])
                else:
                    self.dice.roll()

                # max_die_count: it's back
                if not self.score.is_above_the_line_all_scored():
                    max_die_count_previous, face_max_die_count = self.dice.max_die_count_for_available_category(
                        self.score.get_available_cat_vector())

                potential_max_score_previous, potential_max_score_category_previous = \
                    self.score.get_potential_max_score(self.dice)
                # self.print_cond(self.trace_reward, f"initial pot max = {potential_max_score_previous} "
                #                                    f"initial pot max cat = {potential_max_score_category_previous}")

                for roll in range(2, NUM_ROLLS + 1):

                    # STATE DEFINITION --->
                    # The state now is a concatenation of the dice dict and scored cats

                    state_index = calc_row_index(self.dice.dice(), self.score.get_available_cat_vector(),
                                                 list_all_dice_rolls, list_scoreable_categories)

                    # STATE DEFINITION <---

                    # KEEPING ACTION --->
                    if np.random.random() < epsilon:
                        # Get random dice keeping action
                        possible_random_actions = ma.masked_array([*range(0, NUM_KEEPING_ACTIONS)],
                                                                  keeping_actions_masks[self.dice.as_short_string()])
                        masked_random_actions = possible_random_actions[possible_random_actions.mask == False]
                        action = random.choice(masked_random_actions)
                    else:
                        action = (ma.masked_array(q_table_keeping[state_index][0:NUM_KEEPING_ACTIONS],
                                                  keeping_actions_masks[self.dice.as_short_string()])).argmax()

                    self.dice.make_list_reroll_for_selected_die_faces(action_to_dice_to_keep[action])
                    self.dice.roll_list_reroll()


                    # KEEPING ACTION <---

                    # REWARD --->
                    # so here we are going to find out what would be the max potential score across available categories
                    # if we were to score this roll right now.
                    # We will record this quantity as some trailing max potential score
                    #
                    # If initial roll, the reward is whatever that potential max score is
                    # In subsequent rolls, it gets more interesting:
                    # is the potential max score went down, there is a penalty
                    # if it went down by 20 or more, there is a large penalty
                    # if it's stayed the same, a small reward.
                    #   But if the potential max score was large and it stayed large, then we give a large reward
                    # if the potential max score went up there is a reward, and a large one if it went up by 20 or more
                    # So we need a function that will return potential max score

                    if not self.score.is_above_the_line_all_scored():
                        max_die_count, face_max_die_count = self.dice.max_die_count_for_available_category(
                            self.score.get_available_cat_vector())
                    else:  # might remove this else
                        max_die_count = 0

                    potential_max_score, potential_max_score_category = self.score.get_potential_max_score(self.dice)

                    reward = -1

                    if potential_max_score == 50: # Now valuing dice at 10 per
                        reward = 15  # Max!
                    elif potential_max_score == 35:  # full case
                        reward = 12
                    elif 40 <= potential_max_score < 50:
                        reward = 10
                    elif 21 <= potential_max_score <= 30:
                        reward = 7.5
                    elif potential_max_score > potential_max_score_previous:
                        reward = 5
                    elif potential_max_score_previous > potential_max_score:
                        if potential_max_score_previous >= (potential_max_score+25):
                            reward = -20
                        else:
                            reward = -10
                    elif potential_max_score_previous == potential_max_score:
                        reward = -7.5  # was - 5 for the 20M training
                        if potential_max_score_previous == potential_max_score == 0:  # it's worse
                            reward = -25  # This just added after the 20M training


                    # "Heuristics" zone
                    # Correlating rewards with actions ... not ideal but it works
                    # Punish bad keep all actions
                    # the below emulates bad keep all AND bad K4 ... maybe
                    # if (DiceSet.dice_difference(list_of_rolls[roll-1], list_of_rolls[roll-2]) <= 1) and \
                    #         potential_max_score < 20 and potential_max_score_previous < 20:
                    #     reward += -250  # got 9/1775 bad keep alls with this, about the same as when corr. to action
                    # Punish bad keep all actions
                    if self.dice.is_keep_all() and potential_max_score < 21:
                        reward += -250  # was 100
                    if action_to_dice_to_keep[action] == {'K4'}:
                        # Punish bad K4 action (bad meaning Straight isn't avail but we are doing action K4
                        if not self.score.is_category_available('Straight'):
                            reward += -100
                        # Also punish the following case: is your action 60 pair is in an available above the line
                        # category (rationale is you want to try to get more of that die face)
                        mdc = self.dice.max_die_count_for_available_category(self.score.get_available_cat_vector())[0]
                        if mdc >= 2:
                            reward += -50
                    # Punish bad cross-actions
                    if (potential_max_score_category in ABOVE_THE_LINE_CATEGORIES) and \
                            (len(action_to_dice_to_keep[action]) > 1):
                        reward += -100

                    potential_max_score_previous = potential_max_score
                    max_die_count_previous = max_die_count

                    # REWARD <---

                    # NEW STATE --->
                    # nothing special, we'll just get the new state as always
                    # new_state = myDice.get_dict_as_vector() + score.get_available_cat_vector()
                    # get q_table_row number as "state_index"
                    new_state_index = calc_row_index(self.dice.dice(), self.score.get_available_cat_vector(),
                                                     list_all_dice_rolls, list_scoreable_categories)

                    # SCORE CATEGORY --->
                    # Moved to here so as to get some feedback for the learning
                    # Scoring with previously trained score q table
                    if roll == NUM_ROLLS:  # Last roll
                        # with full size table
                        # category_scored = self.score.score_with_q_table(q_table_scoring, new_state_index, self.dice)
                        # with reduced:
                        category_scored = q_table_scoring[new_state_index] + 1
                        self.score.score_a_category(score_int_to_cat(category_scored), self.dice)
                        # cumulative_score += self.score.get_category_score(score_int_to_cat(category_scored))
                        if score_int_to_cat(category_scored) in ABOVE_THE_LINE_CATEGORIES:
                            # penalty for anything less than 5 of a kind
                            penalty = 5 * (NUM_DICE - (self.score.get_category_score(score_int_to_cat(category_scored)) / category_scored))
                            reward -= penalty
                    # SCORE CATEGORY <---

                    # Q UPDATE --->

                    # for tracking, keep the max arg before update:
                    winner_action_pre = q_table_keeping[state_index][0:NUM_KEEPING_ACTIONS].argmax()

                    max_future_q = np.max(q_table_keeping[new_state_index][action])

                    current_q = q_table_keeping[state_index][action]

                    new_q = (1 - learning_rate) * current_q + learning_rate * \
                            (reward + discount * max_future_q)

                    # # Update Q table with new Q value
                    q_table_keeping[state_index][action] = new_q

                    # Indicate you visited this state
                    q_table_row_visit[state_index] += 1
                    if q_table_row_visit[state_index] == 1:  # first time here
                        # set streak to 0
                        q_table_row_max_streak[state_index] = 0
                    # And has the row changed?
                    if winner_action_pre != q_table_keeping[state_index][0:NUM_KEEPING_ACTIONS].argmax():
                        q_table_row_max_changes[state_index] += 1
                        # reset streak
                        q_table_row_max_streak[state_index] = 0
                    else:
                        # start streak
                        q_table_row_max_streak[state_index] += 1
                all_scored = self.score.all_scored()

            # Decaying is being done every episode if episode number is within decaying range
            if do_epsilon:
                if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                    epsilon -= self.epsilon_decay_value
            # Decaying also LR
            if decrease_learning_rate:
                if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:  # same action range as epsilon
                    learning_rate -= self.learning_rate_decay_value

            if episode % NUM_SHOW == 0:
                print(f"episode = {episode} LR = {learning_rate:.4f} DIS = {discount}")
                scorecard_string = self.score.print_scorecard()
                for score_line in scorecard_string:
                    print(score_line)
                print(f"epsilon = {epsilon:.4f}")
                print("\n")

            # Track scoring: how does it evolve over time --->
            if (episode + NUM_GAMES_IN_LINE_EVAL - 1) % NUM_TRACK_SCORE == 0:
                tracking_score = True
                # we are going to set epsilon to 0 for the eval (table actions only matter)
                # store the current epsilon
                epsilon_mem = epsilon
                # print(f"storing epsilon {epsilon_mem} at episode {episode}")
                epsilon = 0  # -> setting eps to 0 temporarily
                track_average_score = 0
                track_score_array = np.zeros(6)
            elif (episode % NUM_TRACK_SCORE == 0) and (episode != 0):
                tracking_score = False
                epsilon = epsilon_mem  # restore epsilon to resume training
                # print(f"restoring epsilon {epsilon} at episode {episode}")
            if self.track_score:
                # accumulation for average score
                track_average_score += (self.score.get_total_score() + self.score.get_bonus())
                # accumulation for tracking above the line items such as straight, full...
                track_score_array = np.add(track_score_array, np.array(self.score.get_above_the_line_success()))

            if episode != 0 and episode % NUM_TRACK_SCORE == 0:
                filename = f"score_track_progress_episodes_{NUM_EPISODES/1e6}M_LR_{learning_rate_init}_DIS_{discount}.txt"
                with open(filename, "a") as f:
                    f.write(f"{episode}\t{epsilon:.2f}\t{learning_rate:.4f}\t{track_average_score / NUM_GAMES_IN_LINE_EVAL}")
                    for item in track_score_array:
                        f.write(f"\t{(item/NUM_GAMES_IN_LINE_EVAL):.2f}")
                    f.write("\n")

            # Track q_table: how does it evolve over time <---
            if episode != 0 and episode % EVAL_Q_TABLE == 0:
                # Evaluate q table
                # compare the new q_table maxes to the previous ones
                # and count the number of differences
                count_diff_maxes = 0
                for q_table_row in range(0, q_table_height):
                    # identify differences
                    if q_table_track_max[q_table_row] != q_table_keeping[q_table_row][0:NUM_KEEPING_ACTIONS].argmax():
                        count_diff_maxes += 1
                    # update q_table_track_max for comparison next EVAL_Q_TABLE
                    q_table_track_max[q_table_row] = q_table_keeping[q_table_row][0:NUM_KEEPING_ACTIONS].argmax()
                print(f"q_table_diff = {count_diff_maxes}\n")
                with open("q_table_track_progress.txt", "a") as f:
                    f.write(f"{episode}\t{count_diff_maxes}\n")

        with open("q_table_row_visit.txt", "w") as f:
            row_number = 0
            for num_row_visit, q_table_row_max_changes, q_table_row_streak in \
                    zip(q_table_row_visit, q_table_row_max_changes, q_table_row_max_streak):
                f.write(f"{row_number}\t{num_row_visit}\t{q_table_row_max_changes}\t{q_table_row_streak}\n")
                row_number += 1


