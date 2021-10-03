# Core of the keeping training

from constants import *
import random
import numpy as np
from dice import *
from score import *


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
        self.do_epsilon = True
        self.epsilon_decay_value = 0
        self.track_score = True
        self.print = False

    def train(self, q_table_scoring, q_table_keeping, list_all_dice_rolls, list_scoreable_categories,
              keeping_actions_masks, action_to_dice_to_keep, learning_rate, discount):

        if self.do_epsilon:
            epsilon = 1
            self.epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
        else:
            # Set epsilon to 0
            epsilon = 0

        track_average_score = 0
        track_score_array = np.zeros(6)

        # This is to follow the evolution of the scoring
        filename = f"score_track_progress_LR_{learning_rate}_DIS_{discount}.txt"
        with open(filename, "w") as f:
            f.write(f"episode\teps\tscore\tbonus\tstraight\tfull\tlow\thigh\tyum\n")

        for episode in range(NUM_EPISODES + 1):

            turn = 0
            self.score.reset_scores()
            all_scored = False

            while not all_scored:
                turn += 1
                self.dice.reset()
                self.dice.roll()
                # max_die_count: it's back
                if not self.score.is_above_the_line_all_scored():
                    max_die_count_previous, face_max_die_count = self.dice.max_die_count_for_available_category(
                        self.score.get_available_cat_vector())

                # for tracking when you have almost straight, full or yum
                almost_straight_list_per_roll = []
                almost_full_list_per_roll = []
                almost_yum_list_per_roll = []

                almost_straight_list_per_roll.append(self.dice.is_almost_straight())
                almost_full_list_per_roll.append(self.dice.is_almost_full())
                almost_yum_list_per_roll.append(self.dice.is_almost_yum())

                potential_max_score_previous = self.score.get_potential_max_score(self.dice)

                for roll in range(2, NUM_ROLLS + 1):

                    # This is the observation part of the state
                    # old style when we were doing only max die count
                    # max_die_count = face_max_die_count = 0
                    # state_max_die_count_and_face = myDice.max_die_count_for_available_category(score.get_available_cat_vector())
                    # state_max_die_count = state_max_die_count_and_face[0]
                    # state_face_max_die_count = state_max_die_count_and_face[1]
                    # max_die_count_alert = False
                    # if PRINT:
                    #     print("state_max_die_count = ", state_max_die_count)
                    #     print("state_face_max_die_count = ", state_face_max_die_count)
                    # if print_record_games:
                    #     game_events_to_record.append(f"state_max_die_count = {state_max_die_count} state_face_max_die_count = {state_face_max_die_count}")

                    # STATE DEFINITION --->
                    # The state now is a concatenation of the dice dict and scored cats
                    # state = myDice.dice() + score.get_available_cat_vector()
                    # Get q_table_row number as "state_index"
                    # segmented lookup
                    # state_index_dice_roll = list_all_dice_rolls.index(self.dice.dice())
                    # state_index_score = list_scoreable_categories.index(self.score.get_available_cat_vector())
                    # state_index = state_index_dice_roll * TWO_TO_NUM_SCORE_CATEGORIES + state_index_score

                    state_index = calc_row_index(self.dice.dice(), self.score.get_available_cat_vector(),
                                                 list_all_dice_rolls, list_scoreable_categories)

                    if self.print:
                        print("state_index = ", state_index)
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

                    # for tracking when you have almost straight full or yum
                    almost_straight_list_per_roll.append(self.dice.is_almost_straight())
                    almost_full_list_per_roll.append(self.dice.is_almost_full())
                    almost_yum_list_per_roll.append(self.dice.is_almost_yum())

                    if self.print and episode % NUM_SHOW == 0:
                        print(f"action mask = {keeping_actions_masks[self.dice.as_short_string()]}")
                        print(
                            f"roll {self.dice.get_num_rolls()} action {action} list re-roll {self.dice.get_list_reroll()}"
                            f" new dice {self.dice} ")

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

                    potential_max_score = self.score.get_potential_max_score(self.dice)
                    #
                    # if PRINT:
                    #     print(f"roll {myDice.get_num_rolls()} mdc = {max_die_count} fmdc = {face_max_die_count}")
                    # shortcut for the reward:
                    # right now it's max_die_count * face_max_die_count
                    # but that will have to be generalized as explained at the top of this comment section
                    # Experiment: normalize to max die count times, say, 3
                    # reward = max_die_count * face_max_die_count
                    # reward = max_die_count * 3 # face_max_die_count
                    # so as before, if max die count went down ...
                    # special_reward = 0
                    # if PRINT:
                    #     print(f"before mdc check  max_die_count_previous = {max_die_count_previous} max_die_count = {max_die_count} ")
                    reward = -1

                    # not bad, but need better
                    if potential_max_score == 50: # Now valuing dice at 10 per
                        reward = 10  # Max!
                    elif 21 <= potential_max_score <= 40:
                        reward = 7.5
                    elif potential_max_score > potential_max_score_previous:
                        reward = 5
                    elif potential_max_score_previous > potential_max_score:
                        reward = -10
                    elif potential_max_score_previous == potential_max_score:
                        reward = -5

                    # if potential_max_score >= 30:  # Now valuing dice at 10 per
                    #     reward += 90
                    # elif 21 <= potential_max_score < 30:
                    #     reward += 75
                    # elif potential_max_score_previous >= (potential_max_score+25):  # if score went down a lot
                    #     reward += -100 # -90
                    # elif potential_max_score_previous == potential_max_score == 0:  # we are stalled
                    #     reward += -80
                    # elif potential_max_score == 0:  # reward = 0 probably you kept all dice wrongly
                    #     reward += -60
                    # elif potential_max_score >= 20:  # could mean 3 out of 5
                    #     reward += 40
                    # else:
                    #     reward = 0

                    # Stuff for encouraging to try straight or full
                    if self.score.is_category_available('Straight') and not self.dice.is_straight() and potential_max_score < 21:
                        if almost_straight_list_per_roll[roll-2] and not almost_straight_list_per_roll[roll-1]:
                            # we got further away from straight
                            reward += -10  # += to hedge your bets was -2.5
                    if self.score.is_category_available('Full') and not self.dice.is_full() and potential_max_score < 21:
                        if almost_full_list_per_roll[roll-2] and not almost_full_list_per_roll[roll-1]:
                            # we got further away from full
                            reward += -15  # += to hedge your bets was -3.5

                    # if score.is_category_available('Yum'):
                    #     if myDice.is_yum():
                    #         pass
                    #     elif almost_yum_list_per_roll[roll-2] and not almost_yum_list_per_roll[roll-1]:
                    #         # we got further away from yum
                    #         reward -= 40

                    # if not score.is_above_the_line_all_scored():  # MDC only if anything left above the line:
                    #     if max_die_count_previous > max_die_count:
                    #         reward += -60
                    #     elif max_die_count > max_die_count_previous:
                    #         reward += 40

                    potential_max_score_previous = potential_max_score
                    max_die_count_previous = max_die_count

                    # REWARD <---

                    # NEW STATE --->
                    # nothing special, we'll just get the new state as always
                    # new_state = myDice.get_dict_as_vector() + score.get_available_cat_vector()
                    # get q_table_row number as "state_index"
                    # new_state_index_dice = list_all_dice_rolls.index(myDice.dice())
                    # new_state_index_score = list_scoreable_categories.index(self.score.get_available_cat_vector())
                    # new_state_index = new_state_index_dice * TWO_TO_NUM_SCORE_CATEGORIES + new_state_index_score
                    new_state_index = calc_row_index(self.dice.dice(), self.score.get_available_cat_vector(),
                                                     list_all_dice_rolls, list_scoreable_categories)

                    # if PRINT:
                    #     print("new_state_index_dice = ", new_state_index_dice)
                    #     print("new_state_index_score = ", new_state_index_score)
                    #     print("new_state_index = ", new_state_index)
                    # NEW STATE <---

                    # Q UPDATE --->
                    # should be similar to:
                    max_future_q = np.max(q_table_keeping[new_state_index][action])
                    # if PRINT:
                    #     print("max future q = ", max_future_q)
                    #
                    # # Current Q value (for current state and performed action)
                    current_q = q_table_keeping[state_index][action]
                    # if PRINT:
                    #     print("current q = ", current_q)
                    #
                    # # And here's our equation for a new Q value for current state and dice_action
                    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
                    #
                    # # Update Q table with new Q value
                    q_table_keeping[state_index][action] = new_q
                    # if PRINT:
                    #     print("new_q = ", q_table_keeping[state_index][action])

                    # print(f"Episode {episode} Turn {turn} roll {roll}time roll {time.time() - time_start_roll}")

                # SCORE CATEGORY --->
                # Scoring with previously trained score q table
                # need to compute q_table_score index
                # q_table_scoring_index_dice = list_all_dice_rolls.index(myDice.dice())
                # q_table_scoring_index_score = list_scoreable_categories.index(score.get_available_cat_vector())
                # q_table_scoring_index = q_table_scoring_index_dice * TWO_TO_NUM_SCORE_CATEGORIES + q_table_scoring_index_score
                q_table_scoring_index = calc_row_index(self.dice.dice(), self.score.get_available_cat_vector(),
                                                       list_all_dice_rolls, list_scoreable_categories)
                category_scored = self.score.score_with_q_table(q_table_scoring, q_table_scoring_index, self.dice)

                # SCORE CATEGORY <---

                all_scored = self.score.all_scored()

            # Decaying is being done every episode if episode number is within decaying range
            if self.do_epsilon:
                if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                    epsilon -= self.epsilon_decay_value

            if episode % NUM_SHOW == 0:
                print(f"episode = {episode} LR = {learning_rate} DIS = {discount}")
                print("score = ", self.score.get_total_score() + self.score.get_bonus())
                print("bonus = ", self.score.get_bonus())
                scorecard_string = self.score.print_scorecard()
                for score_line in scorecard_string:
                    print(score_line)

                print("epsilon = ", epsilon)
                print("\n")

            # Track scoring: how does it evolve over time --->
            if (episode + NUM_GAMES_IN_LINE_EVAL) % NUM_TRACK_SCORE == 0:
                track_score = True
                # we are going to set epsilon to 0 for the eval (table actions only matter)
                # store the current epsilon
                epsilon_mem = epsilon
                # print(f"storing epsilon {epsilon_mem} at episode {episode}")
                epsilon = 0  # -> setting eps to 0 temporarily
                track_average_score = 0
                track_score_array = np.zeros(6)
            elif (episode % NUM_TRACK_SCORE == 0) and (episode != 0):
                track_score = False
                epsilon = epsilon_mem  # restore epsilon to resume training
                # print(f"restoring epsilon {epsilon} at episode {episode}")
            if self.track_score:
                # accumulation for average score
                track_average_score += (self.score.get_total_score() + self.score.get_bonus())
                # accumulation for tracking above the line items such as straight, full...
                track_score_array = np.add(track_score_array, np.array(self.score.get_above_the_line_success()))

            if episode != 0 and episode % NUM_TRACK_SCORE == 0:
                filename = f"score_track_progress_LR_{learning_rate}_DIS_{discount}.txt"
                with open(filename, "a") as f:
                    f.write(f"{episode}\t{epsilon:.2f}\t{track_average_score / NUM_GAMES_IN_LINE_EVAL}")
                    for item in track_score_array:
                        f.write(f"\t{(item/NUM_GAMES_IN_LINE_EVAL):.2f}")
                    f.write("\n")
            # Track scoring: how does it evolve over time <---

            # if episode != 0 and episode % EVAL_Q_TABLE == 0:
            #     # Evaluate q table
            #     # compare the new q_table maxes to the previous ones
            #     # and count the number of differences
            #     count_diff_maxes = 0
            #     for q_table_row in range (0, q_table_height):
            #         # identify differences
            #         if q_table_track_max[q_table_row] != q_table_keeping[q_table_row][0:NUM_KEEPING_ACTIONS].argmax():
            #             count_diff_maxes += 1
            #         # update q_table_track_max for comparison next EVAL_Q_TABLE
            #         q_table_track_max[q_table_row] = q_table_keeping[q_table_row][0:NUM_KEEPING_ACTIONS].argmax()
            #     print(f"q_table_diff = {count_diff_maxes}\n")
            #     with open("q_table_track_progress.txt", "a") as f:
            #         f.write(f"{episode}\t{count_diff_maxes}\n")
