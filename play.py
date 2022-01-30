# Playing Yum game
# with Q learning

# to avoid error TERM environment not set:
# check "Emulate terminal in output console" in
# in Run/Debug configurations for "play"
# https://stackoverflow.com/questions/38867520/term-environment-variable-not-set-on-python-code

from score import *
from constants import *
import pickle
import numpy as np
from numpy import ma
from do_q_table import do_q_table_rows
from do_keep_action_q_table import action_q_table
import matplotlib.pyplot as plt
import os

def french(category):
    if category == 'Ones':
        french_translation = 'Un'
    if category == 'Twos':
        french_translation = 'Deux'
    if category == 'Threes':
        french_translation = 'Trois'
    if category == 'Fours':
        french_translation = 'Quatre'
    if category == 'Fives':
        french_translation = 'Cinq'
    if category == 'Sixes':
        french_translation = 'Six'
    if category == 'Yum':
        french_translation = 'Yum'
    if category == 'Straight':
        french_translation = 'Séquence'
    if category == 'Full':
        french_translation = 'Full'
    if category == 'Low':
        french_translation = 'Bas'
    if category == 'High':
        french_translation = 'Haut'
    return french_translation


# in "play", using the biased to full scoring table
with open("q_table_scoring_reduced.pickle", "rb") as score_q_table_file:
    q_table_scoring = pickle.load(score_q_table_file)

with open("q_table_keeping_reduced.pickle", "rb") as keeping_q_table_file:
    q_table_keeping = pickle.load(keeping_q_table_file)

# Create q_table_scoring_rows
q_table_scoring_rows = do_q_table_rows()

# Action q table:
action_table = action_q_table()

# Map action to dice to keep and action masks
list_set_keep_actions, keep_action_mask_dict = action_table.print_all_action_q_table()


# For saving games
game_events_to_record = []
print_record_games = True
narrate_games = False
narrate_french = True
roll_seq = False

myDice = DiceSet()
score = Score()

scores = []
bonus_cnt = 0
yum_counter = 0
straight_counter = 0
full_counter = 0
lo_counter = 0
hi_counter = 0
bad_action_K4_counter = 0
empty_set = set()
score_straight_fails = 0
score_full_fails = 0

# Interested in histogram of actions
action_histo = np.zeros(len(list_set_keep_actions))
total_actions = 0

all_scored = False

for game_number in range(NUM_GAMES):

    if print_record_games:
        game_events_to_record.append(f"Game number {game_number+1}\n")

    turn_done = False
    turn = 0

    score.reset_scores()
    all_scored = False

    if narrate_games:
        os.system('clear')

    while not all_scored:

        turn += 1
        myDice.reset()

        straight_detected = False
        full_detected = False

        myDice.roll()

        if myDice.is_straight() and score.is_category_available("Straight"):
            straight_detected = True
        elif myDice.is_full() and score.is_category_available("Full"):
            full_detected = True

        if print_record_games:
            game_events_to_record.append(f"\nTurn {turn}\n")
            game_events_to_record.append(f"roll {myDice.get_num_rolls()} dice {myDice.dice()}")
        if narrate_games:
            if narrate_french:
                print(f"Tour {turn}")
                print(f"lancement numéro: {myDice.get_num_rolls()} dés: {myDice.dice()}")
            else:
                print(f"Turn {turn}")
                print(f"Roll number: {myDice.get_num_rolls()} dice: {myDice.dice()}")
            input()

        if not roll_seq:

            for roll in range(1, NUM_ROLLS):

                # This is what is used now for the state:
                # Rows are in the same order as score q table
                q_table_keeping_rows_index = q_table_scoring_rows[0].index(
                    myDice.get_dict_as_vector() + score.get_available_cat_vector())

                # Action

                # If using big keeping table:
                # keeping_actions_mask = list(keep_action_mask_dict[myDice.as_short_string()])
                # action = (ma.masked_array(q_table_keeping[q_table_keeping_rows_index][0:NUM_KEEPING_ACTIONS],
                # keeping_actions_mask)).argmax()
                # action = (ma.masked_array(q_table_keeping[q_table_keeping_rows_index][0:NUM_KEEPING_ACTIONS],
                #             keep_action_mask_dict[myDice.as_short_string()])).argmax()

                # For the action, we are now using a 1D vector
                # So using q_table_keeping_reduced
                action = q_table_keeping[q_table_keeping_rows_index]
                action_histo[action] += 1
                total_actions += 1

                kept_all = False

                if list_set_keep_actions[action] == {'K4'} and not score.is_category_available('Straight'):
                    bad_action_K4_counter += 1
                    # if print_record_games:
                    #     game_events_to_record.append(f" BAD ACTION 60 ")
                # About to commit action to reroll
                myDice.make_list_reroll_for_selected_die_faces(list_set_keep_actions[action])

                dice_set_before_reroll = myDice.as_set()  # for knowing if it's Keep All action
                # Full override ... this should be done by training ... it was ok before, but now I have to override :/
                if myDice.is_full() and score.is_category_available("Full"):
                    full_detected = True
                    # if print_record_games:
                    #     game_events_to_record.append(f" full detected ")
                # also override for straight but that one is A LOT less of an issue ... like 4 or 5 per thou
                if myDice.is_straight() and score.is_category_available("Straight"):
                    straight_detected = True
                    # if print_record_games:
                    #     game_events_to_record.append(f" straight detected ")
                myDice.roll_list_reroll()  # put in the else if you want override

                if print_record_games:
                    # game_events_to_record.append(f"row = {q_table_keeping_rows_index} ")
                    if list_set_keep_actions[action] == empty_set:
                        game_events_to_record.append(f" keep no dice\n")
                    elif dice_set_before_reroll == list_set_keep_actions[action]:
                        game_events_to_record.append(f" keep all dice")
                        kept_all = True
                        break  # no need to keep going with more rolls
                    else:
                        if list_set_keep_actions[action] == {'K4'}:
                            game_events_to_record.append(f" keep four singletons\n")
                        else:
                            game_events_to_record.append(f" keep the {list_set_keep_actions[action]}\n")
                    game_events_to_record.append(f"roll {myDice.get_num_rolls()} dice {myDice.dice()}")

                if narrate_games:
                    if list_set_keep_actions[action] == empty_set:
                        if narrate_french:
                            print(f"Je garde aucun dé")
                        else:
                            print(f"I keep no dice")
                        input()
                    elif dice_set_before_reroll == list_set_keep_actions[action]:
                        if narrate_french:
                            print(f"Je garde tous les dés")
                        else:
                            print(f"I keep all dice")
                        input()
                        break  # no need to keep going with more rolls
                    else:
                        if narrate_french:
                            print(f"Je garde les {list_set_keep_actions[action]}")
                        else:
                            print(f"I keep the {list_set_keep_actions[action]}")
                        input()
                    if narrate_french:
                        print(f"lancement numéro: {myDice.get_num_rolls()} dés: {myDice.dice()}")
                    else:
                        print(f"Roll number: {myDice.get_num_rolls()} dice: {myDice.dice()}")
                    input()

        else:  # Rolling from a pre-programmed sequence
            myDice.roll_seq(turn)

        # Score category

        # score.score_a_category(score_int_to_cat(state_face_max_die_count), myDice.dice())
        # Commented out the above "easy score", now we use the scoring q_table
        # need to compute q_table_score index
        # hey below whys q_table_scoring_rows[0] -> it's ok!
        q_table_scoring_rows_index = q_table_scoring_rows[0].index(myDice.get_dict_as_vector() + score.get_available_cat_vector())

        # this below if using the full size scoring q_table
        # category_scored = score.score_with_q_table(q_table_scoring, q_table_scoring_rows_index, myDice)
        # Now if using the reduced size scoring q_table
        category_scored = q_table_scoring[q_table_scoring_rows_index] + 1
        # print("cat scored = ", category_scored)
        score.score_a_category(score_int_to_cat(category_scored), myDice)

        if roll_seq:
            print(f"{myDice} category scored = {score_int_to_cat(category_scored)} category score = "
                  f"{score.get_category_score(score_int_to_cat(category_scored))}")

        if print_record_games:
            game_events_to_record.append(f"\nScored {score.get_category_score(score_int_to_cat(category_scored))}"
                                         f" in category {score_int_to_cat(category_scored)}\n")
            if straight_detected and score_int_to_cat(category_scored) != 'Straight':
                game_events_to_record.append(f"FAILED TO SCORE STRAIGHT\n")
                # straight_detected = False  # reset flag
                score_straight_fails += 1
            if full_detected and score_int_to_cat(category_scored) != 'Full':
                game_events_to_record.append(f"FAILED TO SCORE FULL\n")
                # full_detected = False  # reset flag
                # full_score_failed = True
                score_full_fails += 1
            if kept_all and score.get_category_score(score_int_to_cat(category_scored)) == 0:
                game_events_to_record.append(f"Bad keep all\n")

        if narrate_games:
            if narrate_french:
                print(f"Je score {score.get_category_score(score_int_to_cat(category_scored))}"
                      f" dans la catégorie {french(score_int_to_cat(category_scored))}\n")
            else:
                print(f"I score {score.get_category_score(score_int_to_cat(category_scored))}"
                      f" in category {score_int_to_cat(category_scored)}\n")
            input()
            scorecard_string = score.print_scorecard()
            for score_line in scorecard_string:
                print(score_line)
            input()
            os.system('clear')

        straight_detected = False
        full_detected = False

        all_scored = score.all_scored()

    # if full_score_failed:
    if score.get_bonus() != 0:
        bonus_cnt = bonus_cnt+1
    if score.get_category_score('Yum') == 30:
        yum_counter += 1
    if score.get_category_score('Straight') == 25:
        straight_counter += 1
    if score.get_category_score('Full') == 25:
        full_counter += 1
    if score.get_category_score('Low') > 0:
        lo_counter += 1
    if score.get_category_score('High') > 0:
        hi_counter += 1
    game_score = score.get_total_score() + score.get_bonus()
    print(f"Total score for game {game_number + 1} is {game_score}\n")

    if print_record_games:
        game_events_to_record.append(f"\nTotal score for game {game_number+1} is {game_score}"
                                     f" bonus is {score.get_bonus()}\n")
        game_events_to_record.append(f"-----\n\n")

    if narrate_games:
        scorecard_string = score.print_scorecard()
        for score_line in scorecard_string:
            print(score_line)

    scores.append(game_score)

average_score = sum(scores) / NUM_GAMES

if not narrate_games:
    print("average score = ", average_score)
    print("min score = ", min(scores))
    print("max score = ", max(scores))
    print(f"bonus happened {bonus_cnt} times")
    print(f"yum count = {yum_counter}")
    print(f"straight count = {straight_counter}")
    print(f"full count = {full_counter}")
    print(f"lo count = {lo_counter}")
    print(f"hi count = {hi_counter}")
    # print(f"bad action 60 happened {bad_action_60_counter} times")
    # print(f"straight score fails = {score_straight_fails}")
    # print(f"full score fails = {score_full_fails}")

if print_record_games:
    game_events_to_record.append(f"\nSummary: \n")
    game_events_to_record.append(f"average score = {average_score}\n")
    game_events_to_record.append(f"min score = {min(scores)}\n")
    game_events_to_record.append(f"max score = {max(scores)}\n")
    game_events_to_record.append(f"bonus happened {bonus_cnt} times\n")
    game_events_to_record.append(f"yum count = {yum_counter}\n")
    game_events_to_record.append(f"straight count = {straight_counter}\n")
    game_events_to_record.append(f"full count = {full_counter}\n")
    game_events_to_record.append(f"lo count = {lo_counter}\n")
    game_events_to_record.append(f"hi count = {hi_counter}\n")
    # game_events_to_record.append(f"table action 60 happened {bad_action_60_counter} times\n")
    # game_events_to_record.append(f"straight score fails = {score_straight_fails}\n")
    # game_events_to_record.append(f"full score fails = {score_full_fails}\n")

if print_record_games:
    with open("games.txt", "wt") as file_record_games:
        file_record_games.writelines(game_events_to_record)

# Histogram of games
if not narrate_games:
    np_scores = np.array(scores)
    num_bins = 30
    n, bins, patches = plt.hist(scores, density=True, bins=30)
    plt.show()

# Histogram of actions
i=0
action_histo = action_histo/total_actions
cumulative_sum = 0
for action_percentage in action_histo:
    cumulative_sum += action_percentage
    print(f"{i}, {100*action_percentage:.2f}, {list_set_keep_actions[i]}")
    i = i+1
# Sum is 1?
print("cumulative_sum = ", cumulative_sum)

if not narrate_games:
    plt.plot(action_histo)
    # n, bins, patches = plt.hist(action_histo, density=True, bins=NUM_KEEPING_ACTIONS+1)
    plt.show()
