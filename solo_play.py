# Playing Solo Yum games
# with user input


from score import *
from constants import *
import pickle
import numpy as np
from numpy import ma
from do_q_table import do_q_table_rows
from do_keep_action_q_table import action_q_table
import matplotlib.pyplot as plt

PRINT = False


# For saving games
game_events_to_record = []
print_record_games = True

myDice = DiceSet()
score = Score()

scores = []
bonus_cnt = 0

all_scored = False

for game_number in range(NUM_GAMES_SOLO):

    if print_record_games:
        game_events_to_record.append(f"Game number {game_number+1}\n")

    turn_done = False
    turn = 0

    score.reset_scores()
    all_scored = False

    while not all_scored:

        turn += 1
        myDice.reset()
        myDice.roll()

        print("initial roll ", myDice.get_num_rolls())
        print(myDice)
        if print_record_games:
            game_events_to_record.append(f"Turn {turn}\n")
            game_events_to_record.append(f"roll {myDice.get_num_rolls()} dice {myDice.dice()}")

        for roll in range(1, NUM_ROLLS):

            # ask for list reroll
            list_reroll_string = input("Enter dice to keep (r for reroll, k for keep) --> ")
            list_reroll = []
            for c in list_reroll_string:
                if c == 'r':
                    list_reroll.append(True)
                else:
                    list_reroll.append(False)
            myDice.set_list_reroll(list_reroll)
            print(f"list re-roll {myDice.get_list_reroll()}")
            myDice.roll_list_reroll()
            print(f"dice = {myDice}")

        # ask user to score
        score.print_available_cats()
        score_category = score_char_to_cat(input("score (1 2 3 4 5 6 Y S F L H)? --> "))
        score.score_a_category(score_category, myDice)

        print("category scored = ", score_category)
        score.print_scorecard()

        if print_record_games:
            game_events_to_record.append(f"\nScored {score.get_category_score(score_category)}"
                                         f" in category {score_category}\n")

        all_scored = score.all_scored()

    game_score = score.get_total_score()
    print(f"Total score for game {game_number + 1} is {game_score + score.get_bonus()}\n")

    if score.get_bonus() != 0:
        bonus_cnt = bonus_cnt+1

    if print_record_games:
        game_events_to_record.append(f"Total score for game {game_number+1} is {game_score + score.get_bonus()}"
                                     f" bonus is {score.get_bonus()}\n")
        game_events_to_record.append(f"-----\n")

    scores.append(game_score)

print(f"bonus happened {bonus_cnt} times")

if print_record_games:
    with open("solo_games.txt", "at") as file_record_games:
        file_record_games.write("\n")
        file_record_games.writelines(game_events_to_record)

# Histogram
# np_scores = np.array(scores)
# num_bins = 30
# n, bins, patches = plt.hist(scores, num_bins, facecolor='blue', alpha=0.5)
