from dice import *
from score import *
from do_keep_action_q_table import *
import numpy as np
import numpy.ma as ma
myDice = DiceSet()

myScore = Score()
myScore.reset_scores()
CompScore = Score()
CompScore.reset_scores()

my_scorecard_string = myScore.print_scorecard()
comp_scorecard_string = CompScore.print_scorecard()
both_scorecards = zip(my_scorecard_string, comp_scorecard_string)

agent = "Agent"
print(f"Player{agent.rjust(30)}")
for line in both_scorecards:
    print(f"{line[0]}{line[1].rjust(30)}")

print("\n\n")
for line in my_scorecard_string:
    print(line)
