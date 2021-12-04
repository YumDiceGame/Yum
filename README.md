#Yum dice game with Q Learning

##Instructions for setting up

###- Recommending to use Pycharm with Python 3.8
###- Need to install numpy and matplotlib
###- Using GitHub as your VCS recommended but not mandatory

##Instructions for running

###- To run a batch of games and output to "games.txt", just run "play.py".  Set the number of games by changing NUM_GAMES to your choice (in constants.py).  One thousand games takes about 5 minutes.  A histogram will pop up after the games have ran.

###- To retrain the q-tables, start with the scoring table.  Run "scoring_learning.py".  Then, to retrain the keeping table, just run "keeping_learning.py"

##Be advised that training is long!
###The keeping table will take 24 hours for 8M episodes (NUM_EPISODES is in constants.py)
###The scoring table takes less time, about 1/3 of the time it takes for the keeping learning

##You don't need to perform any training to run "play.py".  The tables were pre-trained before uploading to git

##For more info, checkout my channel on youtube/YumDiceGame