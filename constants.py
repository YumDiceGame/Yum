# Diffed with across_line_old 7/22/21

NUM_DICE = 5
NUM_DIE_FACES = 6
NUM_ROLLS = 3
OBS_SPACE_SIZE = 252  # See do_q_table.py
NUM_SCORE_CATEGORIES = 11  # 1's through 6's, Yum, Straight, Full, Low, High
TWO_TO_NUM_SCORE_CATEGORIES = 2048  # 2^NUM_SCORE_CATEGORIES
NUM_SCORE_CAT_ABOVE_LINE = 6
ABOVE_THE_LINE_CATEGORIES = ['Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes']
BELOW_THE_LINE_CATEGORIES = ['Yum', 'Straight', 'Full', 'Low', 'High']
# Column numbers of categories in score table
YUM_COL = 6 # Cat7 - 1 for pos
LEARNING_RATE = 0.2  # was 0.1 till 9/25/21  the 0.2, 0.275 gave good results
DISCOUNT = 0.4  # was 0.95 till 9/21/21.  0.75, 0.725 gave great results for above the line 0.7, 0.5 ok
NUM_EPISODES = 5_500_000  # 10_000_000  # was 10M
START_EPSILON_DECAYING = 500_000
END_EPSILON_DECAYING = NUM_EPISODES-1  # 9*NUM_EPISODES//10  # was 4//5
EVAL_Q_TABLE = 20_000  # time to evaluate q table for convergence
NUM_GAMES = 1000
NUM_GAMES_IN_LINE_EVAL = 100  # was 200 last time
NUM_KEEPING_ACTIONS = 61
Q_TABLE_SAVE_INTERVAL = 12_000_000
NUM_SHOW = 1_000
NUM_TRACK_SCORE = 10_000

POT_MAX_SCORE_IND = 1
POT_MAX_SCORE_CAT_IND = 0
