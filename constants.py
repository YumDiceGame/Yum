# Diffed with across_line_old 7/22/21

NUM_DICE = 5
NUM_DIE_FACES = 6
NUM_ROLLS = 3
OBS_SPACE_SIZE = 252  # See do_q_table.py
NUM_SCORE_CATEGORIES = 11  # 1's through 6's, Yum, Straight, Full, Low, High
TWO_TO_NUM_SCORE_CATEGORIES = 2048  # 2^NUM_SCORE_CATEGORIES
NUM_SCORE_CAT_ABOVE_LINE = 6
ABOVE_THE_LINE_CATEGORIES = ['Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes']
# Column numbers of categories in score table
YUM_COL = 6 # Cat7 - 1 for pos
LEARNING_RATE = 0.125  # was 0.1 till 6/29/21
DISCOUNT = 0.9  # was 0.95 till 7/29/21
NUM_EPISODES = 25_000_000
EVAL_Q_TABLE = 100_000  # time to evaluate q table for convergence
NUM_GAMES = 100
NUM_GAMES_SOLO = 1
NUM_KEEPING_ACTIONS = 61
Q_TABLE_SAVE_INTERVAL = 5_000_000
