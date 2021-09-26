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
LEARNING_RATE = 0.2  # was 0.1 till 9/25/21  the 0.2 is really new
DISCOUNT = 0.725  # was 0.95 till 9/21/21.  0.75 gave great results for above the line
NUM_EPISODES = 6_000_000  # 50_000_000 for keeping / 6_000_000 for scoring (no eps) then 12M with eps on top
EVAL_Q_TABLE = 5_000  # time to evaluate q table for convergence
NUM_GAMES = 100
NUM_GAMES_SOLO = 1
NUM_KEEPING_ACTIONS = 61
Q_TABLE_SAVE_INTERVAL = 10_000_000
NUM_SHOW = 1000
NUM_TRACK_SCORE = 5_000
