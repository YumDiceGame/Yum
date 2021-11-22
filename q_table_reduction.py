import pickle


def reduce_q_table(q_table):
    '''
    Transform the q_table to a row vector
    We only need the max action
    '''
    q_table_reduced = []
    for row in q_table:
        q_table_reduced.append(row)
        # q_table_reduced.append(int(row.argmax()))
        # a_row = []
        # for el in row:
        #     a_row.append(int(el))
        # q_table_reduced.append(a_row)
        # # q_table_reduced.append(int(row))
    return q_table_reduced


with open("q_table_scoring_straight.pickle", "rb") as score_q_table_file:
    q_table_scoring_straight = pickle.load(score_q_table_file)

q_table_scoring_reduced = reduce_q_table(q_table_scoring_straight)
with open("q_table_scoring_reduced.pickle", "wb") as f:
    pickle.dump(q_table_scoring_reduced, f)

with open("q_table_keeping.pickle", "rb") as keeping_q_table_file:
    q_table_keeping = pickle.load(keeping_q_table_file)

q_table_keeping_reduced = reduce_q_table(q_table_keeping)
with open("q_table_keeping_reduced.pickle", "wb") as f:
    pickle.dump(q_table_keeping_reduced, f)