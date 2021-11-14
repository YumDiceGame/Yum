from score import *


def scoring_reward(scored_cat, dice, score, max_die_count, face_max_die_count):

    reward = 0
    scored_amount = score.get_category_score(scored_cat)

    can_yum = dice.is_yum() and score.is_category_available('Yum')
    can_full = dice.is_full() and score.is_category_available('Full')
    can_straight = dice.is_straight() and score.is_category_available('Straight')

    if can_yum:
        if scored_cat == 'Yum':
            reward += 200
        else:
            reward += -240
    else:  # anything other than Yum
        # prioritize above the line scoring
        if can_straight:
            if scored_cat == 'Straight':
                reward += 150
            else:
                reward += -200
        elif can_full:
            if scored_cat == 'Full':
                reward += 150
            else:
                reward += -250
        elif max_die_count >= 3:  # and not can_full:  #
            if scored_cat != score_int_to_cat(face_max_die_count):
                reward -= (90 + 30 * face_max_die_count)  # really need to score 3 of a kind above the line!
                # also pro-rate
            else:  # max_die_count >= 3:  # Right category, and pretty good score, reward!
                reward += 30 * face_max_die_count  # prorate according to face max die count

        # Above the line items
        elif scored_cat in ABOVE_THE_LINE_CATEGORIES:
            num_dice_scored = int(scored_amount / score_cat_to_int(scored_cat))
            if num_dice_scored <= 2:  # Right category, but too low score 2 is bad
                reward += (-30 * score_cat_to_int(scored_cat))  # prorate according to face max die count
                if 0 < max_die_count <= 1:  # 1 is worse
                    reward += (-45 * score_cat_to_int(scored_cat))  # prorate according to face max die count
                # because low score in 6's not as bad as low score in 1's
                # if episode % num_show == 0:
                #     print(f"low mdc {reward}")

        # Hi Lo --->
        # Hi >= 22
        # 21 <= Lo < Hi
        elif scored_cat == 'High' or scored_cat == 'Low':
            # define shorthand quantities
            dice_sum = dice.sum()
            hi_scored = not score.is_category_available('High')
            hi_score = score.get_category_score('High')
            lo_scored = not score.is_category_available('Low')
            lo_score = score.get_category_score('Low')
            if dice_sum >= 21:
                if (hi_scored and hi_score > 0) and (lo_scored and lo_score > 0):
                    reward += 70  # we managed to score both high and low!
                    # if episode % num_show == 0:
                    #     print(f"hi and lo! {reward}")
                elif scored_amount > 0:
                    # the below controls if you scored a too aggressive hi or low
                    # note that scored_amount isn't necessarily equal to dice sum!
                    # scored amount could be zero!
                    reward += score.assess_lo_hi_score(scored_amount, scored_cat)
                    # if episode % num_show == 0:
                    #     print(f"hi lo assess {reward}")
                else:  # you were locked out!
                    reward -= 40
                    # if episode % num_show == 0:
                    #     print(f"hi lo locked out {reward}")
            else:  # dice sum too low
                reward -= 40
                # if episode % num_show == 0:
                #     print(f"hi lo no dice {reward}")
            # Hi Lo <---

        # Scratching a category --->
        if scored_amount == 0:
            reward -= score.scratch_penalty(scored_cat)
            # if episode % num_show == 0:
            #     print(f"scratch {reward}")
        # Scratching a category <---
    return reward

