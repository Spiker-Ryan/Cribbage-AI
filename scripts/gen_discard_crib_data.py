"""Stage 1b data generation: hand EV *and* crib EV.

Stage 1a labelled each split with the expected value of the four cards you
keep. That model is blind to the fact that two cards go somewhere - and
where they go flips the strategy completely.

Two labels per split now:
  hand_ev  - exact, averaged over every cut you cannot see (same as 1a)
  crib_ev  - Monte Carlo, because two of the crib's four cards are the
             opponent's and you cannot see them

The net predicts both. Who deals is applied afterwards as a known rule:
    total = hand_ev + crib_ev   when it is your crib
    total = hand_ev - crib_ev   when it is theirs
so is_dealer never enters the network and the sign can never be learned wrong.
"""
import csv
import itertools
import random

import cribbage.cards
from cribbage.scoring import score_hand


# ===============================================
# Opponent model - deliberately dumb for now
# ===============================================

def opponent_throws(unseen):
    """The 2 cards the opponent throws to the crib.

    Returns (their_throws, cards_still_unaccounted_for).

    # TODO sample 6 cards for the opponent out of `unseen`
    # TODO pick 2 of those 6 as their throws - random, for this first pass
    # TODO return the 2 throws, plus the cards not in their hand
    #      (the cut has to come from what neither player holds)

    Later this gets replaced by the Stage 1a net choosing their discard,
    which is when crib_ev starts differing based on who deals - they throw
    junk into your crib and good cards into their own.
    """
    opp_hand = random.sample(unseen, 6)
    throws = random.sample(opp_hand, 2)

    return (throws, list(set(unseen)-set(opp_hand)))



# ===============================================
# The two labels
# ===============================================

def hand_ev(kept, unseen):
    """Exact expected score of the 4 cards you keep.

    Unchanged from Stage 1a.

    # TODO sum score_hand(kept, cut, False) over every card in `unseen`
    # TODO divide by len(unseen)
    """
    tot = 0
    for cut_card in unseen:
        tot += score_hand(kept, cut_card, False)
    tot /= len(unseen)
    return tot



def crib_ev(my_throws, unseen, n_samples):
    """Expected crib score, averaged over sampled opponent throws.

    # TODO repeat n_samples times:
    # TODO   get the opponent's 2 throws via opponent_throws(unseen)
    # TODO   crib = my 2 throws + their 2 throws
    # TODO   pick a cut at random from the cards neither player holds
    # TODO   score it with score_hand(crib, cut, True)
    # TODO         ^^^^ is_crib=True - a crib flush needs all five cards
    # TODO average over the samples

    Note the cut here is *sampled*, not enumerated. Enumerating all cuts
    per sample would multiply the cost by ~40 for accuracy you do not need
    - the opponent-throw uncertainty already dominates the noise.
    """
    score = 0
    for i in range(n_samples):
        opp_throws, cut_pool = opponent_throws(unseen)
        crib = my_throws + opp_throws
        cut = cribbage.cards.cut_deck(cut_pool)
        score += score_hand(crib, cut, True)
    return score / n_samples

# ===============================================
# Generation
# ===============================================

def main():
    num_of_sims = 100000      # TODO time one deal before scaling this up
    n_samples = 200         # crib Monte Carlo samples per split

    # TODO open the output file at data/raw/discard_crib_labels.csv
    with open("/Users/ryanwilliams/Coding/Cribbage AI/data/raw/discard_crib_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # TODO write header: kept1..kept4, thrown1, thrown2, hand_ev, crib_ev
        writer.writerow(["kept1","kept2","kept3","kept4","thrown1","thrown2","hand_ev", "crib_ev"])
        write_count = 0

        # TODO for each sim:
        for sim in range(num_of_sims):
            dealer_hand, deck = cribbage.cards.gen_one_hand()
            combos = itertools.combinations(dealer_hand, 4)
            for combo in combos:
                thrown = list(set(dealer_hand) - set(combo))
                h = hand_ev(combo, deck)
                c = crib_ev(thrown, deck, n_samples)
                writer.writerow([*combo, *sorted(thrown), h, c])
                write_count += 1
                if write_count % 250 == 0:
                    print(f"Write count: {write_count}")
                    f.flush()


    # TODO   deal 6 cards, get the 46 unseen
    # TODO   for each of the 15 ways to keep 4:
    # TODO     work out the 2 thrown
    # TODO     h = hand_ev(kept, unseen)
    # TODO     c = crib_ev(thrown, unseen, n_samples)
    # TODO     write one row

    # TODO print progress every few thousand rows - this run is far slower
    #      than Stage 1a and a silent terminal is miserable


if __name__ == '__main__':
    main()
