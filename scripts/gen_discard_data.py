"""Stage 1 data generation - no game engine required.

Deal 6 cards, and for each of the 15 splits average score_hand() over all
46 possible cut cards. Exact labels, so any error is a bug rather than
label noise. Writes to data/raw/.
"""
import cribbage.cards
import itertools
import csv

from cribbage.scoring import score_hand

def main():
    #number for how many simulations we want to run
    num_of_sims = 100000
    write_count = 0

    with open("/Users/ryanwilliams/Coding/Cribbage AI/data/raw/discard_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kept1","kept2","kept3","kept4","thrown1","thrown2","label"])
        for i in range(num_of_sims):
            dealer_hand, deck = cribbage.cards.gen_one_hand()
            combos = itertools.combinations(dealer_hand, 4)
            for combo in combos:
                missing_cards = list(set(dealer_hand) - set(combo))
                avg = 0
                for cut_card in deck:
                    avg += score_hand(list(combo), cut_card, False)
                avg /= len(deck)
                writer.writerow([*combo, *sorted(missing_cards), avg])
                write_count += 1
            if write_count % 1000 == 0:
                print(f"Write count: {write_count}")

if __name__ == '__main__':
    main()



