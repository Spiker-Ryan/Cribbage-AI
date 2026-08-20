"""Raw logged state -> tensors.

Kept separate from data generation on purpose: features will change many
times, and re-encoding a saved dataset is cheap while regenerating
millions of games is not.
"""
import numpy as np

# The canonical card order. Built here rather than imported from
# cribbage.cards because that module shuffles - and this order must never
# change. Every trained model depends on it: index 0 is A_of_S forever.
# Reorder this list and old checkpoints silently read the wrong cards.
SUITS = ['S', 'H', 'D', 'C']
VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

DECK = []
for suit in SUITS:
    for val in VALUES:
        DECK.append(f'{val}_of_{suit}')

# card string -> its slot in a 52-length vector
CARD_INDEX = {card: i for i, card in enumerate(DECK)}

DECK_SIZE = len(DECK)          # 52
DISCARD_FEATURE_SIZE = DECK_SIZE * 2   # 104: kept block + thrown block

# Pegging. A segment caps at 31, and with only 8 cards in play the longest
# possible one is 8 cards (four aces and four 2s is 12). Runs of 5 and 6 do
# happen, so the pile window has to cover the whole segment - three cards
# would miss them.
MAX_PILE = 8
MAX_COUNT = 32                 # counts 0..31 inclusive
MAX_OPP_CARDS = 5              # 0..4 cards left

PEGGING_FEATURE_SIZE = (
    DECK_SIZE          # my remaining hand
    + DECK_SIZE        # the candidate card
    + DECK_SIZE        # every card I have seen
    + DECK_SIZE * MAX_PILE
    + MAX_COUNT
    + MAX_OPP_CARDS
    + 1                # is_dealer
)                      # 610


def one_hot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


def encode_cards(cards):
    """A list of card strings -> a 52-length multi-hot vector.

    The reusable primitive. Later this also encodes seen cards, pegging
    piles, and anything else that is a set of cards.
    """
    vec = np.zeros(DECK_SIZE, dtype=np.float32)
    for card in cards:
        vec[CARD_INDEX[card]] = 1.0   # KeyError here means a malformed card string
    return vec


def encode_split(kept, thrown):
    """One discard decision -> a 104-length input vector.

    Takes card lists, not a CSV row, so the same function works during
    training and during actual play.
    """
    return np.concatenate([encode_cards(kept), encode_cards(thrown)])


def seen_cards(record):
    """Every card the acting player legally knows about.

    Their own six (four still held plus the two they threw), the cut, and
    everything either side has played. This is the whole basis for guessing
    what the opponent still holds - which is why gen_pegging_data logs the
    round history rather than a pre-digested summary.
    """
    seen = list(record["my_hand"]) + list(record["my_thrown"]) + [record["cut"]]
    seen += [card for _, card in record["history"]]
    return seen


def encode_pegging(record, candidate):
    """One pegging decision -> a 610-length input vector.

    `candidate` is the card being evaluated, kept separate from the record
    so the same function scores every legal option at play time. The net is
    a value net: run it once per legal card and take the best.

    Note this uses full 52-card encoding while pegging is actually suit
    blind - no rule in the phase looks at suit. It reuses encode_cards for
    consistency with the discard model, at the cost of making the net
    relearn that the 5 of hearts and the 5 of spades peg identically.
    """
    opp = 1 - record["player"]
    opp_played = sum(1 for who, _ in record["history"] if who == opp)

    pile = record["pile"]
    if len(pile) > MAX_PILE:
        raise ValueError(f"pile of {len(pile)} exceeds MAX_PILE={MAX_PILE}")

    # oldest card first, zero-padded on the right so position means
    # "how far into this segment", not "how far from the end"
    pile_blocks = [encode_cards([c]) for c in pile]
    pile_blocks += [np.zeros(DECK_SIZE, dtype=np.float32)] * (MAX_PILE - len(pile))

    return np.concatenate([
        encode_cards(record["my_hand"]),
        encode_cards([candidate]),
        encode_cards(seen_cards(record)),
        *pile_blocks,
        one_hot(record["count"], MAX_COUNT),
        one_hot(4 - opp_played, MAX_OPP_CARDS),
        np.array([float(record["is_dealer"])], dtype=np.float32),
    ])


if __name__ == '__main__':
    kept = ['5_of_H', '5_of_S', '5_of_C', 'J_of_D']
    thrown = ['2_of_H', '9_of_C']
    vec = encode_split(kept, thrown)

    print(f'vector length         {len(vec)}')
    print(f'dtype                 {vec.dtype}')
    print(f'ones in kept block    {vec[:DECK_SIZE].sum()}  (want 4)')
    print(f'ones in thrown block  {vec[DECK_SIZE:].sum()}  (want 2)')

    # decode back, to prove the mapping round-trips
    back_kept = [DECK[i] for i in np.flatnonzero(vec[:DECK_SIZE])]
    back_thrown = [DECK[i] for i in np.flatnonzero(vec[DECK_SIZE:])]
    print(f'kept round-trips      {sorted(back_kept) == sorted(kept)}')
    print(f'thrown round-trips    {sorted(back_thrown) == sorted(thrown)}')
