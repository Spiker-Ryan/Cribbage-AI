"""Stage 3 data generation: one row per pegging decision.

Unlike the discard labels, these cannot be computed - the value of playing a
card depends on how the rest of the round unfolds. So we play real games and
label each decision afterwards with what actually happened.

Two rules this file exists to enforce:

  Only log what the player can legally see. The engine knows both hands;
  a record that leaks the opponent's cards trains a model that cannot be
  used at inference.

  Score differentially. The label is (my pegging points from here on) minus
  (theirs). Logging only your own points teaches the model to take a quick
  fifteen while handing back a run of five.

Raw card strings go to disk, not feature vectors - the encoding will change
several times and regenerating is far more expensive than re-encoding.

Self-play (Stage 4). If a pegging checkpoint exists, both players use it and
the data describes what happens between two competent peggers. Batch one had
to come from random play - a model trained on it is excellent at beating
random opponents and has never seen a good one. Each pass through
generate -> train -> regenerate closes that gap.

Set SELF_PLAY = False to reproduce the random-pegging bootstrap.
"""
import json
import random
from pathlib import Path

from cribbage.engine import CribbageGame, GameOver
from cribbage.players import (NetPlayer, RandomPlayer,
                              load_model, load_pegging_model)

OUT_PATH = Path('/Users/ryanwilliams/Coding/Cribbage AI/data/raw/pegging_decisions.jsonl')
NUM_GAMES = 150_000
SELF_PLAY = True


class LoggingPlayer:
    """Wraps another player and records every pegging decision it makes.

    The wrapper only ever sees what the engine hands its inner player, plus
    the shared round history - so it cannot leak hidden information even by
    accident.
    """

    def __init__(self, inner, idx):
        self.inner = inner
        self.idx = idx
        self.game = None        # set once the game object exists
        self.records = None     # swapped in per round, shared by both players
        self.hand = []          # my 4 kept cards, shrinking as I play
        self.thrown = []        # my 2 discards - I know these, opponent does not

    def choose_discard(self, hand, is_dealer):
        thrown = self.inner.choose_discard(hand, is_dealer)
        self.thrown = list(thrown)
        self.hand = [c for c in hand if c not in thrown]
        return thrown

    def choose_play(self, legal, state):
        card = self.inner.choose_play(legal, state)

        # state is exactly what the engine judged this player may see, so
        # copying it wholesale cannot leak anything. The extras are things
        # only this wrapper knows: the score at decision time (needed to
        # work out the label afterwards), my own discards, and what I chose.
        record = dict(state)
        record["scores_at"] = list(self.game.scores)    # stripped after labelling
        record["my_thrown"] = list(self.thrown)
        record["legal"] = list(legal)
        record["action"] = card
        self.records.append(record)

        self.hand.remove(card)
        return card


def label_round(records, scores_before, scores_after):
    """Back-fill each decision with the points it led to.

    A decision's value is what happened from that moment to the end of the
    round's pegging, counted as my points minus the opponent's.
    """
    for r in records:
        me = r["player"]
        them = 1 - me
        # logged separately, not just as their difference: the net predicts
        # both, so a bad prediction can be traced to overvaluing offence or
        # to underestimating what the play hands over
        r["gain"] = scores_after[me] - r["scores_at"][me]
        r["give"] = scores_after[them] - r["scores_at"][them]
        r["label"] = r["gain"] - r["give"]
        del r["scores_at"]      # was only needed to compute the labels

    # score situation is legal information and matters near 121
    for r in records:
        r["my_score"] = scores_before[r["player"]]
        r["opp_score"] = scores_before[1 - r["player"]]


def play_logged_game(make_player):
    """One full game. Returns the labelled decisions from it."""
    p0 = LoggingPlayer(make_player(), 0)
    p1 = LoggingPlayer(make_player(), 1)

    game = CribbageGame(p0, p1)
    p0.game = p1.game = game
    game.dealer = random.randrange(2)

    out = []
    try:
        while not game.win:
            round_records = []
            p0.records = p1.records = round_records

            game.deal()
            game.collect_crib_cards()
            game.cut_card()

            before = list(game.scores)
            game.play_pegging()
            after = list(game.scores)

            label_round(round_records, before, after)
            out.extend(round_records)

            game.count_cards()
            game.dealer = 1 - game.dealer
    except GameOver:
        # The round in progress never finished, so its decisions have no
        # labels. Dropping them costs about one round in twelve and is far
        # cleaner than inventing a partial label.
        pass

    return out


def main():
    # one model each, shared by every player - loading a checkpoint costs
    # far more than playing a game
    discard = load_model()

    pegging = None
    if SELF_PLAY:
        try:
            pegging = load_pegging_model()
            print("self-play: both sides pegging with the trained model")
        except FileNotFoundError:
            print("no pegging checkpoint yet - falling back to random pegging")
    else:
        print("bootstrap: random pegging")

    make_player = lambda: NetPlayer(discard, pegging)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with open(OUT_PATH, 'w') as f:
        for i in range(NUM_GAMES):
            for record in play_logged_game(make_player):
                f.write(json.dumps(record) + '\n')
                rows += 1

            if (i + 1) % 2000 == 0:
                pct = (i + 1) / NUM_GAMES
                print(f"{i + 1:,}/{NUM_GAMES:,} games ({pct:.0%}), {rows:,} decisions",
                      flush=True)
                f.flush()

    print(f"done: {rows:,} decisions from {NUM_GAMES:,} games -> {OUT_PATH}")


if __name__ == '__main__':
    main()
