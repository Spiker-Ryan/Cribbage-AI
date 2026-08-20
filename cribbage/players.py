"""Player implementations.

Anything with choose_discard() and choose_play() is a player. The engine
never cares which is which.

    choose_discard(hand, is_dealer) -> the 2 cards to throw to the crib
    choose_play(legal, state)       -> one card from legal

`state` carries only what the acting player may legally see: their own
remaining cards, the cut, the count, the current pile, the round history,
and whether they deal. Never the opponent's hand.
"""
import itertools
import random

import numpy as np
import torch

from ml.features import encode_split, encode_pegging
from ml.models import DiscardCribNet, PeggingNet

ROOT = '/Users/ryanwilliams/Coding/Cribbage AI'
CHECKPOINT = f'{ROOT}/checkpoints/discard_crib_net_epoch12.pt'
PEGGING_CHECKPOINT = f'{ROOT}/checkpoints/pegging_net.pt'


class RandomPlayer:
    """Uniform random. The baseline every other player has to beat."""

    def choose_discard(self, hand, is_dealer):
        return random.sample(hand, 2)

    def choose_play(self, legal, state):
        return random.choice(legal)


def load_model(path=CHECKPOINT):
    model = DiscardCribNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def load_pegging_model(path=PEGGING_CHECKPOINT):
    model = PeggingNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


class NetPlayer:
    """Discards with the Stage 1b model. Pegs with the Stage 3 model if one
    is supplied, otherwise at random.

    Random pegging is the bootstrap that generated Stage 3's first batch of
    training data, so leaving pegging_model=None reproduces that behaviour.

    Pass shared models when creating many of these. Loading a checkpoint
    costs far more than playing a game, so a generation run that builds one
    per player per game spends nearly all its time on disk.
    """

    def __init__(self, model=None, pegging_model=None):
        self.model = load_model() if model is None else model
        self.pegging_model = pegging_model
        self.thrown = []            # my own discards - I saw them, they did not

    def choose_play(self, legal, state):
        if self.pegging_model is None:
            return random.choice(legal)

        # encode_pegging wants the two cards I threw as part of what I have
        # seen; the engine cannot supply those without leaking them to the
        # other player, so I remember them myself
        record = dict(state, my_thrown=self.thrown)

        batch = np.array([encode_pegging(record, c) for c in legal],
                         dtype=np.float32)
        with torch.no_grad():
            out = self.pegging_model(torch.from_numpy(batch))
            values = PeggingNet.differential(out).numpy()

        return legal[int(np.argmax(values))]

    def choose_discard(self, hand, is_dealer):
        splits = [(list(kept), [c for c in hand if c not in kept])
                  for kept in itertools.combinations(hand, 4)]

        batch = np.array([encode_split(kept, thrown) for kept, thrown in splits],
                         dtype=np.float32)
        with torch.no_grad():
            out = self.model(torch.from_numpy(batch)).numpy()

        # the crib is yours when you deal and theirs otherwise - a rule the
        # network was deliberately not asked to learn, so it cannot get the
        # sign backwards
        sign = 1 if is_dealer else -1
        totals = out[:, 0] + sign * out[:, 1]

        # the engine wants the cards thrown away, not the ones kept
        thrown = splits[int(np.argmax(totals))][1]
        self.thrown = list(thrown)      # remembered for pegging inference
        return thrown
