"""Metrics, and the baselines every model has to beat.

Stage 1 targets: mean absolute error in points, how often the net picks the
same split as exhaustive search, and what its mistakes cost. Baselines:
random choice, predict-the-mean, and linear regression on the same features.
"""
import itertools
import random

import numpy as np
import torch

import cribbage.cards
from ml.features import encode_split
from ml.models import DiscardNet, DiscardCribNet
from scripts.gen_discard_crib_data import hand_ev, crib_ev

CHECKPOINT = '/Users/ryanwilliams/Coding/Cribbage AI/checkpoints/discard_crib_net_epoch12.pt'
STAGE_1A_CHECKPOINT = '/Users/ryanwilliams/Coding/Cribbage AI/checkpoints/discard_net_epoch15.pt'


def make_testing_hands(n, n_samples=2000, seed=None, verbose=True):
    """n fresh deals. Each deal is a list of 15 (kept, thrown, hand_ev, crib_ev).

    hand_ev is exact - averaged over every cut the dealer cannot see.
    crib_ev is Monte Carlo, so n_samples is much higher here than during
    data generation: at 200 samples the label noise floor is MSE 0.048,
    which would swamp the model's actual error. At 2000 it drops to 0.005.

    Who deals is applied later, by score_model, as hand_ev +/- crib_ev.
    """
    if seed is not None:
        random.seed(seed)

    deals = []
    for i in range(n):
        dealer_hand, deck = cribbage.cards.gen_one_hand()
        splits = []
        for combo in itertools.combinations(dealer_hand, 4):
            kept = list(combo)
            thrown = sorted(set(dealer_hand) - set(combo))
            splits.append((kept, thrown,
                           hand_ev(kept, deck),
                           crib_ev(thrown, deck, n_samples)))
        deals.append(splits)

        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n} deals", flush=True)
    return deals


def discard_net():
    # the Stage 1a model - one output, hand EV only. Kept for comparison;
    # it has a different architecture so it needs its own checkpoint.
    model = DiscardNet()
    model.load_state_dict(torch.load(STAGE_1A_CHECKPOINT))
    model.eval()
    return model

def discard_crib_net():
    model = DiscardCribNet()
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()
    return model
# ===============================================
# Scorers: splits -> one number per split.
# Anything with this shape can be measured by score_model.
# ===============================================

def encode_deal(splits):
    return np.array([encode_split(k, t) for k, t, _, _ in splits], dtype=np.float32)


def make_net_scorer(model, sign):
    def scorer(splits):
        with torch.no_grad():
            batch = torch.from_numpy(encode_deal(splits))
            out = model(batch).numpy()
            return out[:,0] + sign * out[:,1]
    return scorer


def random_scorer(splits):
    return np.random.rand(len(splits))


def make_constant_scorer(value):
    # every split ties, so argmax always returns index 0 - this is really
    # "always keep the first four cards dealt". Its MAE is the number worth
    # reading; its top1 and ev_loss are just that arbitrary policy.
    def scorer(splits):
        return np.full(len(splits), value)
    return scorer


def make_linear_scorer(weights, intercept):
    def scorer(splits):
        return encode_deal(splits) @ weights + intercept
    return scorer


def fit_linear(deals, sign):
    """Least-squares fit on the same 104 features. No training loop needed.

    sign folds in who deals: the target is hand_ev + sign * crib_ev, so
    dealer and pone get separately fitted baselines.
    """
    X = np.array([encode_split(k, t) for d in deals for k, t, _, _ in d], dtype=np.float64)
    y = np.array([h + sign * c for d in deals for _, _, h, c in d])
    # append a column of ones so lstsq solves for the intercept too
    X1 = np.hstack([X, np.ones((len(X), 1))])
    solution, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return solution[:-1], solution[-1]


# ===============================================
# Metrics
# ===============================================

def score_model(scorer, deals, sign):
    abs_error = []
    agreements = []
    ev_losses = []

    for splits in deals:
        truths = np.array([h + sign * c for _, _, h, c in splits])
        scores = scorer(splits)

        abs_error.extend(np.abs(scores - truths))
        best = np.argmax(truths)
        picked = np.argmax(scores)
        agreements.append(best == picked)
        ev_losses.append(truths[best] - truths[picked])

    return {
        "mae": np.mean(abs_error),
        "top1": np.mean(agreements),
        "ev_loss": np.mean(ev_losses),
        "worst_ev": np.max(ev_losses),
    }


def main():
    np.random.seed(0)

    print("generating deals...")
    fit_deals = make_testing_hands(500, seed=1)     # only for the linear baseline
    eval_deals = make_testing_hands(2000, seed=2)   # every model sees these

    model = discard_crib_net()

    print(f"\n{len(eval_deals)} fresh deals, {len(eval_deals) * 15} splits")

    # dealer and pone are different decisions - the crib helps you or hurts
    # you - so every model gets measured twice.
    for role, sign in (("dealer", +1), ("pone", -1)):
        all_labels = [h + sign * c for d in fit_deals for _, _, h, c in d]
        weights, intercept = fit_linear(fit_deals, sign)

        scorers = {
            "random": random_scorer,
            "predict-mean": make_constant_scorer(np.mean(all_labels)),
            "linear": make_linear_scorer(weights, intercept),
            "discard net": make_net_scorer(model, sign),
        }

        print(f"\nas {role}")
        print(f"{'model':<14}{'MAE':>9}{'top-1':>9}{'EV loss':>10}{'worst EV':>10}")
        print("-" * 52)
        for name, scorer in scorers.items():
            m = score_model(scorer, eval_deals, sign)
            print(f"{name:<14}{m['mae']:>9.4f}{m['top1']:>8.1%}{m['ev_loss']:>10.4f}{m['worst_ev']:>10.4f}")


if __name__ == '__main__':
    main()
