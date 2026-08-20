# Cribbage AI

Learning neural networks by building a cribbage agent. Exhaustive search
provides ground-truth labels and grades the models, so every stage can be
checked against a known-correct answer rather than a loss curve.

## Layout

    cribbage/    rules - cards, scoring, engine, players
    ml/          features, models, training, evaluation
    scripts/     data generation and head-to-head evaluation
    tests/       known-answer tests and engine invariants
    data/        generated datasets (gitignored, regenerate with scripts/)
    checkpoints/ trained weights

`cribbage/scoring.py` is verified exhaustively against an independently
written scorer across all 12,994,800 possible (hand + cut) positions:
mean 4.7692, max 29, and the impossible scores come out as 19/25/26/27.

## Stages

**1a - discard value net.** Input: 6 cards plus which 2 are thrown. Output:
expected hand points, labelled exactly by averaging over all 46 unseen cuts.
No game engine required.

> EV loss 0.0009 points per hand, picks the true best split 91.1% of the
> time. Random gives up 3.78. A linear model on the same features manages
> only 15.8% - every cribbage scoring rule is an interaction between cards,
> which a linear model cannot represent at all.

**1b - add the crib.** Two outputs, hand EV and crib EV, combined afterwards
as `hand ± crib` by who deals. Keeping the sign outside the network means it
cannot be learned backwards. Crib EV is Monte Carlo, since the opponent's two
throws are unknown.

> EV loss 0.0044 as dealer, 0.0085 as pone. Throws a 5 into its own crib
> 39.1% of the time and into the opponent's 2.6% - textbook strategy, never
> encoded anywhere, learned entirely from expected values.

**3 - pegging net.** Value net over (state, candidate card) -> points taken
and points given up. Labels come from real games: the differential over the
rest of the round. There is no formula to enumerate here, which is why the
engine had to exist first.

> 68.3% win rate against random pegging, +10 points per game, with both
> sides discarding identically so pegging is the only variable.

**4 - self-play iteration.** Regenerate with the trained model on both sides,
retrain, repeat. Stop when consecutive versions play to a draw.

**2 - opponent model** (not built). Predict the opponent's hidden cards.
The pegging net's `give` column is consistently its weaker one, and that is
exactly the quantity that depends on cards it cannot see.

## Running it

Everything runs as a module from the project root:

    python3 -m scripts.gen_discard_crib_data    # ~3 hours, 1.5M rows
    python3 -m ml.train discard

    python3 -m scripts.gen_pegging_data         # ~20 min, 10M decisions
    python3 -m ml.train pegging
    python3 -m scripts.eval_pegging

Plain `python3 path/to/script.py` will not work - Python puts the script's
own directory on the import path, not the project root.

## Notes to self

- Every model gets compared against a dumb baseline. If it does not clearly
  beat predict-the-mean and a linear fit, something is broken.
- MSE is a proxy. What matters is EV loss for discards and win rate for
  pegging - a model can be accurate on average and still pick wrong.
- Log raw card strings, never feature vectors. Encodings change constantly;
  regenerating data does not.
- Only log what a player can legally see. The engine knows both hands.
