"""Does the Stage 3 pegging model actually play better?

Validation MSE cannot answer that - the labels are realised outcomes, so
most of the remaining error is the inherent randomness of what happened,
not model error. The only honest test is to sit the two policies down
against each other.

Both sides discard with the same Stage 1b model, so pegging is the only
thing that differs and the comparison is clean.
"""
import random
import statistics

from cribbage.engine import CribbageGame
from cribbage.players import NetPlayer, load_model, load_pegging_model

NUM_GAMES = 2000


def match(make_a, make_b, n_games, seed=0):
    """Plays n_games, alternating which side deals first. Returns A's results."""
    random.seed(seed)
    a_wins = 0
    margins = []
    peg_totals = [[], []]

    for i in range(n_games):
        a, b = make_a(), make_b()
        # alternate seats so neither policy gets the first-dealer edge
        players = [a, b] if i % 2 == 0 else [b, a]
        a_idx = 0 if i % 2 == 0 else 1

        game = CribbageGame(players[0], players[1])
        winner = game.play_game()

        if winner == a_idx:
            a_wins += 1
        margins.append(game.scores[a_idx] - game.scores[1 - a_idx])

    return {
        "win_rate": a_wins / n_games,
        "mean_margin": statistics.mean(margins),
    }


def main():
    discard = load_model()
    pegging = load_pegging_model()

    net_peg = lambda: NetPlayer(discard, pegging)
    rand_peg = lambda: NetPlayer(discard, None)

    print(f"{NUM_GAMES:,} games, both sides discarding with the Stage 1b model")
    print("only the pegging policy differs\n")

    r = match(net_peg, rand_peg, NUM_GAMES, seed=1)
    print(f"pegging net vs random pegging")
    print(f"  win rate     {r['win_rate']:.1%}   (50% would mean no improvement)")
    print(f"  mean margin  {r['mean_margin']:+.2f} points")

    # control: the same policy on both sides should land on a coin flip.
    # If this is not ~50%, something in the harness favours a seat.
    c = match(rand_peg, rand_peg, NUM_GAMES // 2, seed=2)
    print(f"\ncontrol - random vs random")
    print(f"  win rate     {c['win_rate']:.1%}   (should be ~50%)")
    print(f"  mean margin  {c['mean_margin']:+.2f}")


if __name__ == '__main__':
    main()
