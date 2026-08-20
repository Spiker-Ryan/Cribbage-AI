"""Stage 2+ data generation: run full games, log one row per decision.

Log raw state (card strings, scores, pile), not feature vectors. Log only
what the acting player can legally see - never the opponent's hand.
"""
