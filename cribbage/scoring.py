#!/usr/bin/env python3
from itertools import combinations

# ===============================================
# Helper Functions
# ===============================================
def get_value_of_card(CARD):
    val = CARD.split('_')
    if val[0] == 'A':
        return 1
    elif val[0] in ['J', 'Q', 'K']:
        return 10
    else:
        return int(val[0])

def get_order_of_card(CARD):
    val = CARD.split('_')
    if val[0] == 'A':
        return 1
    elif val[0] == 'J':
        return 11
    elif val[0] == 'Q':
        return 12
    elif val[0] == 'K':
        return 13
    else:
        return int(val[0])

def arr_in_order(arr):
    arr = sorted(get_order_of_card(c) for c in arr)
    count = 0
    for i in range(1, len(arr)):
        diff = arr[i] - arr[i-1]
        if diff == 1:
            count += 1
    return count+1 if count == len(arr) -1 else 0
# ===============================================
# Pegging Scoring Functions
# Checks the array for if anything has happened
# ===============================================

def peg_15(played_cards):
    total = 0
    for card in played_cards:
        total += get_value_of_card(card)
    return 2 if total == 15 else 0

def peg_31(played_cards):
    total = 0
    for card in played_cards:
        total += get_value_of_card(card)
    return 2 if total == 31 else 0

def peg_straights(played_cards):
    check_list = played_cards
    for i in range(len(played_cards)):
        if len(played_cards) - i < 3:
            return 0
        in_order = arr_in_order(check_list[i:len(played_cards)])
        if in_order > 0:
            return in_order
    return 0

def peg_duplicates(played_cards):
    check_list = [get_order_of_card(c) for c in played_cards]
    for i in range(len(played_cards)):
        if len(played_cards) - i < 2:
            return 0
        deduped_list = set(check_list[i:len(played_cards)])
        if len(deduped_list) == 1:
            n = len(played_cards) - i
            return (n - 1) * n
    return 0

def score_peg(played_cards):
    return peg_15(played_cards) + peg_31(played_cards) + peg_straights(played_cards) + peg_duplicates(played_cards)


# ===============================================
# Hand Scoring functions
# ===============================================

def hand_15s(hand, cut_card):
    hand = [get_value_of_card(c) for c in hand]
    hand.append(get_value_of_card(cut_card))
    count = 0
    for n in range(2, len(hand) + 1):
        for combo in combinations(hand, n):
            count += 2 if sum(combo) == 15 else 0
    return count

def hand_straights(hand, cut_card):
    cards_in_hand = list(hand) + [cut_card]
    for n in range(5, 2, -1):
        ways = 0
        for combo in combinations(cards_in_hand, n):
            if arr_in_order(combo) > 0:
                ways += 1
        if ways > 0:
            return n * ways
    return 0


def hand_duplicates(hand, cut_card):
    hand = [get_order_of_card(c) for c in hand]
    hand.append(get_order_of_card(cut_card))
    count = 0
    for combo in combinations(hand, 2):
        count += 2 if combo[0] == combo[1] else 0
    return count

def hand_flush(hand, cut_card, is_crib:bool):
    suit_of_card1 = hand[0].split("_")[2]
    suit_of_card2 = hand[1].split("_")[2]
    suit_of_card3 = hand[2].split("_")[2]
    suit_of_card4 = hand[3].split("_")[2]
    suit_of_cut = cut_card.split("_")[2]

    if (suit_of_card1 == suit_of_card2 == suit_of_card3 == suit_of_card4):
        if (suit_of_card1 == suit_of_cut):
            return 5
        return 0 if is_crib else 4
    return 0

def hand_Jack(hand, cut_card):
    return 1 if (f"J_of_{cut_card.split("_")[2]}") in hand else 0

def score_hand(hand, cut_card, is_crib):
    return hand_15s(hand, cut_card) + hand_straights(hand, cut_card) + hand_duplicates(hand, cut_card) + hand_flush(hand, cut_card, is_crib) + hand_Jack(hand, cut_card)

