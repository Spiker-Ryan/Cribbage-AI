#!/usr/bin/env python3

import random

def gen_deck():
    SUITS = ['S', 'H', 'D', 'C']
    VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

    # empty array for all cards
    CARDS = []

    for suit in SUITS:
        for val in VALUES:
            CARDS.append(f'{val}_of_{suit}')

    return CARDS

# Generates a new shuffled deck
def gen_new_hands():
    #define all the base values a card could be
    CARDS = gen_deck()

    random.shuffle(CARDS)
    random.shuffle(CARDS)

    dealer_hand = []
    other_hand = []

    for i in range(0, 12, 2):
        other_hand.append(CARDS[i])
        dealer_hand.append(CARDS[i+1])

    cut_card = cut_deck(CARDS)

    return dealer_hand, other_hand, cut_card

# Cut the deck to find the cut card
def cut_deck(DECK):
    cut_pos = random.randrange(len(DECK))
    return DECK[cut_pos]

def gen_one_hand():
    #define all the base values a card could be
    CARDS = gen_deck()

    random.shuffle(CARDS)
    random.shuffle(CARDS)

    dealer_hand = []

    for i in range(0, 12, 2):
        dealer_hand.append(CARDS[i+1])

    return dealer_hand, list(set(CARDS) - set(dealer_hand))