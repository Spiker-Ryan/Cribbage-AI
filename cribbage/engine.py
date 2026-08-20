"""Game state and the play loop. Pure rules - no printing, no input().

Needed for Stage 2 onward. Holds the game/round/pegging loops and whatever
represents a position (scores, dealer, hands, crib, cut, pile, count).
"""
from random import shuffle, randrange

from cribbage import cards, scoring


class GameOver(Exception):
    """Raised by award() the moment someone reaches 121.

    Cribbage ends mid-count - if pone crosses while counting their hand, the
    dealer never counts theirs. Rather than checking a flag after every one
    of the dozen places points get awarded (his heels, each peg, each go,
    three separate counts), this unwinds straight out to play_game.
    """


class CribbageGame():
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.scores = [0, 0]
        self.crib = []
        self.hands = [[], []]
        self.cut = ""
        self.deck = []
        self.dealer = 0
        self.win = False
        self.winner = None

        # class variables for pegging
        self.played_stack = []
        self.running_count = 0

    def award(self, player_idx: int, points: int):
        self.scores[player_idx] += points
        if self.scores[player_idx] >= 121:
            self.win = True
            self.winner = player_idx
            raise GameOver

    def deal(self):
        self.deck = cards.gen_deck()
        shuffle(self.deck)

        self.hands = [[], []]

        for i in range(0, 12, 2):
            self.hands[1- self.dealer].append(self.deck[i])
            self.hands[self.dealer].append(self.deck[i+1])

        self.deck = self.deck[12:] # remove the cards we dealt from the deck

    def collect_crib_cards(self):
        self.crib = []

        for idx in (0, 1):
            hand = self.hands[idx]
            # hand out a copy - a player that mutates what it is given would
            # silently corrupt the game rather than failing loudly
            thrown = self.players[idx].choose_discard(list(hand), idx == self.dealer)

            if len(thrown) != 2:
                raise ValueError(f"player {idx} discarded {len(thrown)} cards, expected 2")
            for card in thrown:
                if card not in hand:
                    raise ValueError(f"player {idx} discarded {card}, which is not in their hand")

            for card in thrown:
                hand.remove(card)
            self.crib.extend(thrown)

    def cut_card(self):
        self.cut = self.deck[randrange(len(self.deck))]
        if "J" in self.cut:
            self.award(self.dealer, 2)

    def play_legal(self, card, count):
        return count + scoring.get_value_of_card(card) <= 31

    def play_pegging(self):
        """Pegging phase. Points are awarded through award(), so a player
        reaching 121 mid-pegging raises GameOver and unwinds out of here.

        A "segment" is one run of play up to 31 or until nobody can go.
        played_stack holds only the current segment - the peg scorers read
        whatever list they are handed, so a stale stack would find runs
        spanning a reset.
        """
        # deep copy: a shallow one shares the inner lists, and pegging would
        # empty the real hands before count_cards ever sees them
        hands = [list(h) for h in self.hands]

        turn = 1 - self.dealer      # pone always leads
        last_player = None          # who played the most recent card
        self.played_stack = []
        self.running_count = 0

        # every card played this round by either side, in order. Unlike
        # played_stack this survives segment resets - a player reasoning
        # about what the opponent still holds needs the whole round.
        self.round_history = []

        while hands[0] or hands[1]:
            legal = [c for c in hands[turn] if self.play_legal(c, self.running_count)]

            if not legal:
                # this player says "go". If the other can still play, they
                # continue - turns are not a strict alternation.
                other = 1 - turn
                if any(self.play_legal(c, self.running_count) for c in hands[other]):
                    turn = other
                    continue

                # neither can play: segment over. Whoever played last takes 1.
                # (31 is not handled here - peg_31 already paid 2 for it.)
                self.award(last_player, 1)
                self.played_stack = []
                self.running_count = 0
                turn = 1 - last_player
                continue

            # everything the acting player is entitled to know. Deliberately
            # excludes the opponent's cards, which this method is holding in
            # `hands` - handing over the whole game object would leak them.
            state = {
                "player": turn,
                "is_dealer": turn == self.dealer,
                "my_hand": list(hands[turn]),
                "cut": self.cut,
                "count": self.running_count,
                "pile": list(self.played_stack),
                "history": [list(p) for p in self.round_history],
            }

            card = self.players[turn].choose_play(list(legal), state)
            if card not in legal:
                raise ValueError(f"player {turn} played {card}, which is not a legal play")

            hands[turn].remove(card)
            self.round_history.append([turn, card])
            self.played_stack.append(card)
            self.running_count += scoring.get_value_of_card(card)
            last_player = turn

            points = (scoring.peg_15(self.played_stack)
                      + scoring.peg_31(self.played_stack)
                      + scoring.peg_straights(self.played_stack)
                      + scoring.peg_duplicates(self.played_stack))
            if points:
                self.award(turn, points)

            if self.running_count == 31:
                self.played_stack = []
                self.running_count = 0
                turn = 1 - last_player
            else:
                turn = 1 - turn

        # the very last card of the round scores 1, unless it landed on 31
        # (in which case the reset above already zeroed the count)
        if self.running_count > 0:
            self.award(last_player, 1)



    def count_cards(self):
        # order matters: if pone crosses 121 here, the dealer never counts.
        # award() raises GameOver, so no checks are needed between these.
        pone = 1 - self.dealer

        self.award(pone, scoring.score_hand(self.hands[pone], self.cut, False))
        self.award(self.dealer, scoring.score_hand(self.hands[self.dealer], self.cut, False))
        self.award(self.dealer, scoring.score_hand(self.crib, self.cut, True))

    def play_round(self):
        self.deal()
        self.collect_crib_cards()
        self.cut_card()
        self.play_pegging()
        self.count_cards()

    def play_game(self):
        self.dealer = randrange(2)

        # the only place GameOver is caught - it unwinds out of whatever
        # phase, loop, or count was in progress when 121 was reached
        while not self.win:
            try:
                self.play_round()
            except GameOver:
                break
            self.dealer = 1 - self.dealer

        return self.winner



