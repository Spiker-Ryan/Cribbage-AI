"""nn.Module definitions.

Stage 1: discard value net (MLP, scalar output).
Stage 2: opponent model. Stage 3: pegging net.
"""
from ml.features import DISCARD_FEATURE_SIZE, PEGGING_FEATURE_SIZE
import torch
import torch.nn as nn

class DiscardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DISCARD_FEATURE_SIZE, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DiscardCribNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DISCARD_FEATURE_SIZE, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2) # Predicting 2 because outputs are hand EV and crib EV

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PeggingNet(nn.Module):
    """Value net over (state, candidate card) -> expected point differential.

    Run it once per legal card and take the best. Framing it this way means
    illegal moves are simply never scored, so there is no action mask.

    Wider than the discard nets because the input is six times larger and
    the target is far noisier - pegging labels come from real games, not
    from enumeration.

    Two outputs: points I go on to take, and points I hand over. Their
    difference is the decision value, but keeping them apart says whether a
    bad prediction came from overvaluing offence or underestimating what it
    gives away - the same diagnostic that made hand_ev/crib_ev worth
    splitting.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(PEGGING_FEATURE_SIZE, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    @staticmethod
    def differential(out):
        """(batch, 2) -> (batch,) - my points minus theirs."""
        return out[:, 0] - out[:, 1]
