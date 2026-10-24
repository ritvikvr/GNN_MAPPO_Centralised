from .greedy_auction import GreedyAuction
from .hungarian import HungarianAssignment
from .nearest import NearestFeasible

CENTRALIZED = {"nearest": NearestFeasible, "greedy": GreedyAuction, "hungarian": HungarianAssignment}
