"""Safe fallback behaviour: always keep at least one safe action available."""
from __future__ import annotations

from typing import Any

from ..environment.uav import Action, RETURN
from .feasibility import FeasibilityChecker

EMERGENCY = "EMERGENCY"


def safe_fallback(uav: Any, checker: FeasibilityChecker, emergency_policy: str = "land_in_place") -> Action | str:
    """Return RETURN if the recovery point is reachable, WAIT (hold) if energy remains for hovering,
    otherwise the configured emergency policy marker (never an invented feasible return)."""
    if checker.can_return(uav):
        return RETURN
    if uav.energy > 0.0 and emergency_policy == "continue_to_base":
        return RETURN        # explicit, configured policy: try anyway
    if uav.energy > 0.0 and emergency_policy == "land_in_place":
        return EMERGENCY
    return EMERGENCY


def action_availability(uav: Any, checker: FeasibilityChecker, has_commitment: bool) -> dict[str, bool]:
    """Which *special* actions are permitted for this UAV right now."""
    forced = checker.must_return(uav)
    return {
        "wait": not forced,
        "return": checker.can_return(uav) or checker.safety.emergency_policy == "continue_to_base",
        "continue": has_commitment and not forced,
        "forced_return": forced,
    }
