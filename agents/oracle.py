"""Oracle agent — Dijkstra-based optimal dispatcher for score calibration.

Dispatches the nearest *idle* ambulance (by true graph distance) to the
highest-priority unassigned emergency, then routes to the lowest-occupancy
hospital that has available capacity.

Used as the upper-bound baseline when populating README score tables.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from env.models import ActionModel, AmbulanceState, ObservationModel, Severity


_SEVERITY_RANK = {Severity.CRITICAL: 3, Severity.HIGH: 2, Severity.NORMAL: 1}


class OracleAgent:
    """Optimal-dispatch oracle using shortest-path look-ups."""

    def __init__(self, city_graph: Optional[nx.Graph] = None):
        """
        Args:
            city_graph: NetworkX graph of the city.  When provided, true
                        Dijkstra distances are used; otherwise Manhattan-style
                        |node_a - node_b| approximation is used as a fallback.
        """
        self._graph = city_graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, observation: ObservationModel) -> ActionModel:
        """Return the optimal dispatch action for the current observation."""
        idle_ambs = [a for a in observation.ambulances if a.state == AmbulanceState.IDLE]
        unassigned = [e for e in observation.emergencies if not e.assigned]

        if not idle_ambs or not unassigned:
            return ActionModel(ambulance_id=None, emergency_id="", hospital_id=None, is_noop=True)

        # 1. Pick the highest-priority emergency (break ties by earliest deadline)
        target_emg = max(
            unassigned,
            key=lambda e: (_SEVERITY_RANK.get(e.severity, 0), -e.time_remaining),
        )

        # 2. Pick the ambulance closest to that emergency
        best_amb = min(
            idle_ambs,
            key=lambda a: self._dist(a.node, target_emg.node),
        )

        # 3. Pick the hospital with most remaining capacity (ties broken by proximity)
        available_hosps = [h for h in observation.hospitals if h.current_patients < h.capacity]
        if not available_hosps:
            available_hosps = list(observation.hospitals)  # no choice — accept overflow

        best_hosp = min(
            available_hosps,
            key=lambda h: (h.current_patients, self._dist(target_emg.node, h.node)),
        )

        return ActionModel(
            ambulance_id=best_amb.id,
            emergency_id=target_emg.id,
            hospital_id=best_hosp.id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dist(self, src: int, dst: int) -> float:
        if src == dst:
            return 0.0
        if self._graph is not None:
            try:
                return float(nx.shortest_path_length(self._graph, src, dst))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        # Fallback: numeric node label difference (works for grid-like graphs)
        return float(abs(src - dst))
