# Architecture
See README (diagrams, schemas). Information flow: simulator truth → network → each UAV's `LocalReservationTable` (the only source
for its observation) → `GraphBuilder` → safety mask → actor → protocol → simulator. The critic reads `build_global_state` only inside
`algorithms/mappo.py`. Step order and every simplification are documented in `environment/simulator.py` and `environment/dynamics.py`.
Reservation validity rules: `coordination/reservations.py` docstring. Total order: `coordination/conflict_resolution.py`.
