# Experiments log (real runs only)
* Baseline validation: 3 centralized baselines complete all easy scenarios; scenario tuned to 4 tasks/UAV, slack 60–150 s so methods differ.
* BC: 14k demos (8 episodes) → 99.8% agreement with local_greedy.
* Run 1 (`results/old_run`): lr 3e-4, no critic warm-up: validation return fell from 21.0 to ≈16–18 → degraded BC policy.
* Run 2 (`m2`): lr 1e-4, 3 critic-only iterations, BC 4 episodes; stopped at iteration 10 (compute). Entropy ≈0.005, KL≈0.
* Evaluation: 4 test seeds per condition, 3 per fleet size; see README for numbers and caveats.
Reproduce longer: `python scripts/train.py --config configs/training.yaml -o train.bc.enabled=true -o train.iterations=200 -o train.ent_coef=0.05`.
Ablations: add `-o model.actor_type=mlp`, `-o model.aggregation=mean`, `-o train.algo=ippo`, `-o model.recurrent=true` (training);
`-o coordination.enabled=false`, `-o safety.energy_filter=false`, `-o comm.range=200` (evaluation; simulation only).
