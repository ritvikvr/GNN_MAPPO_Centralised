# Mathematical formulation
Decision problem: maximise E[ Σ_j v_j·1(task j completed before d_j) ] − energy and time costs, s.t. for every commitment of UAV i to
task j: e_i − (1+m)(c·‖x_i−x_j‖ + c_s·s_j + c·‖x_j−b*‖) ≥ ρ·E_i (reserve ρ), capability/payload match, arrival+ETA(1+m_t)+s_j ≤ d_j, geofence.
Dec-POMDP with local graph observation G_t^i=(V,E,X,R); policy π_θ(a|G_t^i) shared; centralized value V_φ(s_t).
PPO: L=−E[min(rA, clip(r,1±ε)A)] − c_e H + c_v (V_φ−R)²; A from GAE(γ,λ) over the shared team reward
r_t = w_c ΔV/v̄ − w_e ΔE/E_cap − w_t Δt/T·1[open] − w_l ExpV/v̄ − w_s n_switch − w_d n_dup − w_q kB/(N).
Attention: e_ds = aᵀLeakyReLU(W_l h_s + W_r h_d + W_e e_ds); α = softmax_s; m_d = Σ α(W_l h_s + W_e e_ds).
Bid = priority·urgency / ((ETA + s/2 + 5)(1 + 0.5 c·dist/E_cap)); claim rank = (phase, round(bid,6), −uav_id).
