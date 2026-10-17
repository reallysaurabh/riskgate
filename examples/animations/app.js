/* Interactive Canvas + Step-by-Step State Machine for RiskGate Examples */

const SCENARIOS = [
  {
    id: "cstr",
    index: "01 · CHEMICAL PROCESS",
    badge: "PROBE → CERTIFY",
    badgeClass: "badge-probe",
    title: "CSTR Exothermic Reactor",
    shortDesc: "Sticky coolant valve vs. drifting thermometer",
    meta: "α = 5.0% · Tube = 0.96 K · N_cal = 400",
    yLabel: "Reactor Temperature T (Kelvin)",
    yMin: 330,
    yMax: 412,
    upperLimit: { val: 405, label: "Runaway Limit (405 K)" },
    lowerLimit: { val: 335, label: "Quench Limit (335 K)" },
    hypotheses: [
      {
        id: "h1",
        name: "h₁: sticky_coolant_valve",
        score: "r = 0.61 ≤ τ",
        plausible: true,
        eli5: "Coolant valve is stuck halfway; reactor genuinely overheating.",
        math: "Residual r(h₁, x) = 0.61 ≤ τ (1.68); kept in Ĥ_τ."
      },
      {
        id: "h2",
        name: "h₂: temp_sensor_drift (+1.6 K/m)",
        score: "r = 0.94 ≤ τ",
        plausible: true,
        eli5: "Thermometer reads +16 K too high; actual reactor is normal (358 K).",
        math: "Residual r(h₂, x) = 0.94 ≤ τ (1.68); kept in Ĥ_τ."
      },
      {
        id: "h3",
        name: "h₃: feed_conc_spike",
        score: "r = 4.82 > τ",
        plausible: false,
        eli5: "Ruled out because jacket heat balance doesn't match.",
        math: "Residual r(h₃, x) = 4.82 > τ (1.68); excluded from Ĥ_τ."
      }
    ],
    candidates: [
      {
        rank: "#1",
        name: "coolant_sp(-8 K) [LLM Top-1]",
        statusAtStage3: "rejected",
        statusAtStage4: "rejected",
        margins: "h₁: +14.2 K | h₂: -4.8 K (QUENCH)",
        eli5: "Dangerous! If h₂ (sensor drift) is true, aggressive cooling freezes the batch below 335 K.",
        math: "s(a₁) = min(+14.2, -4.8) = -4.8 K < λ̂₁ (+1.4 K)"
      },
      {
        rank: "PROBE",
        name: "probe_backup_sensor_3min()",
        statusAtStage3: "probed",
        statusAtStage4: "probed",
        margins: "Safe under h₁ & h₂ · Resolves 100%",
        eli5: "3-minute safe test: cross-check spare thermometer & trim feed 10% to see which hypothesis is true.",
        math: "Admissible ∀h ∈ Ĥ_τ; separation ||y(h₁) - y(h₂)||_σ = 4.6 > 1.0"
      },
      {
        rank: "#2",
        name: "switch_to_backup_sensor()",
        statusAtStage3: "pending",
        statusAtStage4: "certified",
        margins: "Post-probe (h₂ confirmed): +18.2 K ≥ λ̂₂",
        eli5: "Once the 3-minute probe confirms thermometer drift (h₂), switching sensors is certified safe!",
        math: "Stage-2 score s(a₂) = +18.2 K ≥ λ̂₂ (+1.1 K) → CERTIFY"
      }
    ],
    stages: [
      {
        name: "1. Observe Alarm",
        eli5Title: "Step 1 · Alarm rings: Reactor thermometer climbs from 362 K to 388 K",
        eli5Body: "Over the last 10 minutes, the primary temperature readout has risen toward the 405 K thermal runaway limit. The AI agent is called to intervene.",
        mathTitle: "Stage 1 · Context Observation x_{t-10:t} & Safety Envelope S",
        mathBody: "Spec S = Envelope(T_true ∈ [335 K, 405 K]). At t = 0, measured T_meas(0) = 388 K, approaching upper limit 405 K."
      },
      {
        name: "2. Plausible Set Ĥ_τ",
        eli5Title: "Step 2 · Two different problems explain the exact same thermometer rise",
        eli5Body: "Instead of betting everything on the single most likely guess (h₁: sticky coolant valve), RiskGate keeps every fault hypothesis that matches the last 10 minutes of sensor data—including h₂ (a drifting thermometer).",
        mathTitle: "Stage 2 · Conformal Plausible Hypothesis Set Ĥ_τ(x) = {h : r(h, x) ≤ τ}",
        mathBody: "Calibrated on held-out episodes at δ = 0.05 (τ = 1.68), Ĥ_τ(x) retains {h₁: sticky_valve, h₂: sensor_drift} and excludes h₃ (r = 4.82 > τ). P(h* ∈ Ĥ_τ) ≥ 1 − δ."
      },
      {
        name: "3. Check Candidates",
        eli5Title: "Step 3 · RiskGate catches that the LLM's #1 action would quench the reactor under h₂",
        eli5Body: "The LLM wants to blast cooling (-8 K). Under h₁ that works. But under h₂ (drifting thermometer), the real reactor is only at 358 K—blasting cooling drives it below the 335 K quench limit! So RiskGate blocks #1.",
        mathTitle: "Stage 3 · Hypothesis-Conditioned Conformal Rollouts & Worst-Case Margin",
        mathBody: "With 0.96 K conformal tubes, m(a₁, h₁) = +14.2 K but m(a₁, h₂) = -4.8 K. Worst-case score s(a₁) = -4.8 K < λ̂₁ (+1.4 K), so a₁ cannot be certified."
      },
      {
        name: "4. Active Probe → Certify",
        eli5Title: "Step 4 · Verdict: Run a safe 3-minute probe, confirm h₂, then CERTIFY recovery!",
        eli5Body: "Because h₁ and h₂ disagree on which recovery is safe, RiskGate issues Verdict.PROBE (read spare sensor for 3 min). The probe proves the thermometer was drifting (h₂), shrinking Ĥ_τ to {h₂} and certifying switch_to_backup_sensor()!",
        mathTitle: "Stage 4 · Two-Stage CRC Probing (λ̂₁ at ρ·α, λ̂₂ at (1−ρ)·α)",
        mathBody: "Stage 1 returns Verdict.PROBE. After 3 min of probe observations, r(h₁) = 5.4 > τ while r(h₂) = 0.5 ≤ τ. Stage 2 certifies a₂ with m(a₂, h₂) = +18.2 K ≥ λ̂₂. Operator hand-overs drop from 27.8% to 12.2%."
      }
    ],
    verdictAtStage: [
      { cls: "ask", title: "MONITORING · Evaluating Alarm Context", reason: "Awaiting candidate evaluation across plausible hypotheses." },
      { cls: "ask", title: "AMBIGUOUS DIAGNOSIS · |Ĥ_τ| = 2 Plausible Faults", reason: "Both h₁ (sticky_valve) and h₂ (sensor_drift) are consistent with evidence." },
      { cls: "probe", title: "STAGE 1 VERDICT: PROBE (probe_backup_sensor_3min)", reason: "LLM Top-1 (coolant -8 K) is unsafe under h₂ (-4.8 K). Safe 3-min probe separates h₁ vs h₂." },
      { cls: "certify", title: "STAGE 2 VERDICT: CERTIFY (switch_to_backup_sensor)", reason: "3-min probe ruled out h₁. Action #2 certified with margin +18.2 K ≥ λ̂₂ (unsafe rate 4.8% ≤ α=5%)." }
    ]
  },
  {
    id: "k8s",
    index: "02 · CLOUD INFRASTRUCTURE",
    badge: "WALK-DOWN → CERTIFY #4",
    badgeClass: "badge-certify",
    title: "Kubernetes SRE Incident",
    shortDesc: "Bad canary deploy vs. database lock contention",
    meta: "α = 5.0% · Check Latency = 0.08 ms · N_cal = 500",
    yLabel: "Checkout API p95 Latency (ms)",
    yMin: 100,
    yMax: 1050,
    upperLimit: { val: 500, label: "SLA Ceiling (p95 ≤ 500 ms)" },
    lowerLimit: null,
    hypotheses: [
      {
        id: "h1",
        name: "h₁: bad_canary_deploy (v2.14)",
        score: "r = 0.62 ≤ τ",
        plausible: true,
        eli5: "New code deployed 9 minutes ago introduced a regression.",
        math: "r(h₁, x) = 0.62 ≤ τ (1.45); plausible."
      },
      {
        id: "h2",
        name: "h₂: db_lock_contention (db_cpu 91%)",
        score: "r = 0.89 ≤ τ",
        plausible: true,
        eli5: "Unindexed query is locking rows on the primary Postgres database.",
        math: "r(h₂, x) = 0.89 ≤ τ (1.45); plausible."
      },
      {
        id: "h3",
        name: "h₃: noisy_neighbor_cpu",
        score: "r = 1.31 ≤ τ",
        plausible: true,
        eli5: "Co-located batch job throttling node CPU.",
        math: "r(h₃, x) = 1.31 ≤ τ (1.45); plausible."
      }
    ],
    candidates: [
      {
        rank: "#1",
        name: "drop_prod_replica_table()",
        statusAtStage3: "rejected",
        statusAtStage4: "rejected",
        margins: "Blocked by HardRule('no_destructive_ddl')",
        eli5: "Deterministic hard rule immediately blocks destructive database commands.",
        math: "Screened out in O(1) by HardRule('no_destructive_ddl', Verdict.REJECT)."
      },
      {
        rank: "#2",
        name: "rollback_deploy(v2.13) [Top Viable]",
        statusAtStage3: "rejected",
        statusAtStage4: "rejected",
        margins: "h₁: +0.38 | h₂: -0.78 (SLA breach)",
        eli5: "Fixes a bad deploy (h₁), but does nothing if the database is locked (h₂: p95 stays at 890 ms).",
        math: "s(a₂) = min(+0.38, -0.78, -0.41) = -0.78 < λ̂ (+0.14)."
      },
      {
        rank: "#3",
        name: "scale_replicas(+4)",
        statusAtStage3: "rejected",
        statusAtStage4: "rejected",
        margins: "h₁: +0.15 | h₂: -1.12 (exhausts DB pool)",
        eli5: "Adding 4 more pods opens 80 more DB connections, crashing the locked database under h₂!",
        math: "s(a₃) = -1.12 < λ̂ (+0.14) (violates error_rate & p95 under h₂)."
      },
      {
        rank: "#4",
        name: "shed_noncritical_traffic(25%)",
        statusAtStage3: "certified",
        statusAtStage4: "certified",
        margins: "h₁: +0.34 | h₂: +0.28 | h₃: +0.31 ≥ λ̂",
        eli5: "Shedding 25% of background/analytics traffic relieves both pod CPU and DB locks—safe under all 3 causes!",
        math: "s(a₄) = min(+0.34, +0.28, +0.31) = +0.28 ≥ λ̂ (+0.14) → CERTIFY."
      }
    ],
    stages: [
      {
        name: "1. Incident Alert",
        eli5Title: "Step 1 · Pager fires: checkout-api p95 latency jumps to 920 ms (SLA ≤ 500 ms)",
        eli5Body: "Canary v2.14 rolled out 9 minutes ago, AND database CPU is at 91%. The SRE agent proposes a ranked list of 4 remediation actions.",
        mathTitle: "Stage 1 · Multi-Metric SLA Spec & Ranked Candidate List",
        mathBody: "Spec S = {p95_ms ≤ 500, error_rate ≤ 2%, avail ≥ 99.5%}. Agent proposes [a₁, a₂, a₃, a₄]."
      },
      {
        name: "2. Plausible Set Ĥ_τ",
        eli5Title: "Step 2 · Three root causes remain plausible given 1-minute telemetry",
        eli5Body: "Telemetry cannot yet separate a bad canary deploy (h₁), database lock contention (h₂), or CPU throttling (h₃). All three stay in Ĥ_τ.",
        mathTitle: "Stage 2 · Plausible Root-Cause Set Ĥ_τ(x) = {h₁, h₂, h₃}",
        mathBody: "All three root-cause models have residual score r(h, x) ≤ τ = 1.45."
      },
      {
        name: "3. Ranked Walk-Down",
        eli5Title: "Step 3 · RiskGate blocks #1 (hard rule), skips #2 and #3, and finds #4!",
        eli5Body: "Rolling back (#2) fails if the DB is locked (h₂). Scaling pods (#3) makes DB lock contention worse! Walking down the list, #4 (shedding 25% non-critical traffic) brings p95 below 360 ms under all three causes.",
        mathTitle: "Stage 3 · Policy-Aware Ranked Walk-Down Over Viable Candidates",
        mathBody: "a₁ blocked by HardRule. s(a₂) = -0.78 < λ̂; s(a₃) = -1.12 < λ̂; s(a₄) = +0.28 ≥ λ̂ (+0.14)."
      },
      {
        name: "4. Certify #4",
        eli5Title: "Step 4 · Verdict: CERTIFY #4 (shed_noncritical_traffic) in 0.08 ms",
        eli5Body: "Instead of paging a human on-call engineer (like a top-1 gate that only checks #2), RiskGate automatically certifies candidate #4 with a full contrastive explanation.",
        mathTitle: "Stage 4 · Certified Execution & Hash-Chained Audit Log",
        mathBody: "Verdict.CERTIFY on a₄ with worst-case margin +0.28 ≥ λ̂. Safe autonomous execution rises from 52.0% (top-1 CRC) to 84.4%."
      }
    ],
    verdictAtStage: [
      { cls: "ask", title: "INCIDENT ACTIVE · p95 = 920 ms (> 500 ms SLA)", reason: "Evaluating 4 ranked SRE agent proposals." },
      { cls: "ask", title: "ROOT-CAUSE AMBIGUITY · 3 Plausible Causes in Ĥ_τ", reason: "h₁ (canary), h₂ (db_lock), and h₃ (noisy_neighbor) all fit telemetry." },
      { cls: "certify", title: "WALK-DOWN MATCH · Candidate #4 Passes All h ∈ Ĥ_τ", reason: "Skipped #1 (HardRule), #2 (fails h₂), #3 (fails h₂). #4 has s = +0.28 ≥ λ̂." },
      { cls: "certify", title: "VERDICT: CERTIFY #4 · shed_noncritical_traffic(25%)", reason: "Predicted p95 drops to 340–365 ms across all h ∈ Ĥ_τ. Logged to HashChainAudit." }
    ]
  },
  {
    id: "refunds",
    index: "03 · FINTECH / E-COMMERCE",
    badge: "MULTI-RISK + PROBE",
    badgeClass: "badge-probe",
    title: "Customer Refund Agent",
    shortDesc: "Jointly bounding policy violations & dollar leakage",
    meta: "α_safety = 5% · α_cost = 3% · N_cal = 500",
    yLabel: "Expected Risk / Leakage Rate (%)",
    yMin: 0,
    yMax: 25,
    upperLimit: { val: 5.0, label: "Target Risk Budget α = 5.0%" },
    lowerLimit: null,
    hypotheses: [
      {
        id: "h1",
        name: "h₁: genuine_carrier_damage",
        score: "r = 0.45 ≤ τ",
        plausible: true,
        eli5: "Customer is telling the truth; parcel was damaged in transit.",
        math: "r(h₁, x) = 0.45 ≤ τ; full refund ($185) is policy-compliant."
      },
      {
        id: "h2",
        name: "h₂: opportunistic_wardrobing",
        score: "r = 1.05 ≤ τ",
        plausible: true,
        eli5: "Customer has 2 prior unverified claims and no photo attached.",
        math: "r(h₂, x) = 1.05 ≤ τ; unverified cash refund >$150 violates policy & leaks $185."
      }
    ],
    candidates: [
      {
        rank: "#1",
        name: "issue_instant_refund($185)",
        statusAtStage3: "rejected",
        statusAtStage4: "rejected",
        margins: "h₁: +0.45 | h₂: -0.52 (violates policy & cost)",
        eli5: "Instant $185 payout without photo violates compliance policy and loses $185 if h₂ is true.",
        math: "s(a₁) = -0.52 < λ̂ = max(λ̂_safety, λ̂_cost) = +0.19."
      },
      {
        rank: "PROBE",
        name: "request_photo_and_carrier_scan()",
        statusAtStage3: "probed",
        statusAtStage4: "probed",
        margins: "Zero dollar leakage · Separates h₁ vs h₂",
        eli5: "Ask customer for a photo & query carrier damage scan before releasing funds.",
        math: "Zero risk during probe; separates h₁ vs h₂ with 94% resolution rate."
      },
      {
        rank: "#2",
        name: "issue_verified_refund($185)",
        statusAtStage3: "pending",
        statusAtStage4: "certified",
        margins: "Post-probe (photo + carrier scan match h₁): +0.45 ≥ λ̂",
        eli5: "Once carrier scan & photo confirm damage (h₁), the $185 refund is automatically certified!",
        math: "Stage-2 score s(a₂) = +0.45 ≥ λ̂₂ (+0.19) → CERTIFY."
      }
    ],
    stages: [
      {
        name: "1. Refund Claim",
        eli5Title: "Step 1 · VIP customer requests an instant $185 refund ('box arrived crushed')",
        eli5Body: "The customer is VIP-tier, but has 2 prior claims in 90 days and didn't upload a photo. The support LLM wants to issue an instant $185 refund to keep CSAT high.",
        mathTitle: "Stage 1 · Joint Multi-Risk Specification (Safety + Dollar Leakage)",
        mathBody: "Gate controls two risks simultaneously via extra_losses: policy violation (α₁ = 5%) and net dollar overpay (α₂ = 3%)."
      },
      {
        name: "2. Plausible Set Ĥ_τ",
        eli5Title: "Step 2 · Both 'Genuine Damage' (h₁) and 'Opportunistic Abuse' (h₂) are plausible",
        eli5Body: "Without a photo or carrier scan, the evidence is consistent with both a damaged parcel (h₁) and policy abuse (h₂).",
        mathTitle: "Stage 2 · Plausible Customer-Intent Set Ĥ_τ(x) = {h₁, h₂}",
        mathBody: "Both h₁ (r = 0.45) and h₂ (r = 1.05) fall below τ = 1.38."
      },
      {
        name: "3. Multi-Risk Check",
        eli5Title: "Step 3 · Joint threshold λ̂ = max(λ̂_policy, λ̂_cost) blocks unverified $185 payout",
        eli5Body: "As the threshold λ sweeps left-to-right on the calibration curve, RiskGate picks λ̂ = +0.19 where BOTH policy risk ≤ 5% and dollar leakage ≤ 3% hold. Instant refund scores -0.52 < +0.19.",
        mathTitle: "Stage 3 · Multi-Risk Threshold Selection λ̂ = max_j λ̂_j",
        mathBody: "λ̂_policy = +0.08, λ̂_cost = +0.19 ⇒ λ̂ = +0.19. Candidate a₁ scores s(a₁) = -0.52 < +0.19."
      },
      {
        name: "4. Probe → Certify",
        eli5Title: "Step 4 · Verdict: Request photo + carrier scan (PROBE), then auto-certify $185!",
        eli5Body: "Instead of routing the ticket to a human agent queue, RiskGate triggers an automated probe (request photo + pull carrier scan). When the scan confirms damage, h₂ is eliminated and the $185 refund is certified.",
        mathTitle: "Stage 4 · Two-Stage Verification Resolves Epistemic Uncertainty",
        mathBody: "Probe eliminates h₂; Stage-2 check certifies a₂ with margin +0.45 ≥ λ̂₂. Human escalations drop by more than half while holding both risks below target."
      }
    ],
    verdictAtStage: [
      { cls: "ask", title: "CLAIM RECEIVED · $185 Instant Refund Proposed", reason: "Checking policy compliance (α₁=5%) and dollar leakage (α₂=3%)." },
      { cls: "ask", title: "INTENT AMBIGUITY · Ĥ_τ = {genuine_damage, opportunistic_abuse}", reason: "Missing defect photo and carrier weight/damage scan." },
      { cls: "probe", title: "STAGE 1 VERDICT: PROBE (request_photo_and_carrier_scan)", reason: "Instant refund violates cost/policy bound under h₂ (s = -0.52 < λ̂ = +0.19)." },
      { cls: "certify", title: "STAGE 2 VERDICT: CERTIFY ($185 Refund Approved Post-Probe)", reason: "Carrier scan + photo ruled out h₂. Certified with margin +0.45 ≥ λ̂₂." }
    ]
  },
  {
    id: "prop1",
    index: "04 · CORE THEORY (PROP. 1)",
    badge: "26.6 K vs 0.96 K",
    badgeClass: "badge-ask",
    title: "Why Top-1 Conformal Fails",
    shortDesc: "Visualizing Proposition 1 on the CSTR benchmark",
    meta: "Top-1 Tube = 26.6 K (0% cert) · Set Tube = 0.96 K (87.8% cert)",
    yLabel: "90% Conformal Uncertainty Tube Width (K)",
    yMin: 0,
    yMax: 30,
    upperLimit: { val: 15.0, label: "Max Usable Tube Before 100% Abstention" },
    lowerLimit: null,
    hypotheses: [
      {
        id: "h1",
        name: "Top-1 Calibration (uses only ĥ₁)",
        score: "Tube = 26.6 K",
        plausible: false,
        eli5: "When the top-1 guess is wrong ~25% of the time, its 90% error bar must swallow massive wrong-fault errors (26.6 K)!",
        math: "P(ĥ₁ ≠ h*) = 24% > α ⇒ (1−α) quantile is set by misdiagnosed episodes (26.6 K)."
      },
      {
        id: "h2",
        name: "RiskGate Set Calibration (uses Ĥ_τ)",
        score: "Tube = 0.96 K",
        plausible: true,
        eli5: "Because the true fault is inside the set Ĥ_τ 91.2% of the time, each hypothesis only needs a tiny 0.96 K simulation error bar!",
        math: "Conditioned on h ∈ Ĥ_τ, tube quantile reflects ODE integration noise only (0.96 K, 28× sharper)."
      }
    ],
    candidates: [
      {
        rank: "Baseline A",
        name: "Uncalibrated Twin Gate (m ≥ 0 on ĥ₁)",
        statusAtStage3: "rejected",
        statusAtStage4: "rejected",
        margins: "Unsafe = 44.6% (violates α = 5% by 9×!)",
        eli5: "Blindly trusts the top-1 simulation with zero safety margin—causes 44.6% thermal runaways!",
        math: "No conformal calibration: empirical unsafe rate = 44.6% ≫ α = 5%."
      },
      {
        rank: "Baseline B",
        name: "Top-1 Conformal Gate (26.6 K tube)",
        statusAtStage3: "probed",
        statusAtStage4: "probed",
        margins: "Unsafe = 0.0% | Certified = 0.0% (100% ASK)",
        eli5: "Safe only because its ±26.6 K error bar is wider than the reactor's safe operating window—so it paralyzes the plant (100% ASK)!",
        math: "By Proposition 1, λ̂_top1 = ∞ ⇒ 0% certification, 100% operator hand-over."
      },
      {
        rank: "RiskGate",
        name: "Hypothesis-Set RiskGate + Probe (0.96 K tube)",
        statusAtStage3: "certified",
        statusAtStage4: "certified",
        margins: "Unsafe = 4.8% ≤ α | Certified = 87.8% | Ask = 12.2%",
        eli5: "Sharp 0.96 K tube across plausible hypotheses + 3-min probing certifies 87.8% of episodes safely!",
        math: "Theorem 1: E[L(λ̂)] = 4.8% ≤ 5.0%, with 87.8% safe autonomous execution."
      }
    ],
    stages: [
      {
        name: "1. The Dilemma",
        eli5Title: "Step 1 · Why you cannot just slap conformal prediction onto a single best diagnosis",
        eli5Body: "In real incidents, the first 10 minutes of data rarely identify a single unique fault—the top-1 guess ĥ₁ is wrong ~24% of the time.",
        mathTitle: "Stage 1 · Epistemic Fault Ambiguity vs. Aleatoric Twin Error",
        mathBody: "Top-1 error P(ĥ₁(X) ≠ h*) = 0.24 exceeds target risk budget α = 0.05."
      },
      {
        name: "2. Top-1 Blowup",
        eli5Title: "Step 2 · Top-1 Conformal must inflate its error bar to ±26.6 K to cover wrong diagnoses",
        eli5Body: "To be right 90% of the time when the fault guess is wrong 24% of the time, the top-1 conformal tube balloons to ±26.6 K—wider than the reactor's entire safe margin!",
        mathTitle: "Stage 2 · Proposition 1 (Collapse of Single-Hypothesis Calibration)",
        mathBody: "Because P(ĥ₁ ≠ h* ∧ unsafe) > α, any threshold controlling risk at α must exceed all misdiagnosed scores, forcing q_top = 26.6 K and 0% certification."
      },
      {
        name: "3. Set Sharpness",
        eli5Title: "Step 3 · RiskGate separates 'which fault?' (Ĥ_τ) from 'how accurate is the simulator?' (0.96 K)",
        eli5Body: "By keeping the 2–3 plausible faults in a set Ĥ_τ, the true fault is inside the set >91% of the time. Now the simulator tube only needs ±0.96 K!",
        mathTitle: "Stage 3 · Hypothesis-Conditioned Tube Sharpness (28× Reduction)",
        mathBody: "On the event {h* ∈ Ĥ_τ}, trajectory error is bounded by q_set = 0.96 K (a 28× reduction from 26.6 K)."
      },
      {
        name: "4. Pareto Frontier",
        eli5Title: "Step 4 · Result: 4.8% unsafe (≤ 5% target) and 87.8% certified autonomous recovery",
        eli5Body: "Uncalibrated gates blow up the plant (44.6% unsafe). Top-1 conformal paralyzes the plant (0% certified). RiskGate + Probing hits the sweet spot: 4.8% unsafe and 87.8% certified.",
        mathTitle: "Stage 4 · Empirical Validity Across 500 Held-Out Test Episodes",
        mathBody: "Twin Gate: 44.6% unsafe. Top-1 CRC: 0% certified. RiskGate: 4.8% unsafe, 72.2% certified. RiskGate + Probe: 4.8% unsafe, 87.8% certified."
      }
    ],
    verdictAtStage: [
      { cls: "reject", title: "UNCALIBRATED TWIN GATE · 44.6% Unsafe Runaways", reason: "Trusting a point prediction under fault ambiguity violates safety 9× over budget." },
      { cls: "ask", title: "TOP-1 CONFORMAL · 26.6 K Tube → 0.0% Certified (100% ASK)", reason: "Proposition 1: top-1 misdiagnosis rate > α forces total abstention." },
      { cls: "certify", title: "RISKGATE HYPOTHESIS SET · 0.96 K Tube (28× Sharper)", reason: "Conditioning on Ĥ_τ decouples diagnostic ambiguity from simulator error." },
      { cls: "certify", title: "RISKGATE + PROBE · 4.8% Unsafe (≤ 5%) · 87.8% Certified", reason: "Achieves finite-sample risk control with minimal operator hand-overs (12.2%)." }
    ]
  }
];

let currentScenarioIdx = 0;
let currentStageIdx = 0;
let explainMode = "eli5"; // "eli5" | "math"
let isPlaying = true;
let stageProgress = 0; // 0.0 -> 1.0 within current stage
let lastFrameTime = performance.now();
const STAGE_DURATION_MS = 3600;

const canvas = document.getElementById("stage-canvas");
const ctx = canvas.getContext("2d");

function init() {
  renderScenarioTabs();
  bindControls();
  updateUI();
  requestAnimationFrame(tick);
}

function renderScenarioTabs() {
  const nav = document.getElementById("scenario-nav");
  nav.innerHTML = SCENARIOS.map((sc, idx) => `
    <button id="tab-${sc.id}" class="scenario-tab ${idx === currentScenarioIdx ? "active" : ""}" data-idx="${idx}">
      <div class="tab-index">
        <span>${sc.index}</span>
        <span class="tab-badge ${sc.badgeClass}">${sc.badge}</span>
      </div>
      <div class="tab-title">${sc.title}</div>
      <div class="tab-desc">${sc.shortDesc}</div>
    </button>
  `).join("");

  nav.querySelectorAll(".scenario-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentScenarioIdx = Number(btn.dataset.idx);
      currentStageIdx = 0;
      stageProgress = 0;
      renderScenarioTabs();
      updateUI();
    });
  });
}

function bindControls() {
  const playBtn = document.getElementById("btn-play-pause");
  playBtn.addEventListener("click", () => {
    isPlaying = !isPlaying;
    playBtn.textContent = isPlaying ? "❚❚ Pause" : "▶ Play";
  });

  document.getElementById("btn-prev").addEventListener("click", () => {
    currentStageIdx = (currentStageIdx + 3) % 4;
    stageProgress = 0.85;
    updateUI();
  });

  document.getElementById("btn-next").addEventListener("click", () => {
    currentStageIdx = (currentStageIdx + 1) % 4;
    stageProgress = 0.85;
    updateUI();
  });

  document.querySelectorAll(".stage-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      currentStageIdx = Number(pill.dataset.stage);
      stageProgress = 0.85;
      updateUI();
    });
  });

  document.getElementById("mode-eli5").addEventListener("click", () => {
    explainMode = "eli5";
    document.getElementById("mode-eli5").classList.add("active");
    document.getElementById("mode-math").classList.remove("active");
    updateUI();
  });

  document.getElementById("mode-math").addEventListener("click", () => {
    explainMode = "math";
    document.getElementById("mode-math").classList.add("active");
    document.getElementById("mode-eli5").classList.remove("active");
    updateUI();
  });
}

function updateUI() {
  const sc = SCENARIOS[currentScenarioIdx];
  const st = sc.stages[currentStageIdx];

  document.getElementById("canvas-title").textContent = `${sc.title} — Live Simulation`;
  document.getElementById("canvas-meta").textContent = sc.meta;

  // Stepper pills
  document.querySelectorAll(".stage-pill").forEach((pill, i) => {
    pill.classList.toggle("active", i === currentStageIdx);
    pill.classList.toggle("completed", i < currentStageIdx);
    pill.querySelector(".step-label").textContent = sc.stages[i].name;
  });

  // Narrative box
  document.getElementById("narrative-title").textContent =
    explainMode === "eli5" ? st.eli5Title : st.mathTitle;
  document.getElementById("narrative-body").textContent =
    explainMode === "eli5" ? st.eli5Body : st.mathBody;

  // Hypotheses list
  const hypContainer = document.getElementById("hyp-list");
  hypContainer.innerHTML = sc.hypotheses.map((h) => {
    const showFilter = currentStageIdx >= 1;
    const stateCls = !showFilter ? "" : h.plausible ? "plausible" : "excluded";
    const sub = explainMode === "eli5" ? h.eli5 : h.math;
    return `
      <div class="hyp-item ${stateCls}">
        <div class="hyp-row">
          <span class="mono-code">${h.name}</span>
          <span class="mono-code" style="color:${h.plausible ? "#15803D" : "#B91C1C"}">${h.score}</span>
        </div>
        <div class="item-sub">${sub}</div>
      </div>
    `;
  }).join("");

  // Candidates list
  const candContainer = document.getElementById("cand-list");
  candContainer.innerHTML = sc.candidates.map((c) => {
    let statusCls = "";
    if (currentStageIdx === 2) statusCls = c.statusAtStage3;
    if (currentStageIdx >= 3) statusCls = c.statusAtStage4;
    const sub = explainMode === "eli5" ? c.eli5 : c.math;
    return `
      <div class="cand-item ${statusCls}">
        <div class="cand-row">
          <span class="mono-code"><strong>${c.rank}</strong> · ${c.name}</span>
          <span class="mono-code" style="font-size:10.5px">${c.margins}</span>
        </div>
        <div class="item-sub">${sub}</div>
      </div>
    `;
  }).join("");

  // Verdict box
  const v = sc.verdictAtStage[currentStageIdx];
  const vbox = document.getElementById("verdict-box");
  vbox.className = `verdict-box ${v.cls}`;
  document.getElementById("verdict-title").textContent = v.title;
  document.getElementById("verdict-reason").textContent = v.reason;
}

function tick(now) {
  const dt = now - lastFrameTime;
  lastFrameTime = now;

  if (isPlaying) {
    stageProgress += dt / STAGE_DURATION_MS;
    if (stageProgress >= 1.0) {
      stageProgress = 0.0;
      currentStageIdx = (currentStageIdx + 1) % 4;
      updateUI();
    }
  }

  drawCanvas();
  requestAnimationFrame(tick);
}

function drawCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }
  ctx.save();
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const sc = SCENARIOS[currentScenarioIdx];
  if (sc.id === "cstr") {
    drawCSTR(ctx, W, H, currentStageIdx, stageProgress);
  } else if (sc.id === "k8s") {
    drawK8s(ctx, W, H, currentStageIdx, stageProgress);
  } else if (sc.id === "refunds") {
    drawRefunds(ctx, W, H, currentStageIdx, stageProgress);
  } else {
    drawProp1(ctx, W, H, currentStageIdx, stageProgress);
  }

  ctx.restore();
}

function drawCSTR(ctx, W, H, stage, prog) {
  const pad = { l: 54, r: 24, t: 26, b: 36 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const tx = (tMin) => pad.l + ((tMin + 10) / 30) * pw;
  const ty = (tempK) => pad.t + ph - ((tempK - 330) / (415 - 330)) * ph;

  // Safe envelope shading
  const yHi = ty(405);
  const yLo = ty(335);
  ctx.fillStyle = "rgba(185, 28, 28, 0.06)";
  ctx.fillRect(pad.l, pad.t, pw, yHi - pad.t);
  ctx.fillRect(pad.l, yLo, pw, pad.t + ph - yLo);
  ctx.fillStyle = "rgba(21, 128, 61, 0.05)";
  ctx.fillRect(pad.l, yHi, pw, yLo - yHi);

  // Limit lines
  ctx.strokeStyle = "#B91C1C";
  ctx.lineWidth = 1.3;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, yHi);
  ctx.lineTo(pad.l + pw, yHi);
  ctx.moveTo(pad.l, yLo);
  ctx.lineTo(pad.l + pw, yLo);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#B91C1C";
  ctx.font = "600 10.5px 'JetBrains Mono', monospace";
  ctx.fillText("Runaway 405 K", pad.l + 8, yHi - 5);
  ctx.fillText("Quench 335 K", pad.l + 8, yLo + 13);

  // t = 0 decision line
  const x0 = tx(0);
  ctx.strokeStyle = "#787672";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(x0, pad.t);
  ctx.lineTo(x0, pad.t + ph);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#52514E";
  ctx.fillText("t = 0 (alarm)", x0 - 68, pad.t + 13);

  // Y axis labels
  ctx.fillStyle = "#787672";
  ctx.font = "10px 'JetBrains Mono', monospace";
  [340, 360, 380, 400].forEach((k) => {
    ctx.fillText(`${k}K`, 12, ty(k) + 3);
  });

  // 1) Observed trajectory (-10 to 0 min)
  const obsEnd = stage === 0 ? Math.min(1, prog * 1.4) : 1;
  ctx.strokeStyle = "#18181B";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  for (let i = 0; i <= 40 * obsEnd; i++) {
    const frac = i / 40;
    const t = -10 + 10 * frac;
    const temp = 362 + 26 * Math.pow(frac, 1.25);
    const x = tx(t);
    const y = ty(temp);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Plausible hypothesis fan at stage >= 1
  if (stage >= 1) {
    // True temperature under h2 (sensor drift) was actually 358 K at t=0
    ctx.strokeStyle = "#1D4ED8";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    for (let i = 0; i <= 40; i++) {
      const frac = i / 40;
      const t = -10 + 10 * frac;
      const temp = 354 + 4 * frac;
      const x = tx(t);
      const y = ty(temp);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#1D4ED8";
    ctx.fillText("True T under h₂ (358K)", x0 - 132, ty(356) - 6);
  }

  // Stage >= 2: Rollouts of LLM Top-1 (coolant -8 K) under h1 vs h2
  if (stage >= 2) {
    const rollFrac = stage === 2 ? Math.min(1, prog * 1.4) : 1;
    // Under h1 (safe, stabilizes at 370 K)
    ctx.strokeStyle = "#686662";
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i <= 40 * rollFrac; i++) {
      const t = (i / 40) * 20;
      const temp = 388 - 18 * (1 - Math.exp(-t / 5));
      if (i === 0) ctx.moveTo(tx(t), ty(temp));
      else ctx.lineTo(tx(t), ty(temp));
    }
    ctx.stroke();

    // Under h2 (quenches below 335 K!)
    ctx.strokeStyle = "#B91C1C";
    ctx.lineWidth = 2.2;
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let i = 0; i <= 40 * rollFrac; i++) {
      const t = (i / 40) * 20;
      const temp = 358 - 28 * (1 - Math.exp(-t / 4.5));
      if (i === 0) ctx.moveTo(tx(t), ty(temp));
      else ctx.lineTo(tx(t), ty(temp));
    }
    ctx.stroke();

    if (rollFrac > 0.7) {
      ctx.fillStyle = "#686662";
      ctx.fillText("#1 | h₁ (370K OK)", tx(12), ty(372) - 5);
      ctx.fillStyle = "#B91C1C";
      ctx.fillText("#1 | h₂ (330K QUENCH ✗)", tx(10.5), ty(331) - 4);
    }
  }

  // Stage == 3: 3-minute probe band + certified recovery trajectory
  if (stage >= 3) {
    const probeFrac = Math.min(1, prog * 1.35);
    const xProbe = tx(3);
    ctx.fillStyle = "rgba(29, 78, 216, 0.09)";
    ctx.fillRect(x0, pad.t, xProbe - x0, ph);
    ctx.fillStyle = "#1D4ED8";
    ctx.fillText("3m Probe", x0 + 4, pad.t + 26);

    ctx.strokeStyle = "#15803D";
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    for (let i = 0; i <= 40 * probeFrac; i++) {
      const t = (i / 40) * 20;
      const temp = 388 - 20 * (1 - Math.exp(-t / 4.0));
      if (i === 0) ctx.moveTo(tx(t), ty(temp));
      else ctx.lineTo(tx(t), ty(temp));
    }
    ctx.stroke();

    if (probeFrac > 0.6) {
      ctx.fillStyle = "#15803D";
      ctx.fillText("✓ Probe + Certified Recovery (368K ± 0.96K)", tx(5.2), ty(366) - 8);
    }
  }
}

function drawK8s(ctx, W, H, stage, prog) {
  const pad = { l: 56, r: 24, t: 28, b: 36 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const tx = (tMin) => pad.l + ((tMin + 5) / 15) * pw;
  const ty = (ms) => pad.t + ph - ((ms - 100) / (1050 - 100)) * ph;

  const ySla = ty(500);
  ctx.fillStyle = "rgba(185, 28, 28, 0.06)";
  ctx.fillRect(pad.l, pad.t, pw, ySla - pad.t);
  ctx.fillStyle = "rgba(21, 128, 61, 0.05)";
  ctx.fillRect(pad.l, ySla, pw, pad.t + ph - ySla);

  ctx.strokeStyle = "#B91C1C";
  ctx.lineWidth = 1.3;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.l, ySla);
  ctx.lineTo(pad.l + pw, ySla);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#B91C1C";
  ctx.font = "600 10.5px 'JetBrains Mono', monospace";
  ctx.fillText("SLA Ceiling: p95 ≤ 500 ms", pad.l + 8, ySla - 6);

  // Y labels
  ctx.fillStyle = "#787672";
  ctx.font = "10px 'JetBrains Mono', monospace";
  [200, 400, 600, 800, 1000].forEach((ms) => {
    ctx.fillText(`${ms}`, 14, ty(ms) + 3);
  });

  const x0 = tx(0);
  ctx.strokeStyle = "#787672";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(x0, pad.t);
  ctx.lineTo(x0, pad.t + ph);
  ctx.stroke();
  ctx.setLineDash([]);

  // Incident spike (-5m to 0m): 210 ms -> 920 ms
  ctx.strokeStyle = "#18181B";
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  for (let i = 0; i <= 30; i++) {
    const frac = i / 30;
    const t = -5 + 5 * frac;
    const ms = 210 + 710 * Math.pow(frac, 1.8);
    if (i === 0) ctx.moveTo(tx(t), ty(ms));
    else ctx.lineTo(tx(t), ty(ms));
  }
  ctx.stroke();

  if (stage >= 2) {
    const f = stage === 2 ? Math.min(1, prog * 1.4) : 1;
    // Candidate #2 (rollback) under h2 (db_lock): stays at 890 ms!
    ctx.strokeStyle = "#B45309";
    ctx.lineWidth = 1.8;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i <= 30 * f; i++) {
      const t = (i / 30) * 10;
      const ms = 920 - 30 * (1 - Math.exp(-t / 2));
      if (i === 0) ctx.moveTo(tx(t), ty(ms));
      else ctx.lineTo(tx(t), ty(ms));
    }
    ctx.stroke();

    // Candidate #3 (scale +4) under h2: spikes to 1010 ms (connection pool exhausted)
    ctx.strokeStyle = "#B91C1C";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i <= 30 * f; i++) {
      const t = (i / 30) * 10;
      const ms = 920 + 95 * (1 - Math.exp(-t / 1.8));
      if (i === 0) ctx.moveTo(tx(t), ty(ms));
      else ctx.lineTo(tx(t), ty(ms));
    }
    ctx.stroke();
    ctx.setLineDash([]);

    if (f > 0.6) {
      ctx.fillStyle = "#B91C1C";
      ctx.fillText("#3 scale(+4) | h₂ (1015ms DB pool exhausted ✗)", tx(2.2), ty(1005) + 14);
      ctx.fillStyle = "#B45309";
      ctx.fillText("#2 rollback | h₂ (890ms SLA breach ✗)", tx(2.2), ty(885) - 5);
    }
  }

  if (stage >= 3) {
    const f = Math.min(1, prog * 1.35);
    // Candidate #4 (shed 25% non-critical traffic): drops cleanly to 345 ms under ALL h in H_tau!
    ctx.strokeStyle = "#15803D";
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    for (let i = 0; i <= 30 * f; i++) {
      const t = (i / 30) * 10;
      const ms = 345 + (920 - 345) * Math.exp(-t / 2.0);
      if (i === 0) ctx.moveTo(tx(t), ty(ms));
      else ctx.lineTo(tx(t), ty(ms));
    }
    ctx.stroke();

    if (f > 0.5) {
      ctx.fillStyle = "#15803D";
      ctx.fillText("✓ #4 shed_noncritical_25% (p95 = 345 ms ≤ 500 ms across h₁, h₂, h₃)", tx(1.5), ty(335) + 15);
    }
  }
}

function drawRefunds(ctx, W, H, stage, prog) {
  const pad = { l: 56, r: 28, t: 30, b: 40 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  // Plot calibration risk curves R_safety(lambda) and R_cost(lambda) vs threshold lambda in [-0.6, +0.6]
  const lx = (lam) => pad.l + ((lam + 0.6) / 1.2) * pw;
  const ry = (pct) => pad.t + ph - (pct / 24) * ph;

  // Target alpha lines (5% policy, 3% cost)
  const y5 = ry(5.0);
  const y3 = ry(3.0);
  ctx.strokeStyle = "#15803D";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(pad.l, y5);
  ctx.lineTo(pad.l + pw, y5);
  ctx.stroke();

  ctx.strokeStyle = "#1D4ED8";
  ctx.beginPath();
  ctx.moveTo(pad.l, y3);
  ctx.lineTo(pad.l + pw, y3);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.font = "600 10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#15803D";
  ctx.fillText("Policy target α₁ = 5.0%", pad.l + 8, y5 - 5);
  ctx.fillStyle = "#1D4ED8";
  ctx.fillText("Dollar leakage target α₂ = 3.0%", pad.l + 8, y3 - 5);

  // Draw monotone non-increasing risk curves
  // Curve 1: Policy risk R_1(lam) crosses 5% at lam = +0.08
  ctx.strokeStyle = "#15803D";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  for (let i = 0; i <= 60; i++) {
    const lam = -0.6 + (i / 60) * 1.2;
    const risk = 21 / (1 + Math.exp(3.7 * (lam + 0.22)));
    if (i === 0) ctx.moveTo(lx(lam), ry(risk));
    else ctx.lineTo(lx(lam), ry(risk));
  }
  ctx.stroke();

  // Curve 2: Cost risk R_2(lam) crosses 3% at lam = +0.19
  ctx.strokeStyle = "#1D4ED8";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  for (let i = 0; i <= 60; i++) {
    const lam = -0.6 + (i / 60) * 1.2;
    const risk = 19 / (1 + Math.exp(3.8 * (lam + 0.25)));
    if (i === 0) ctx.moveTo(lx(lam), ry(risk));
    else ctx.lineTo(lx(lam), ry(risk));
  }
  ctx.stroke();

  // X-axis labels
  ctx.fillStyle = "#787672";
  ctx.font = "10px 'JetBrains Mono', monospace";
  [-0.5, -0.25, 0.0, 0.19, 0.45].forEach((l) => {
    ctx.fillText(`λ=${l.toFixed(2)}`, lx(l) - 18, pad.t + ph + 18);
  });

  if (stage >= 2) {
    const xLam = lx(0.19);
    ctx.strokeStyle = "#18181B";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(xLam, pad.t);
    ctx.lineTo(xLam, pad.t + ph);
    ctx.stroke();
    ctx.fillStyle = "#18181B";
    ctx.fillText("Joint threshold λ̂ = max(+0.08, +0.19) = +0.19", xLam - 130, pad.t + 14);

    // Candidate #1 score (-0.52) on left (rejected)
    const xA1 = lx(-0.52);
    ctx.fillStyle = "#B91C1C";
    ctx.beginPath();
    ctx.arc(xA1, ry(16.0), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText("a₁: unverified $185 (s = -0.52 < λ̂)", xA1 + 8, ry(16.0) + 4);
  }

  if (stage >= 3) {
    // Post-probe score (+0.45 >= +0.19) on right (certified)
    const xA2 = lx(0.45);
    ctx.fillStyle = "#15803D";
    ctx.beginPath();
    ctx.arc(xA2, ry(1.2), 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText("✓ Post-probe a₂ (s = +0.45 ≥ λ̂)", xA2 - 165, ry(1.2) - 10);
  }
}

function drawProp1(ctx, W, H, stage, prog) {
  const pad = { l: 44, r: 32, t: 36, b: 36 };
  const pw = W - pad.l - pad.r;

  ctx.font = "600 11px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#18181B";
  ctx.fillText("Comparison on 500 Held-Out CSTR Fault Episodes (Target α = 5.0%)", pad.l, pad.t);

  const rows = [
    {
      label: "1. Uncalibrated Twin Gate (trusts ĥ₁ point prediction)",
      unsafe: 44.6,
      cert: 55.4,
      ask: 0.0,
      tube: "0.0 K (no UQ)",
      color: "#B91C1C"
    },
    {
      label: "2. Top-1 Conformal Gate (Proposition 1: 26.6 K tube)",
      unsafe: 0.0,
      cert: 0.0,
      ask: 100.0,
      tube: "26.6 K tube",
      color: "#B45309"
    },
    {
      label: "3. RiskGate Hypothesis Set Ĥ_τ (0.96 K tube)",
      unsafe: 4.8,
      cert: 72.2,
      ask: 27.8,
      tube: "0.96 K tube",
      color: "#15803D"
    },
    {
      label: "4. RiskGate + 3-Min Active Probe (0.96 K tube)",
      unsafe: 4.8,
      cert: 87.8,
      ask: 12.2,
      tube: "0.96 K + probe",
      color: "#1D4ED8"
    }
  ];

  rows.forEach((r, idx) => {
    const y = pad.t + 24 + idx * 62;
    const visible = idx <= stage;
    ctx.globalAlpha = visible ? 1.0 : 0.22;

    ctx.fillStyle = "#18181B";
    ctx.font = "600 11px 'Inter', sans-serif";
    ctx.fillText(r.label, pad.l, y);
    ctx.font = "10.5px 'JetBrains Mono', monospace";
    ctx.fillStyle = r.color;
    ctx.fillText(`${r.tube} · Unsafe: ${r.unsafe}% · Certified: ${r.cert}% · Ask: ${r.ask}%`, pad.l + pw - 340, y);

    // Stacked bar: Unsafe (red), Certified Safe (green), Ask (amber/grey)
    const barY = y + 8;
    const barH = 20;
    const animScale = idx === stage ? Math.min(1, 0.3 + prog * 0.9) : 1;

    const wUnsafe = (r.unsafe / 100) * pw * animScale;
    const wCert = (r.cert / 100) * pw * animScale;
    const wAsk = (r.ask / 100) * pw * animScale;

    ctx.fillStyle = "#ECEAE3";
    ctx.fillRect(pad.l, barY, pw, barH);

    ctx.fillStyle = "#B91C1C";
    ctx.fillRect(pad.l, barY, wUnsafe, barH);

    ctx.fillStyle = "#15803D";
    ctx.fillRect(pad.l + wUnsafe, barY, wCert, barH);

    ctx.fillStyle = "#D6D3C9";
    ctx.fillRect(pad.l + wUnsafe + wCert, barY, wAsk, barH);
  });
  ctx.globalAlpha = 1.0;
}

window.addEventListener("DOMContentLoaded", init);
