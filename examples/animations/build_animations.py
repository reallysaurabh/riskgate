"""Generate minimal, Swiss-editorial animated SVGs explaining RiskGate and its 3 domain examples."""
# ruff: noqa: E501
from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent

COMMON_STYLE = """
  <style>
    .bg { fill: #FAF9F5; }
    .panel { fill: #FFFFFF; stroke: #E4E2DD; stroke-width: 1.2; }
    .panel-subtle { fill: #F4F2EC; stroke: #E4E2DD; stroke-width: 1; }
    .ink { fill: #18181B; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    .muted { fill: #686662; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    .mono { font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace; fill: #18181B; }
    .mono-muted { font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace; fill: #686662; }
    .title { font-size: 16px; font-weight: 600; letter-spacing: -0.02em; }
    .subtitle { font-size: 11.5px; font-weight: 400; }
    .label { font-size: 11px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
    .small { font-size: 11px; }
    .tiny { font-size: 10px; }
    .axis { stroke: #D4D1C8; stroke-width: 1; }
    .grid { stroke: #ECEAE3; stroke-width: 1; stroke-dasharray: 3 3; }
    .limit-line { stroke: #B91C1C; stroke-width: 1.4; stroke-dasharray: 5 4; }
    .thresh-line { stroke: #18181B; stroke-width: 1.4; stroke-dasharray: 4 3; }
    .safe-fill { fill: #15803D; fill-opacity: 0.06; }
    .danger-fill { fill: #B91C1C; fill-opacity: 0.07; }

    /* Animated stages over a 10s loop */
    @keyframes fadeStage1 {
      0%, 5% { opacity: 0.25; }
      10%, 92% { opacity: 1; }
      98%, 100% { opacity: 0.25; }
    }
    @keyframes fadeStage2 {
      0%, 22% { opacity: 0.15; }
      28%, 92% { opacity: 1; }
      98%, 100% { opacity: 0.15; }
    }
    @keyframes fadeStage3 {
      0%, 44% { opacity: 0.12; }
      52%, 92% { opacity: 1; }
      98%, 100% { opacity: 0.12; }
    }
    @keyframes fadeStage4 {
      0%, 66% { opacity: 0.1; transform: translateY(2px); }
      74%, 92% { opacity: 1; transform: translateY(0px); }
      98%, 100% { opacity: 0.1; transform: translateY(2px); }
    }
    @keyframes drawCurve {
      0%, 25% { stroke-dashoffset: 520; }
      55%, 92% { stroke-dashoffset: 0; }
      98%, 100% { stroke-dashoffset: 520; }
    }
    @keyframes drawProbeCurve {
      0%, 52% { stroke-dashoffset: 420; opacity: 0; }
      56% { opacity: 1; }
      80%, 92% { stroke-dashoffset: 0; opacity: 1; }
      98%, 100% { stroke-dashoffset: 420; opacity: 0; }
    }
    @keyframes pulseDot {
      0%, 100% { r: 4; opacity: 0.9; }
      50% { r: 6.5; opacity: 1; }
    }
    @keyframes progressSweep {
      0% { width: 0px; }
      95% { width: 820px; }
      100% { width: 0px; }
    }
    @keyframes scanHighlight1 {
      0%, 24% { opacity: 0; }
      28%, 50% { opacity: 1; }
      55%, 100% { opacity: 0.35; }
    }
    @keyframes scanHighlight2 {
      0%, 50% { opacity: 0.35; }
      56%, 92% { opacity: 1; }
      98%, 100% { opacity: 0.35; }
    }
    .st1 { animation: fadeStage1 10s infinite ease-in-out; }
    .st2 { animation: fadeStage2 10s infinite ease-in-out; }
    .st3 { animation: fadeStage3 10s infinite ease-in-out; }
    .st4 { animation: fadeStage4 10s infinite ease-in-out; }
    .curve-anim { stroke-dasharray: 520; animation: drawCurve 10s infinite ease-in-out; }
    .probe-anim { stroke-dasharray: 420; animation: drawProbeCurve 10s infinite ease-in-out; }
    .pulse { animation: pulseDot 2s infinite ease-in-out; }
    .progress-bar { animation: progressSweep 10s infinite linear; }
    .scan1 { animation: scanHighlight1 10s infinite ease-in-out; }
    .scan2 { animation: scanHighlight2 10s infinite ease-in-out; }
  </style>
"""


def build_1_pipeline_concept() -> str:
    """Animation 1: The core RiskGate 4-verdict verification pipeline."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 390" width="100%" height="100%">
  {COMMON_STYLE}
  <rect class="bg" width="880" height="390" rx="10"/>

  <!-- Top header -->
  <text x="30" y="34" class="ink title">RiskGate · Consequence-Based Verification Loop</text>
  <text x="30" y="52" class="muted subtitle">Every tool call is simulated across plausible world hypotheses before reaching the real system (E[certified ∧ unsafe] ≤ α)</text>
  <text x="850" y="34" text-anchor="end" class="mono-muted tiny">10s LOOP · 4 STAGES</text>

  <!-- Timeline progress bar -->
  <rect x="30" y="64" width="820" height="3" fill="#E4E2DD" rx="1.5"/>
  <rect x="30" y="64" height="3" fill="#18181B" rx="1.5" class="progress-bar"/>

  <!-- Stage 1: Doer (LLM Agent) -->
  <g class="st1">
    <rect x="30" y="84" width="185" height="250" rx="8" class="panel"/>
    <text x="46" y="108" class="muted label">01 · Doer (LLM Agent)</text>
    <text x="46" y="128" class="ink small">Proposes ranked candidates</text>

    <!-- Candidate 1 (Rejected) -->
    <g class="scan1">
      <rect x="44" y="144" width="157" height="48" rx="6" fill="#FEF2F2" stroke="#FECACA" stroke-width="1"/>
      <text x="54" y="163" class="mono tiny" fill="#991B1B">#1 aggressive_action()</text>
      <text x="54" y="180" class="muted tiny">High reward, risky under h₂</text>
    </g>

    <!-- Candidate 2 (Certified) -->
    <g class="scan2">
      <rect x="44" y="202" width="157" height="48" rx="6" fill="#F0FDF4" stroke="#BBF7D0" stroke-width="1.2"/>
      <text x="54" y="221" class="mono tiny" fill="#166534">#2 conservative_mitigate()</text>
      <text x="54" y="238" class="muted tiny">Safe under all h ∈ Ĥ_τ</text>
    </g>

    <!-- Candidate 3 -->
    <rect x="44" y="260" width="157" height="48" rx="6" class="panel-subtle"/>
    <text x="54" y="279" class="mono-muted tiny">#3 fallback_hold()</text>
    <text x="54" y="296" class="muted tiny">Backup option</text>
  </g>

  <!-- Arrow 1 -> 2 -->
  <line x1="215" y1="208" x2="245" y2="208" stroke="#18181B" stroke-width="1.4"/>
  <polygon points="245,208 238,204 238,212" fill="#18181B"/>

  <!-- Stage 2: Conformal Hypothesis Set -->
  <g class="st2">
    <rect x="248" y="84" width="196" height="250" rx="8" class="panel"/>
    <text x="264" y="108" class="muted label">02 · Plausible Set Ĥ_τ(x)</text>
    <text x="264" y="128" class="ink small">Keep all h with r(h, x) ≤ τ</text>

    <!-- H1 -->
    <rect x="262" y="144" width="168" height="42" rx="5" fill="#F8FAFC" stroke="#CBD5E1"/>
    <text x="272" y="162" class="mono tiny">h₁: primary_fault</text>
    <text x="420" y="162" text-anchor="end" class="mono tiny" fill="#15803D">r=0.4 ≤ τ ✓</text>
    <text x="272" y="177" class="muted tiny">Plausible (top-1 guess)</text>

    <!-- H2 -->
    <rect x="262" y="196" width="168" height="42" rx="5" fill="#F8FAFC" stroke="#CBD5E1"/>
    <text x="272" y="214" class="mono tiny">h₂: sensor_or_db_fault</text>
    <text x="420" y="214" text-anchor="end" class="mono tiny" fill="#15803D">r=1.1 ≤ τ ✓</text>
    <text x="272" y="229" class="muted tiny">Also fits recent evidence!</text>

    <!-- H3 (Excluded) -->
    <rect x="262" y="248" width="168" height="42" rx="5" fill="#FAF9F5" stroke="#E4E2DD" opacity="0.55"/>
    <text x="272" y="266" class="mono-muted tiny">h₃: pipe_rupture</text>
    <text x="420" y="266" text-anchor="end" class="mono tiny" fill="#B91C1C">r=6.8 &gt; τ ✗</text>
    <text x="272" y="281" class="muted tiny">Ruled out by evidence</text>

    <text x="264" y="316" class="mono-muted tiny">P(h* ∈ Ĥ_τ) ≥ 1 − δ (95%)</text>
  </g>

  <!-- Arrow 2 -> 3 -->
  <line x1="444" y1="208" x2="474" y2="208" stroke="#18181B" stroke-width="1.4"/>
  <polygon points="474,208 467,204 467,212" fill="#18181B"/>

  <!-- Stage 3: Worst-Case Margin Matrix -->
  <g class="st3">
    <rect x="477" y="84" width="195" height="250" rx="8" class="panel"/>
    <text x="493" y="108" class="muted label">03 · Margin Check</text>
    <text x="493" y="128" class="ink small">s(a) = min_{{h ∈ Ĥ_τ}} m(a, h)</text>

    <!-- Candidate #1 check -->
    <rect x="491" y="144" width="167" height="68" rx="6" fill="#FEF2F2" stroke="#FECACA"/>
    <text x="501" y="162" class="mono tiny">Candidate #1</text>
    <text x="501" y="179" class="mono tiny" fill="#15803D">m(#1, h₁) = +0.42 (safe)</text>
    <text x="501" y="194" class="mono tiny" fill="#B91C1C">m(#1, h₂) = -0.38 (UNSAFE)</text>
    <text x="648" y="162" text-anchor="end" class="mono tiny" fill="#B91C1C">s = -0.38 &lt; λ̂</text>

    <!-- Candidate #2 check -->
    <rect x="491" y="222" width="167" height="68" rx="6" fill="#F0FDF4" stroke="#BBF7D0"/>
    <text x="501" y="240" class="mono tiny">Candidate #2</text>
    <text x="501" y="257" class="mono tiny" fill="#15803D">m(#2, h₁) = +0.31 (safe)</text>
    <text x="501" y="272" class="mono tiny" fill="#15803D">m(#2, h₂) = +0.24 (safe)</text>
    <text x="648" y="240" text-anchor="end" class="mono tiny" fill="#15803D">s = +0.24 ≥ λ̂</text>

    <text x="493" y="316" class="mono-muted tiny">Calibrated threshold λ̂ = +0.12</text>
  </g>

  <!-- Arrow 3 -> 4 -->
  <line x1="672" y1="208" x2="702" y2="208" stroke="#18181B" stroke-width="1.4"/>
  <polygon points="702,208 695,204 695,212" fill="#18181B"/>

  <!-- Stage 4: Four Verdicts -->
  <g class="st4">
    <rect x="705" y="84" width="145" height="250" rx="8" class="panel"/>
    <text x="720" y="108" class="muted label">04 · Verdict</text>

    <!-- CERTIFY (active) -->
    <rect x="717" y="124" width="121" height="48" rx="6" fill="#15803D"/>
    <text x="729" y="144" class="mono tiny" fill="#FFFFFF" font-weight="600">✓ CERTIFY #2</text>
    <text x="729" y="160" class="tiny" fill="#DCFCE7">Execute action #2</text>

    <!-- PROBE -->
    <rect x="717" y="180" width="121" height="44" rx="6" fill="#EFF6FF" stroke="#BFDBFE"/>
    <text x="729" y="199" class="mono tiny" fill="#1D4ED8">◉ PROBE</text>
    <text x="729" y="214" class="muted tiny">Safe diagnostic test</text>

    <!-- ASK -->
    <rect x="717" y="232" width="121" height="44" rx="6" fill="#FFFBEB" stroke="#FDE68A"/>
    <text x="729" y="251" class="mono tiny" fill="#B45309">? ASK HUMAN</text>
    <text x="729" y="266" class="muted tiny">Escalate + explain</text>

    <!-- REJECT -->
    <rect x="717" y="284" width="121" height="40" rx="6" fill="#FEF2F2" stroke="#FECACA"/>
    <text x="729" y="302" class="mono tiny" fill="#B91C1C">✕ REJECT</text>
    <text x="729" y="316" class="muted tiny">All h ∈ Ĥ_τ unsafe</text>
  </g>

  <!-- Footer caption -->
  <text x="30" y="366" class="muted tiny">Why it works: Top-1 gates only check h₁ (and fail when h₂ is true). RiskGate checks every plausible hypothesis h ∈ Ĥ_τ and walks down the ranked list.</text>
</svg>
"""


def build_2_cstr_reactor_probe() -> str:
    """Animation 2: Exothermic CSTR Reactor — Active Diagnostic Probing (PROBE -> CERTIFY)."""
    # Generate smooth SVG path coordinates for the reactor trajectory plot
    # Plot box: x in [50, 520], y in [105, 315]
    # Temperature range: 330 K (y=315) to 415 K (y=105)
    def tx(t_min: float) -> float:
        # t from -10 to +20 min -> x from 65 to 505
        return 65.0 + (t_min + 10.0) / 30.0 * 440.0

    def ty(temp_k: float) -> float:
        return 315.0 - (temp_k - 330.0) / (415.0 - 330.0) * 210.0

    # Observation window (-10 to 0 min): rises from 362 K to 388 K
    obs_pts = " ".join(
        f"{'M' if i == 0 else 'L'}{tx(t):.1f},{ty(362 + 26 * ((t + 10) / 10) ** 1.25):.1f}"
        for i, t in enumerate(range(-10, 1))
    )
    # Candidate #1 (coolant -8 K) under h1 (sticky valve): stabilizes nicely around 370 K
    c1_h1 = " ".join(
        f"{'M' if i == 0 else 'L'}{tx(t):.1f},{ty(388 - 18 * (1 - math.exp(-t / 5.0))):.1f}"
        for i, t in enumerate(range(0, 21, 2))
    )
    # Candidate #1 (coolant -8 K) under h2 (sensor drift +1.6 K/min): true T was only 358 K -> quenches below 335 K!
    c1_h2 = " ".join(
        f"{'M' if i == 0 else 'L'}{tx(t):.1f},{ty(358 - 28 * (1 - math.exp(-t / 4.5))):.1f}"
        for i, t in enumerate(range(0, 21, 2))
    )
    # Probe (0 to 3 min) + Certified recovery (3 to 20 min): stays safely at 368 K!
    probe_rec = " ".join(
        f"{'M' if i == 0 else 'L'}{tx(t):.1f},{ty(388 - 20 * (1 - math.exp(-t / 4.0))):.1f}"
        for i, t in enumerate(range(0, 21, 2))
    )

    y_hi = ty(405.0)
    y_lo = ty(335.0)
    x_zero = tx(0.0)
    x_probe_end = tx(3.0)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 390" width="100%" height="100%">
  {COMMON_STYLE}
  <rect class="bg" width="880" height="390" rx="10"/>

  <!-- Header -->
  <text x="30" y="34" class="ink title">Example 1 · Chemical Reactor (CSTR) — Active Probing Resolves Fault Ambiguity</text>
  <text x="30" y="52" class="muted subtitle">Measured T_meas climbs to 388 K. Is the coolant valve sticking (h₁) or is the thermometer drifting +1.6 K/min (h₂)?</text>
  <text x="850" y="34" text-anchor="end" class="mono-muted tiny">α = 5% · TUBE = 0.96 K</text>

  <!-- Timeline progress bar -->
  <rect x="30" y="64" width="820" height="3" fill="#E4E2DD" rx="1.5"/>
  <rect x="30" y="64" height="3" fill="#1D4ED8" rx="1.5" class="progress-bar"/>

  <!-- Left panel: Trajectory Plot -->
  <rect x="30" y="82" width="495" height="258" rx="8" class="panel"/>
  <text x="48" y="103" class="muted label">Reactor Temperature Trajectory T(t) [Kelvin]</text>

  <!-- Safe / Danger bands -->
  <rect x="65" y="108" width="440" height="{y_hi - 108:.1f}" class="danger-fill"/>
  <rect x="65" y="{y_hi:.1f}" width="440" height="{y_lo - y_hi:.1f}" class="safe-fill"/>
  <rect x="65" y="{y_lo:.1f}" width="440" height="{320 - y_lo:.1f}" class="danger-fill"/>

  <!-- Upper & Lower safety limits -->
  <line x1="65" y1="{y_hi:.1f}" x2="505" y2="{y_hi:.1f}" class="limit-line"/>
  <text x="502" y="{y_hi - 4:.1f}" text-anchor="end" class="mono tiny" fill="#B91C1C">Runaway limit 405 K</text>
  <line x1="65" y1="{y_lo:.1f}" x2="505" y2="{y_lo:.1f}" class="limit-line"/>
  <text x="502" y="{y_lo + 11:.1f}" text-anchor="end" class="mono tiny" fill="#B91C1C">Quench limit 335 K</text>

  <!-- Decision time t=0 and Probe window t=3m -->
  <line x1="{x_zero:.1f}" y1="108" x2="{x_zero:.1f}" y2="320" class="thresh-line"/>
  <text x="{x_zero - 5:.1f}" y="120" text-anchor="end" class="mono-muted tiny">t = 0 (alarm)</text>
  <rect x="{x_zero:.1f}" y="108" width="{x_probe_end - x_zero:.1f}" height="212" fill="#1D4ED8" fill-opacity="0.08"/>
  <text x="{(x_zero + x_probe_end)/2:.1f}" y="314" text-anchor="middle" class="mono tiny" fill="#1D4ED8">3m probe</text>

  <!-- Observed history (-10 to 0 min) -->
  <path d="{obs_pts}" fill="none" stroke="#18181B" stroke-width="2.4"/>
  <circle cx="{x_zero:.1f}" cy="{ty(388):.1f}" r="4.5" fill="#18181B" class="pulse"/>

  <!-- Rollouts for LLM Top-1 (coolant -8 K): h1 safe, h2 quenches! -->
  <path d="{c1_h1}" fill="none" stroke="#686662" stroke-width="1.8" stroke-dasharray="4 3" class="curve-anim"/>
  <text x="415" y="{ty(373):.1f}" class="mono-muted tiny">coolant -8K | h₁ (OK)</text>

  <path d="{c1_h2}" fill="none" stroke="#B91C1C" stroke-width="2.2" class="curve-anim"/>
  <text x="395" y="{ty(331):.1f}" class="mono tiny" fill="#B91C1C">coolant -8K | h₂ (QUENCH!)</text>

  <!-- Probe + Certified Recovery trajectory -->
  <path d="{probe_rec}" fill="none" stroke="#15803D" stroke-width="2.8" class="probe-anim"/>
  <text x="385" y="{ty(361):.1f}" class="mono tiny" fill="#15803D" class="st4">✓ after 3m probe: certified safe</text>

  <!-- Right Panel: Step-by-Step Decision Trace -->
  <rect x="540" y="82" width="310" height="258" rx="8" class="panel"/>
  <text x="556" y="103" class="muted label">RiskGate Two-Stage Decision Trace</text>

  <!-- Step A -->
  <g class="st1">
    <rect x="556" y="114" width="278" height="48" rx="6" class="panel-subtle"/>
    <text x="566" y="131" class="mono tiny">1. Plausible set Ĥ_τ = &#123;h₁: valve, h₂: sensor&#125;</text>
    <text x="566" y="148" class="muted tiny">Both match the 10-min rise within τ = 1.68</text>
  </g>

  <!-- Step B -->
  <g class="st2">
    <rect x="556" y="170" width="278" height="50" rx="6" fill="#FEF2F2" stroke="#FECACA"/>
    <text x="566" y="187" class="mono tiny" fill="#991B1B">2. Top-1 `coolant_sp -8 K` blocked</text>
    <text x="566" y="204" class="muted tiny">Safe under h₁ (+14 K), but quenches under h₂ (-4.8 K)</text>
  </g>

  <!-- Step C -->
  <g class="st3">
    <rect x="556" y="228" width="278" height="50" rx="6" fill="#EFF6FF" stroke="#BFDBFE"/>
    <text x="566" y="245" class="mono tiny" fill="#1D4ED8">3. Verdict: PROBE (read_backup_sensor_3m)</text>
    <text x="566" y="262" class="muted tiny">Certified safe under h₁ &amp; h₂; resolves 100% of pairs</text>
  </g>

  <!-- Step D -->
  <g class="st4">
    <rect x="556" y="286" width="278" height="44" rx="6" fill="#15803D"/>
    <text x="566" y="304" class="mono tiny" fill="#FFFFFF" font-weight="600">4. Stage 2: CERTIFY recovery (m = +18.2 K ≥ λ̂₂)</text>
    <text x="566" y="320" class="tiny" fill="#DCFCE7">Hand-overs cut from 27.8% → 12.2% (unsafe = 4.8% ≤ 5%)</text>
  </g>

  <text x="30" y="366" class="muted tiny">Benchmark result (500 test episodes): Uncalibrated twin gate = 44.6% thermal runaways · Top-1 conformal = 0% certified · RiskGate + Probe = 4.8% unsafe, 87.8% certified.</text>
</svg>
"""


def build_3_k8s_sre_walkdown() -> str:
    """Animation 3: Kubernetes Incident SRE Agent — Ranked Candidate Walk-Down."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 390" width="100%" height="100%">
  {COMMON_STYLE}
  <rect class="bg" width="880" height="390" rx="10"/>

  <!-- Header -->
  <text x="30" y="34" class="ink title">Example 2 · Kubernetes SRE Agent — Ranked Walk-Down Under Root-Cause Ambiguity</text>
  <text x="30" y="52" class="muted subtitle">Alert: checkout-api p95 = 920 ms, error_rate = 6.4%. Deploy v2.14 rolled out 9m ago, AND db_cpu is at 91%.</text>
  <text x="850" y="34" text-anchor="end" class="mono-muted tiny">LATENCY = 0.08 ms</text>

  <!-- Timeline progress bar -->
  <rect x="30" y="64" width="820" height="3" fill="#E4E2DD" rx="1.5"/>
  <rect x="30" y="64" height="3" fill="#15803D" rx="1.5" class="progress-bar"/>

  <!-- Left: Telemetry & Plausible Root Causes -->
  <g class="st1">
    <rect x="30" y="82" width="250" height="258" rx="8" class="panel"/>
    <text x="46" y="104" class="muted label">Live Cluster Context &amp; Ĥ_τ</text>

    <rect x="46" y="116" width="218" height="60" rx="6" class="panel-subtle"/>
    <text x="56" y="134" class="mono tiny">p95_latency:  920 ms (SLA ≤ 500)</text>
    <text x="56" y="150" class="mono tiny">error_rate:   6.4%   (SLA ≤ 2.0%)</text>
    <text x="56" y="166" class="mono tiny">db_cpu / pod: 91% / 78%</text>

    <text x="46" y="198" class="muted label">Plausible Root Causes (Ĥ_τ)</text>
    <rect x="46" y="208" width="218" height="36" rx="5" fill="#F8FAFC" stroke="#CBD5E1"/>
    <text x="56" y="224" class="mono tiny">h₁: bad_canary_deploy</text>
    <text x="56" y="238" class="muted tiny">r = 0.62 ≤ τ (fits error spike)</text>

    <rect x="46" y="250" width="218" height="36" rx="5" fill="#F8FAFC" stroke="#CBD5E1"/>
    <text x="56" y="266" class="mono tiny">h₂: db_lock_contention</text>
    <text x="56" y="280" class="muted tiny">r = 0.89 ≤ τ (fits db_cpu = 91%)</text>

    <rect x="46" y="292" width="218" height="36" rx="5" fill="#F8FAFC" stroke="#CBD5E1"/>
    <text x="56" y="308" class="mono tiny">h₃: noisy_neighbor_cpu</text>
    <text x="56" y="322" class="muted tiny">r = 1.31 ≤ τ (plausible)</text>
  </g>

  <!-- Center: Candidate Walk-Down Matrix -->
  <g class="st2">
    <rect x="296" y="82" width="352" height="258" rx="8" class="panel"/>
    <text x="312" y="104" class="muted label">LLM Ranked Proposals × Plausible Causes</text>

    <!-- Table header -->
    <text x="312" y="128" class="mono-muted tiny">CANDIDATE ACTION</text>
    <text x="495" y="128" text-anchor="middle" class="mono-muted tiny">h₁</text>
    <text x="545" y="128" text-anchor="middle" class="mono-muted tiny">h₂</text>
    <text x="595" y="128" text-anchor="middle" class="mono-muted tiny">h₃</text>
    <line x1="312" y1="135" x2="632" y2="135" stroke="#E4E2DD"/>

    <!-- Row 1: Hard Rule Blocked -->
    <rect x="310" y="143" width="324" height="40" rx="5" fill="#FEF2F2" stroke="#FECACA"/>
    <text x="320" y="160" class="mono tiny" fill="#991B1B">#1 drop_prod_replica_table</text>
    <text x="320" y="175" class="muted tiny">Blocked by HardRule(&quot;no_destructive_ddl&quot;)</text>
    <text x="624" y="166" text-anchor="end" class="mono tiny" fill="#B91C1C">HARD BLOCK</text>

    <!-- Row 2: Top-1 viable fails under h2 -->
    <g class="scan1">
      <rect x="310" y="191" width="324" height="44" rx="5" fill="#FFFBEB" stroke="#FDE68A"/>
      <text x="320" y="209" class="mono tiny">#2 rollback_deploy(v2.13)</text>
      <text x="320" y="225" class="muted tiny">Fixes h₁, but leaves DB lock h₂ at p95=890ms</text>
      <text x="495" y="214" text-anchor="middle" class="mono tiny" fill="#15803D">+0.38</text>
      <text x="545" y="214" text-anchor="middle" class="mono tiny" fill="#B91C1C">-0.78</text>
      <text x="595" y="214" text-anchor="middle" class="mono tiny" fill="#B91C1C">-0.41</text>
    </g>

    <!-- Row 3: Scale replicas +4 fails under h2 (exhausts DB pool!) -->
    <rect x="310" y="243" width="324" height="40" rx="5" class="panel-subtle"/>
    <text x="320" y="260" class="mono tiny">#3 scale_replicas(+4)</text>
    <text x="320" y="274" class="muted tiny">More pods exhaust DB connection pool under h₂</text>
    <text x="495" y="264" text-anchor="middle" class="mono tiny" fill="#15803D">+0.15</text>
    <text x="545" y="264" text-anchor="middle" class="mono tiny" fill="#B91C1C">-1.12</text>
    <text x="595" y="264" text-anchor="middle" class="mono tiny" fill="#15803D">+0.29</text>

    <!-- Row 4: Certified! -->
    <g class="scan2">
      <rect x="310" y="291" width="324" height="40" rx="5" fill="#F0FDF4" stroke="#86EFAC" stroke-width="1.5"/>
      <text x="320" y="308" class="mono tiny" fill="#166534">#4 shed_noncritical_traffic(25%)</text>
      <text x="320" y="322" class="muted tiny">Drops load on pods AND DB; safe under all h ∈ Ĥ_τ</text>
      <text x="495" y="312" text-anchor="middle" class="mono tiny" fill="#15803D">+0.34</text>
      <text x="545" y="312" text-anchor="middle" class="mono tiny" fill="#15803D">+0.28</text>
      <text x="595" y="312" text-anchor="middle" class="mono tiny" fill="#15803D">+0.31</text>
    </g>
  </g>

  <!-- Right: Resulting Verdict & Contrastive Audit Record -->
  <g class="st4">
    <rect x="664" y="82" width="186" height="258" rx="8" class="panel"/>
    <text x="680" y="104" class="muted label">Certified Output</text>

    <rect x="678" y="116" width="158" height="52" rx="6" fill="#15803D"/>
    <text x="690" y="137" class="mono tiny" fill="#FFFFFF" font-weight="600">✓ CERTIFY #4</text>
    <text x="690" y="155" class="mono tiny" fill="#DCFCE7">s = +0.28 ≥ λ̂ (+0.14)</text>

    <text x="680" y="190" class="muted label">Contrastive Explanation</text>
    <rect x="678" y="200" width="158" height="126" rx="6" class="panel-subtle"/>
    <text x="686" y="218" class="mono tiny">&quot;Skipped #2 rollback:</text>
    <text x="686" y="233" class="muted tiny">under h₂ (db_lock),</text>
    <text x="686" y="247" class="muted tiny">latency_p95 = -0.78.</text>
    <text x="686" y="266" class="mono tiny" fill="#15803D">Certified #4 shed_25%:</text>
    <text x="686" y="281" class="muted tiny">worst margin +0.28</text>
    <text x="686" y="295" class="muted tiny">across &#123;h₁, h₂, h₃&#125;.</text>
    <text x="686" y="314" class="mono-muted tiny">sha256: 9f4c...8a1b&quot;</text>
  </g>

  <text x="30" y="366" class="muted tiny">Why ranked walk-down matters: Top-1 CRC can only accept or reject #2 (high ask rate). RiskGate walks down the agent's list and certifies #4 automatically.</text>
</svg>
"""


def build_4_refunds_multirisk() -> str:
    """Animation 4: Refund Agent Multi-Risk Calibration + Why Top-1 Fails (Proposition 1)."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 390" width="100%" height="100%">
  {COMMON_STYLE}
  <rect class="bg" width="880" height="390" rx="10"/>

  <!-- Header -->
  <text x="30" y="34" class="ink title">Example 3 · Refund Agent (Multi-Risk Control) &amp; Why Single-Hypothesis Gates Fail</text>
  <text x="30" y="52" class="muted subtitle">Left: Proposition 1 (Top-1 misdiagnosis blows up the conformal tube). Right: Jointly bounding policy risk AND dollar loss.</text>
  <text x="850" y="34" text-anchor="end" class="mono-muted tiny">THEOREM 1 + PROP. 1</text>

  <!-- Timeline progress bar -->
  <rect x="30" y="64" width="820" height="3" fill="#E4E2DD" rx="1.5"/>
  <rect x="30" y="64" height="3" fill="#B45309" rx="1.5" class="progress-bar"/>

  <!-- Left Panel: Proposition 1 (26.6 K vs 0.96 K tube) -->
  <g class="st1">
    <rect x="30" y="82" width="400" height="258" rx="8" class="panel"/>
    <text x="48" y="104" class="muted label">Proposition 1 · Why Top-1 Conformal Collapses</text>
    <text x="48" y="123" class="muted tiny">When P(ĥ₁ ≠ h*) &gt; α, top-1 quantiles are dominated by misdiagnosed runs:</text>

    <!-- Bar 1: Top-1 Conformal Tube -->
    <text x="48" y="152" class="mono tiny" fill="#B91C1C">Top-1 Conformal (single best guess ĥ₁)</text>
    <rect x="48" y="160" width="350" height="28" rx="4" fill="#FEF2F2" stroke="#FECACA"/>
    <rect x="48" y="160" width="336" height="28" rx="4" fill="#B91C1C" fill-opacity="0.85" class="st2"/>
    <text x="392" y="178" text-anchor="end" class="mono tiny" fill="#FFFFFF" font-weight="600">26.6 K tube → 0.0% certified (100% ASK)</text>

    <!-- Bar 2: Hypothesis-Set RiskGate Tube -->
    <text x="48" y="214" class="mono tiny" fill="#15803D">RiskGate Hypothesis Set Ĥ_τ (covers h* 91.2% of runs)</text>
    <rect x="48" y="222" width="350" height="28" rx="4" fill="#F0FDF4" stroke="#BBF7D0"/>
    <rect x="48" y="222" width="24" height="28" rx="4" fill="#15803D"/>
    <text x="80" y="240" class="mono tiny" fill="#15803D" font-weight="600">0.96 K tube (28× sharper) → 87.8% certified!</text>

    <!-- Takeaway box -->
    <rect x="48" y="266" width="364" height="58" rx="6" class="panel-subtle"/>
    <text x="58" y="285" class="mono tiny">Key mathematical insight:</text>
    <text x="58" y="301" class="muted tiny">Separating epistemic fault ambiguity (Ĥ_τ) from physical</text>
    <text x="58" y="315" class="muted tiny">simulation error keeps tubes tight and certification high.</text>
  </g>

  <!-- Right Panel: E-Commerce Refund Agent Multi-Risk Control -->
  <g class="st2">
    <rect x="450" y="82" width="400" height="258" rx="8" class="panel"/>
    <text x="468" y="104" class="muted label">Refund Agent · Joint Multi-Risk Threshold λ̂</text>
    <text x="468" y="123" class="muted tiny">Claim: $185 item &quot;arrived damaged&quot;, VIP tier, 2 prior claims, no photo</text>

    <!-- Two risk curves / bars -->
    <rect x="468" y="136" width="364" height="52" rx="6" class="panel-subtle"/>
    <text x="478" y="154" class="mono tiny">Risk 1 (Policy violation ≤ α₁ = 5.0%):   λ̂_safety = +0.08</text>
    <text x="478" y="172" class="mono tiny">Risk 2 (Overpay leakage ≤ α₂ = 3.0%):    λ̂_cost   = +0.19</text>
    <text x="822" y="163" text-anchor="end" class="mono tiny" fill="#18181B" font-weight="600">λ̂ = max = +0.19</text>

    <!-- Plausible hypotheses for the refund claim -->
    <g class="st3">
      <rect x="468" y="198" width="176" height="62" rx="6" fill="#F0FDF4" stroke="#BBF7D0"/>
      <text x="478" y="216" class="mono tiny" fill="#166534">h₁: genuine_damage</text>
      <text x="478" y="233" class="muted tiny">full_refund($185) valid</text>
      <text x="478" y="249" class="mono tiny" fill="#15803D">margin = +0.45</text>

      <rect x="656" y="198" width="176" height="62" rx="6" fill="#FEF2F2" stroke="#FECACA"/>
      <text x="666" y="216" class="mono tiny" fill="#991B1B">h₂: opportunistic_abuse</text>
      <text x="666" y="233" class="muted tiny">unverified &gt;$150 violates SLA</text>
      <text x="666" y="249" class="mono tiny" fill="#B91C1C">margin = -0.52 &lt; λ̂</text>
    </g>

    <!-- Probe action selected -->
    <g class="st4">
      <rect x="468" y="270" width="364" height="54" rx="6" fill="#EFF6FF" stroke="#BFDBFE"/>
      <text x="480" y="290" class="mono tiny" fill="#1D4ED8" font-weight="600">◉ Verdict: PROBE → request_defect_photo_and_carrier_scan()</text>
      <text x="480" y="308" class="muted tiny">Zero dollar leakage during probe; once photo arrives, h₂ is ruled out → CERTIFY.</text>
    </g>
  </g>

  <text x="30" y="366" class="muted tiny">Across all 3 domains (Reactor, Kubernetes SRE, Refund Agent), RiskGate holds empirical unsafe execution below α = 5% on 100% of random calibration/test splits.</text>
</svg>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "1_pipeline_concept.svg": build_1_pipeline_concept(),
        "2_cstr_reactor_probe.svg": build_2_cstr_reactor_probe(),
        "3_k8s_sre_walkdown.svg": build_3_k8s_sre_walkdown(),
        "4_refunds_multirisk.svg": build_4_refunds_multirisk(),
    }
    for name, content in files.items():
        (OUT / name).write_text(content.strip() + "\n")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
