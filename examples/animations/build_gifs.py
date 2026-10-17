"""Render minimal, Swiss-editorial animated GIFs for RiskGate and its 3 domain examples."""
# ruff: noqa: E501
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

# Render at 2x supersampling (1920 x 860) and downsample with LANCZOS to (960 x 430) for razor-sharp typography & curves
SCALE = 2
W, H = 960 * SCALE, 430 * SCALE
OUT_W, OUT_H = 960, 430

# Palette (Swiss editorial paper & semantic inks)
BG = (250, 249, 245)
PANEL = (255, 255, 255)
SUBTLE = (243, 241, 235)
BORDER = (226, 224, 216)
BORDER_STRONG = (196, 193, 184)
INK = (24, 24, 27)
MUTED = (104, 102, 98)
LIGHT_INK = (145, 142, 136)

GREEN = (21, 128, 61)
GREEN_BG = (240, 253, 244)
GREEN_BORDER = (187, 247, 208)

RED = (185, 28, 28)
RED_BG = (254, 242, 242)
RED_BORDER = (254, 202, 202)

BLUE = (29, 78, 216)
BLUE_BG = (239, 246, 255)
BLUE_BORDER = (191, 219, 254)

AMBER = (180, 83, 9)
AMBER_BG = (255, 251, 235)
AMBER_BORDER = (253, 230, 138)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size * SCALE)


F_TITLE = font("DejaVuSans-Bold.ttf", 16)
F_SUB = font("DejaVuSans.ttf", 11)
F_LABEL = font("DejaVuSans-Bold.ttf", 10)
F_BODY = font("DejaVuSans.ttf", 11)
F_BOLD = font("DejaVuSans-Bold.ttf", 11)
F_SMALL = font("DejaVuSans.ttf", 10)
F_MONO = font("DejaVuSansMono.ttf", 10)
F_MONO_B = font("DejaVuSansMono-Bold.ttf", 10)
F_MONO_S = font("DejaVuSansMono.ttf", 9)


def s(val: float) -> int:
    return int(round(val * SCALE))


def blend(c1: tuple[int, int, int], c2: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    a = max(0.0, min(1.0, alpha))
    return (
        int(round(c1[0] * (1 - a) + c2[0] * a)),
        int(round(c1[1] * (1 - a) + c2[1] * a)),
        int(round(c1[2] * (1 - a) + c2[2] * a)),
    )


def ease(t: float, t0: float, t1: float) -> float:
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    u = (t - t0) / (t1 - t0)
    return u * u * (3.0 - 2.0 * u)


def rrect(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], r: float, fill, outline=None, width: int = 1):
    x0, y0, x1, y1 = s(box[0]), s(box[1]), s(box[2]), s(box[3])
    draw.rounded_rectangle([x0, y0, x1, y1], radius=s(r), fill=fill, outline=outline, width=max(1, int(width * SCALE)))


def text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], txt: str, fnt, fill=INK, anchor: str = "la"):
    draw.text((s(xy[0]), s(xy[1])), txt, font=fnt, fill=fill, anchor=anchor)


def dashed_line(draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float], fill, width: float = 1.4, dash: float = 6, gap: float = 4):
    x0, y0 = p0
    x1, y1 = p1
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist == 0:
        return
    dx, dy = (x1 - x0) / dist, (y1 - y0) / dist
    pos = 0.0
    while pos < dist:
        end = min(dist, pos + dash)
        draw.line([(s(x0 + dx * pos), s(y0 + dy * pos)), (s(x0 + dx * end), s(y0 + dy * end))], fill=fill, width=s(width))
        pos += dash + gap


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str, right_tag: str, progress: float, bar_color=INK):
    text(draw, (28, 20), title, F_TITLE, INK)
    text(draw, (28, 43), subtitle, F_SUB, MUTED)
    text(draw, (932, 24), right_tag, F_MONO_B, MUTED, anchor="ra")
    # Progress bar
    rrect(draw, (28, 64, 932, 68), 2, BORDER)
    pw = max(6.0, 904.0 * min(1.0, progress))
    rrect(draw, (28, 64, 28 + pw, 68), 2, bar_color)


def save_gif(frames: list[Image.Image], durations: list[int], path: Path) -> None:
    small = [im.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS).quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in frames]
    small[0].save(path, save_all=True, append_images=small[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {path} ({len(small)} frames, {path.stat().st_size // 1024} KB)")


# ==============================================================================================
# GIF 1: Core RiskGate 4-Verdict Verification Pipeline
# ==============================================================================================
def build_gif_1() -> None:
    frames, durations = [], []
    n_frames = 44
    for k in range(n_frames):
        t = k / (n_frames - 1)
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)

        draw_header(
            d,
            "RiskGate · Consequence-Based Verification Loop",
            "Every proposed tool call is checked across plausible world hypotheses before execution: P(certified and unsafe) <= alpha",
            f"STAGE {min(4, int(t * 4) + 1)} OF 4",
            t,
            INK,
        )

        a2 = ease(t, 0.18, 0.36)
        a3 = ease(t, 0.42, 0.62)
        a4 = ease(t, 0.68, 0.85)

        # Panel 1: Doer (LLM Agent)
        rrect(d, (28, 82, 238, 378), 8, PANEL, BORDER)
        text(d, (42, 96), "01 · DOER (LLM AGENT)", F_LABEL, MUTED)
        text(d, (42, 115), "Proposes ranked actions", F_SMALL, INK)

        # Candidate #1
        c1_active = 0.40 <= t < 0.64
        rrect(d, (40, 140, 226, 202), 6, RED_BG if t >= 0.48 else SUBTLE, RED if c1_active else (RED_BORDER if t >= 0.48 else BORDER), 2 if c1_active else 1)
        text(d, (50, 150), "#1 aggressive_act()", F_MONO_B, RED if t >= 0.48 else INK)
        text(d, (50, 169), "Fast recovery if h1 true", F_SMALL, MUTED)
        text(d, (50, 184), "Violates limit under h2!", F_MONO_S, RED if t >= 0.48 else MUTED)

        # Candidate #2
        c2_active = t >= 0.60
        rrect(d, (40, 214, 226, 276), 6, GREEN_BG if c2_active else SUBTLE, GREEN if c2_active else BORDER, 2 if c2_active else 1)
        text(d, (50, 224), "#2 safe_mitigate()", F_MONO_B, GREEN if c2_active else INK)
        text(d, (50, 243), "Safe under all h in H_tau", F_SMALL, MUTED)
        text(d, (50, 258), "Worst margin +0.24 >= lam", F_MONO_S, GREEN if c2_active else MUTED)

        # Candidate #3
        rrect(d, (40, 288, 226, 344), 6, SUBTLE, BORDER)
        text(d, (50, 298), "#3 fallback_hold()", F_MONO, MUTED)
        text(d, (50, 317), "Backup conservative option", F_SMALL, LIGHT_INK)

        # Arrow 1 -> 2
        c_arr1 = blend(BORDER, INK, a2)
        d.line([(s(238), s(230)), (s(262), s(230))], fill=c_arr1, width=s(1.6))

        # Panel 2: Plausible Set H_tau
        p2_bg = blend(BG, PANEL, 0.3 + 0.7 * a2)
        rrect(d, (262, 82, 488, 378), 8, p2_bg, blend(BORDER, BORDER_STRONG, a2))
        text(d, (276, 96), "02 · PLAUSIBLE SET H_tau(x)", F_LABEL, blend(LIGHT_INK, MUTED, a2))
        text(d, (276, 115), "Keep all h with r(h, x) <= tau", F_SMALL, blend(LIGHT_INK, INK, a2))

        if a2 > 0.05:
            rrect(d, (274, 140, 476, 194), 6, GREEN_BG, GREEN_BORDER)
            text(d, (284, 149), "h1: primary_fault", F_MONO_B, INK)
            text(d, (466, 149), "r=0.4 <= tau", F_MONO_S, GREEN, anchor="ra")
            text(d, (284, 169), "Plausible (top-1 guess)", F_SMALL, MUTED)

            rrect(d, (274, 204, 476, 258), 6, GREEN_BG, GREEN_BORDER)
            text(d, (284, 213), "h2: sensor_or_db_fault", F_MONO_B, INK)
            text(d, (466, 213), "r=1.1 <= tau", F_MONO_S, GREEN, anchor="ra")
            text(d, (284, 233), "Also fits evidence! Kept in H_tau", F_SMALL, MUTED)

            rrect(d, (274, 268, 476, 322), 6, SUBTLE, BORDER)
            text(d, (284, 277), "h3: pipe_rupture", F_MONO, LIGHT_INK)
            text(d, (466, 277), "r=6.8 > tau", F_MONO_S, RED, anchor="ra")
            text(d, (284, 297), "Excluded (contradicts data)", F_SMALL, LIGHT_INK)

            text(d, (276, 344), "Guarantee: P(h* in H_tau) >= 1 - delta", F_MONO_S, MUTED)

        # Arrow 2 -> 3
        c_arr2 = blend(BORDER, INK, a3)
        d.line([(s(488), s(230)), (s(512), s(230))], fill=c_arr2, width=s(1.6))

        # Panel 3: Worst-Case Margin Check
        rrect(d, (512, 82, 744, 378), 8, blend(BG, PANEL, 0.3 + 0.7 * a3), blend(BORDER, BORDER_STRONG, a3))
        text(d, (526, 96), "03 · WORST-CASE MARGIN", F_LABEL, blend(LIGHT_INK, MUTED, a3))
        text(d, (526, 115), "s(a) = min_{h in H_tau} m(a, h)", F_SMALL, blend(LIGHT_INK, INK, a3))

        if a3 > 0.05:
            rrect(d, (524, 140, 732, 224), 6, RED_BG, RED_BORDER)
            text(d, (534, 150), "Check #1 aggressive_act()", F_MONO_B, RED)
            text(d, (534, 171), "m(#1, h1) = +0.42  (safe)", F_MONO_S, GREEN)
            text(d, (534, 188), "m(#1, h2) = -0.38  (UNSAFE!)", F_MONO_B, RED)
            text(d, (534, 205), "Worst s(#1) = -0.38 < lam (+0.12)", F_MONO_S, RED)

        if t >= 0.56:
            rrect(d, (524, 236, 732, 320), 6, GREEN_BG, GREEN, 1.6)
            text(d, (534, 246), "Check #2 safe_mitigate()", F_MONO_B, GREEN)
            text(d, (534, 267), "m(#2, h1) = +0.31  (safe)", F_MONO_S, GREEN)
            text(d, (534, 284), "m(#2, h2) = +0.24  (safe)", F_MONO_S, GREEN)
            text(d, (534, 301), "Worst s(#2) = +0.24 >= lam (+0.12)", F_MONO_B, GREEN)

        text(d, (526, 344), "CRC threshold lam_hat = +0.12", F_MONO_S, MUTED)

        # Arrow 3 -> 4
        c_arr3 = blend(BORDER, INK, a4)
        d.line([(s(744), s(230)), (s(768), s(230))], fill=c_arr3, width=s(1.6))

        # Panel 4: Four Verdicts
        rrect(d, (768, 82, 932, 378), 8, PANEL, BORDER)
        text(d, (782, 96), "04 · VERDICT", F_LABEL, MUTED)

        # CERTIFY
        cert_hi = a4 > 0.2
        rrect(d, (780, 122, 920, 180), 6, GREEN if cert_hi else GREEN_BG, GREEN, 2 if cert_hi else 1)
        text(d, (790, 134), "CERTIFY #2", F_MONO_B, (255, 255, 255) if cert_hi else GREEN)
        text(d, (790, 155), "Execute #2 safely", F_SMALL, (220, 252, 231) if cert_hi else MUTED)

        # PROBE
        rrect(d, (780, 190, 920, 242), 6, BLUE_BG, BLUE_BORDER)
        text(d, (790, 200), "PROBE", F_MONO_B, BLUE)
        text(d, (790, 219), "Safe diagnostic test", F_SMALL, MUTED)

        # ASK
        rrect(d, (780, 252, 920, 304), 6, AMBER_BG, AMBER_BORDER)
        text(d, (790, 262), "ASK HUMAN", F_MONO_B, AMBER)
        text(d, (790, 281), "Escalate + explain", F_SMALL, MUTED)

        # REJECT
        rrect(d, (780, 314, 920, 364), 6, RED_BG, RED_BORDER)
        text(d, (790, 324), "REJECT", F_MONO_B, RED)
        text(d, (790, 343), "All h in H_tau unsafe", F_SMALL, MUTED)

        text(
            d,
            (28, 398),
            "Key takeaway: Top-1 gates check only h1 (and fail when h2 is true). RiskGate checks all plausible h in H_tau and walks down the ranked list.",
            F_SMALL,
            MUTED,
        )

        frames.append(im)
        durations.append(110 if k < n_frames - 1 else 1400)

    save_gif(frames, durations, OUT / "1_pipeline_concept.gif")


# ==============================================================================================
# GIF 2: CSTR Exothermic Reactor — Active Probing (PROBE -> CERTIFY)
# ==============================================================================================
def build_gif_2() -> None:
    frames, durations = [], []
    n_frames = 48

    def tx(t_min: float) -> float:
        return 68.0 + (t_min + 10.0) / 30.0 * 470.0

    def ty(temp_k: float) -> float:
        return 345.0 - (temp_k - 328.0) / (415.0 - 328.0) * 235.0

    for k in range(n_frames):
        t = k / (n_frames - 1)
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)

        draw_header(
            d,
            "Example 1 · Chemical Reactor (CSTR) — Active Probing Resolves Fault Ambiguity",
            "Measured T climbs to 388 K. Is the coolant valve sticking (h1) or is the thermometer drifting +1.6 K/min (h2)?",
            "alpha = 5% · TUBE = 0.96 K",
            t,
            BLUE,
        )

        # Left Plot Panel
        rrect(d, (28, 82, 560, 378), 8, PANEL, BORDER)
        text(d, (44, 95), "REACTOR TEMPERATURE TRAJECTORY T(t) [KELVIN]", F_LABEL, MUTED)

        y_hi = ty(405.0)
        y_lo = ty(335.0)
        x0 = tx(0.0)
        xp = tx(3.0)

        # Danger & Safe bands
        d.rectangle([s(68), s(112), s(538), s(y_hi)], fill=(254, 242, 242))
        d.rectangle([s(68), s(y_hi), s(538), s(y_lo)], fill=(244, 251, 246))
        d.rectangle([s(68), s(y_lo), s(538), s(352)], fill=(254, 242, 242))

        dashed_line(d, (68, y_hi), (538, y_hi), RED, 1.4)
        text(d, (534, y_hi - 14), "Runaway Limit 405 K", F_MONO_S, RED, anchor="ra")
        dashed_line(d, (68, y_lo), (538, y_lo), RED, 1.4)
        text(d, (534, y_lo + 4), "Quench Limit 335 K", F_MONO_S, RED, anchor="ra")

        # Y-axis labels
        for temp_lbl in (340, 360, 380, 400):
            text(d, (62, ty(temp_lbl) - 6), f"{temp_lbl}K", F_MONO_S, MUTED, anchor="ra")

        # t = 0 line
        dashed_line(d, (x0, 112), (x0, 352), MUTED, 1.1)
        text(d, (x0 - 5, 116), "t=0 (alarm)", F_MONO_S, MUTED, anchor="ra")

        # 1) Observed curve (-10 to 0 min)
        obs_p = ease(t, 0.0, 0.20)
        n_obs = max(2, int(40 * obs_p))
        pts_obs = []
        for i in range(n_obs + 1):
            frac = (i / 40.0)
            tm = -10.0 + 10.0 * frac
            tk = 362.0 + 26.0 * (frac ** 1.25)
            pts_obs.append((s(tx(tm)), s(ty(tk))))
        if len(pts_obs) >= 2:
            d.line(pts_obs, fill=INK, width=s(2.5))
            cx, cy = pts_obs[-1]
            d.ellipse([cx - s(4), cy - s(4), cx + s(4), cy + s(4)], fill=INK)

        # 2) True temperature under h2 (sensor drift)
        if t >= 0.20:
            dashed_line(d, (tx(-10), ty(354)), (tx(0), ty(358)), BLUE, 1.4)
            text(d, (x0 - 6, ty(356) - 14), "True T under h2 (358K)", F_MONO_S, BLUE, anchor="ra")

        # 3) Rollouts of LLM Top-1 (coolant -8K) under h1 vs h2
        if t >= 0.36:
            roll_p = ease(t, 0.36, 0.62)
            n_r = max(2, int(40 * roll_p))
            pts_h1, pts_h2 = [], []
            for i in range(n_r + 1):
                tm = (i / 40.0) * 20.0
                tk1 = 388.0 - 18.0 * (1.0 - math.exp(-tm / 5.0))
                tk2 = 358.0 - 28.0 * (1.0 - math.exp(-tm / 4.5))
                pts_h1.append((s(tx(tm)), s(ty(tk1))))
                pts_h2.append((s(tx(tm)), s(ty(tk2))))
            d.line(pts_h1, fill=MUTED, width=s(1.8))
            d.line(pts_h2, fill=RED, width=s(2.4))
            if roll_p > 0.65:
                text(d, (tx(19.5), ty(371) - 14), "coolant -8K | h1 (safe)", F_MONO_S, MUTED, anchor="ra")
                text(d, (tx(19.5), ty(330) - 14), "coolant -8K | h2 (QUENCH!)", F_MONO_B, RED, anchor="ra")

        # 4) 3-min Probe + Certified Recovery
        if t >= 0.64:
            probe_p = ease(t, 0.64, 0.88)
            d.rectangle([s(x0), s(112), s(xp), s(352)], fill=(219, 234, 254))
            text(d, ((x0 + xp) / 2, 336), "3m Probe", F_MONO_S, BLUE, anchor="ma")
            n_p = max(2, int(40 * probe_p))
            pts_rec = []
            for i in range(n_p + 1):
                tm = (i / 40.0) * 20.0
                tk = 388.0 - 20.0 * (1.0 - math.exp(-tm / 4.0))
                pts_rec.append((s(tx(tm)), s(ty(tk))))
            d.line(pts_rec, fill=GREEN, width=s(3.0))
            if probe_p > 0.6:
                text(d, (tx(19.5), ty(366) - 16), "After 3m probe: CERTIFIED (368K)", F_MONO_B, GREEN, anchor="ra")

        # Right Panel: Step-by-Step Decision Trace
        rrect(d, (576, 82, 932, 378), 8, PANEL, BORDER)
        text(d, (592, 95), "RISKGATE TWO-STAGE DECISION TRACE", F_LABEL, MUTED)

        # Card 1
        rrect(d, (590, 116, 918, 172), 6, SUBTLE if t < 0.2 else GREEN_BG, BORDER if t < 0.2 else GREEN_BORDER)
        text(d, (600, 125), "1. Plausible set H_tau = {h1: valve, h2: sensor}", F_MONO_B, INK)
        text(d, (600, 145), "Both match the 10-min rise within tau = 1.68", F_SMALL, MUTED)

        # Card 2
        if t >= 0.34:
            rrect(d, (590, 180, 918, 236), 6, RED_BG, RED_BORDER)
            text(d, (600, 189), "2. LLM Top-1 `coolant_sp -8 K` BLOCKED", F_MONO_B, RED)
            text(d, (600, 209), "Safe under h1 (+14K), quenches under h2 (-4.8K)", F_SMALL, MUTED)

        # Card 3
        if t >= 0.58:
            rrect(d, (590, 244, 918, 300), 6, BLUE_BG, BLUE_BORDER)
            text(d, (600, 253), "3. Stage 1 Verdict: PROBE (backup_sensor_3m)", F_MONO_B, BLUE)
            text(d, (600, 273), "Safe under h1 & h2; separates trajectories 100%", F_SMALL, MUTED)

        # Card 4
        if t >= 0.76:
            rrect(d, (590, 308, 918, 366), 6, GREEN, GREEN)
            text(d, (600, 317), "4. Stage 2 Verdict: CERTIFY recovery (+18.2K)", F_MONO_B, (255, 255, 255))
            text(d, (600, 338), "Hand-overs cut 27.8% -> 12.2% (unsafe 4.8% <= 5%)", F_SMALL, (220, 252, 231))

        text(
            d,
            (28, 398),
            "500-episode CSTR benchmark: Uncalibrated twin gate = 44.6% runaways · Top-1 conformal = 0% certified · RiskGate + Probe = 4.8% unsafe, 87.8% certified.",
            F_SMALL,
            MUTED,
        )

        frames.append(im)
        durations.append(110 if k < n_frames - 1 else 1500)

    save_gif(frames, durations, OUT / "2_cstr_reactor_probe.gif")


# ==============================================================================================
# GIF 3: Kubernetes Incident SRE Agent — Ranked Candidate Walk-Down
# ==============================================================================================
def build_gif_3() -> None:
    frames, durations = [], []
    n_frames = 44
    for k in range(n_frames):
        t = k / (n_frames - 1)
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)

        draw_header(
            d,
            "Example 2 · Kubernetes SRE Agent — Ranked Walk-Down Under Root-Cause Ambiguity",
            "Alert: checkout-api p95 = 920 ms (SLA <= 500 ms), error_rate = 6.4%. Deploy v2.14 rolled out 9m ago AND db_cpu = 91%.",
            "CHECK TIME = 0.08 ms",
            t,
            GREEN,
        )

        # Left Panel: Telemetry & Plausible Root Causes
        rrect(d, (28, 82, 288, 378), 8, PANEL, BORDER)
        text(d, (42, 95), "LIVE CLUSTER CONTEXT & H_tau", F_LABEL, MUTED)

        rrect(d, (42, 114, 274, 182), 6, SUBTLE, BORDER)
        text(d, (52, 123), "p95_latency: 920 ms (SLA<=500)", F_MONO_S, RED)
        text(d, (52, 141), "error_rate:  6.4%   (SLA<=2%)", F_MONO_S, RED)
        text(d, (52, 159), "db_cpu / pod: 91% / 78%", F_MONO_S, INK)

        text(d, (42, 194), "PLAUSIBLE CAUSES H_tau(x)", F_LABEL, MUTED)
        causes = [
            ("h1: bad_canary_deploy", "r = 0.62 <= tau (fits errors)"),
            ("h2: db_lock_contention", "r = 0.89 <= tau (fits db_cpu)"),
            ("h3: noisy_neighbor_cpu", "r = 1.31 <= tau (plausible)"),
        ]
        for idx, (cname, csub) in enumerate(causes):
            y0 = 212 + idx * 50
            rrect(d, (42, y0, 274, y0 + 44), 5, GREEN_BG if t >= 0.12 else SUBTLE, GREEN_BORDER if t >= 0.12 else BORDER)
            text(d, (52, y0 + 7), cname, F_MONO_B, INK)
            text(d, (52, y0 + 24), csub, F_SMALL, MUTED)

        # Center Panel: Ranked Walk-Down Matrix
        rrect(d, (302, 82, 692, 378), 8, PANEL, BORDER)
        text(d, (318, 95), "AGENT RANKED PROPOSALS x PLAUSIBLE CAUSES", F_LABEL, MUTED)

        text(d, (318, 120), "CANDIDATE ACTION", F_MONO_S, MUTED)
        text(d, (545, 120), "h1", F_MONO_B, MUTED, anchor="ma")
        text(d, (600, 120), "h2", F_MONO_B, MUTED, anchor="ma")
        text(d, (655, 120), "h3", F_MONO_B, MUTED, anchor="ma")
        d.line([(s(318), s(136)), (s(678), s(136))], fill=BORDER, width=s(1))

        # Row 1: HardRule block (active at t in [0.18, 0.34))
        s1_on = t >= 0.18
        rrect(d, (316, 144, 678, 194), 5, RED_BG if s1_on else SUBTLE, RED if 0.18 <= t < 0.35 else (RED_BORDER if s1_on else BORDER), 2 if 0.18 <= t < 0.35 else 1)
        text(d, (326, 152), "#1 drop_prod_replica_table()", F_MONO_B, RED if s1_on else INK)
        text(d, (326, 171), "Blocked by HardRule('no_destructive_ddl')", F_SMALL, MUTED)
        if s1_on:
            text(d, (668, 160), "HARD BLOCK", F_MONO_B, RED, anchor="ra")

        # Row 2: Rollback (active at t in [0.35, 0.52))
        s2_on = t >= 0.35
        rrect(d, (316, 202, 678, 252), 5, AMBER_BG if s2_on else SUBTLE, AMBER if 0.35 <= t < 0.52 else (AMBER_BORDER if s2_on else BORDER), 2 if 0.35 <= t < 0.52 else 1)
        text(d, (326, 210), "#2 rollback_deploy(v2.13)", F_MONO_B, INK)
        text(d, (326, 229), "Fixes h1, leaves DB lock h2 at 890ms", F_SMALL, MUTED)
        if s2_on:
            text(d, (545, 218), "+0.38", F_MONO_S, GREEN, anchor="ma")
            text(d, (600, 218), "-0.78", F_MONO_B, RED, anchor="ma")
            text(d, (655, 218), "-0.41", F_MONO_S, RED, anchor="ma")

        # Row 3: Scale replicas +4 (active at t in [0.52, 0.68))
        s3_on = t >= 0.52
        rrect(d, (316, 260, 678, 310), 5, RED_BG if s3_on else SUBTLE, RED if 0.52 <= t < 0.68 else (RED_BORDER if s3_on else BORDER), 2 if 0.52 <= t < 0.68 else 1)
        text(d, (326, 268), "#3 scale_replicas(+4)", F_MONO_B, INK)
        text(d, (326, 287), "More pods exhaust DB pool under h2!", F_SMALL, MUTED)
        if s3_on:
            text(d, (545, 276), "+0.15", F_MONO_S, GREEN, anchor="ma")
            text(d, (600, 276), "-1.12", F_MONO_B, RED, anchor="ma")
            text(d, (655, 276), "+0.29", F_MONO_S, GREEN, anchor="ma")

        # Row 4: Shed noncritical traffic 25% (Certified at t >= 0.68!)
        s4_on = t >= 0.68
        rrect(d, (316, 318, 678, 368), 5, GREEN_BG if s4_on else SUBTLE, GREEN if s4_on else BORDER, 2 if s4_on else 1)
        text(d, (326, 326), "#4 shed_noncritical_traffic(25%)", F_MONO_B, GREEN if s4_on else INK)
        text(d, (326, 345), "Relieves pods & DB; safe for all h in H_tau", F_SMALL, MUTED)
        if s4_on:
            text(d, (545, 334), "+0.34", F_MONO_B, GREEN, anchor="ma")
            text(d, (600, 334), "+0.28", F_MONO_B, GREEN, anchor="ma")
            text(d, (655, 334), "+0.31", F_MONO_B, GREEN, anchor="ma")

        # Right Panel: Certified Output & Audit
        rrect(d, (706, 82, 932, 378), 8, PANEL, BORDER)
        text(d, (720, 95), "CERTIFIED OUTPUT", F_LABEL, MUTED)

        if t >= 0.70:
            rrect(d, (720, 114, 918, 174), 6, GREEN, GREEN)
            text(d, (732, 126), "CERTIFY #4", F_MONO_B, (255, 255, 255))
            text(d, (732, 147), "s = +0.28 >= lam (+0.14)", F_MONO_S, (220, 252, 231))

            text(d, (720, 190), "CONTRASTIVE AUDIT LOG", F_LABEL, MUTED)
            rrect(d, (720, 208, 918, 364), 6, SUBTLE, BORDER)
            text(d, (730, 220), "Skipped #2 rollback:", F_MONO_S, RED)
            text(d, (730, 238), "under h2 (db_lock),", F_SMALL, MUTED)
            text(d, (730, 254), "p95 margin = -0.78.", F_SMALL, MUTED)
            text(d, (730, 278), "Certified #4 shed_25%:", F_MONO_S, GREEN)
            text(d, (730, 296), "worst margin +0.28", F_SMALL, MUTED)
            text(d, (730, 312), "across {h1, h2, h3}.", F_SMALL, MUTED)
            text(d, (730, 338), "sha256: 9f4c...8a1b", F_MONO_S, LIGHT_INK)
        else:
            rrect(d, (720, 114, 918, 174), 6, SUBTLE, BORDER)
            text(d, (732, 136), "Scanning candidates...", F_MONO_S, MUTED)

        text(
            d,
            (28, 398),
            "Why ranked walk-down matters: Top-1 CRC only checks #2 (52.0% safe execution). RiskGate walks down the list to #4 (84.4% safe execution).",
            F_SMALL,
            MUTED,
        )

        frames.append(im)
        durations.append(110 if k < n_frames - 1 else 1500)

    save_gif(frames, durations, OUT / "3_k8s_sre_walkdown.gif")


# ==============================================================================================
# GIF 4: Refund Agent Multi-Risk Calibration & Proposition 1
# ==============================================================================================
def build_gif_4() -> None:
    frames, durations = [], []
    n_frames = 44
    for k in range(n_frames):
        t = k / (n_frames - 1)
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)

        draw_header(
            d,
            "Example 3 · Refund Agent (Multi-Risk Control) & Why Top-1 Gates Fail (Prop. 1)",
            "Left: Top-1 misdiagnosis blows up the conformal tube (26.6 K vs 0.96 K). Right: Jointly bounding policy & dollar risk.",
            "THEOREM 1 + PROP. 1",
            t,
            AMBER,
        )

        # Left Panel: Proposition 1 (26.6 K vs 0.96 K)
        rrect(d, (28, 82, 468, 378), 8, PANEL, BORDER)
        text(d, (44, 95), "PROPOSITION 1 · WHY TOP-1 CONFORMAL COLLAPSES", F_LABEL, MUTED)
        text(d, (44, 115), "When P(h_hat_1 != h*) > alpha, top-1 quantiles explode:", F_SMALL, MUTED)

        b1_p = ease(t, 0.08, 0.38)
        text(d, (44, 144), "Top-1 Conformal Gate (uses only best guess h_hat_1)", F_MONO_B, RED)
        rrect(d, (44, 164, 450, 198), 5, RED_BG, RED_BORDER)
        bar1_w = max(8.0, 392.0 * b1_p)
        rrect(d, (44, 164, 44 + bar1_w, 198), 5, RED)
        if b1_p > 0.6:
            text(d, (54, 174), "26.6 K tube  ->  0.0% certified (100% ASK)", F_MONO_B, (255, 255, 255))

        b2_p = ease(t, 0.32, 0.56)
        text(d, (44, 214), "RiskGate Hypothesis Set H_tau (covers h* 91.2%)", F_MONO_B, GREEN)
        rrect(d, (44, 234, 450, 268), 5, GREEN_BG, GREEN_BORDER)
        bar2_w = max(8.0, 26.0 * b2_p)
        rrect(d, (44, 234, 44 + bar2_w, 268), 5, GREEN)
        if b2_p > 0.4:
            text(d, (80, 244), "0.96 K tube (28x sharper)  ->  87.8% certified!", F_MONO_B, GREEN)

        rrect(d, (44, 286, 450, 362), 6, SUBTLE, BORDER)
        text(d, (56, 298), "Why? Separating epistemic fault ambiguity (H_tau)", F_BOLD, INK)
        text(d, (56, 318), "from simulator error keeps tubes tight (0.96 K) instead", F_SMALL, MUTED)
        text(d, (56, 336), "of inflating them to swallow wrong diagnoses (26.6 K).", F_SMALL, MUTED)

        # Right Panel: Refund Agent Multi-Risk Calibration
        rrect(d, (484, 82, 932, 378), 8, PANEL, BORDER)
        text(d, (500, 95), "REFUND AGENT · JOINT MULTI-RISK THRESHOLD", F_LABEL, MUTED)
        text(d, (500, 115), "Claim: $185 'damaged item', VIP tier, 2 prior claims, no photo", F_SMALL, MUTED)

        rrect(d, (500, 138, 916, 196), 6, SUBTLE, BORDER)
        text(d, (512, 148), "Risk 1 (Policy violation <= 5.0%):  lam_safety = +0.08", F_MONO_S, GREEN)
        text(d, (512, 168), "Risk 2 (Dollar overpay   <= 3.0%):  lam_cost   = +0.19", F_MONO_S, BLUE)
        if t >= 0.40:
            text(d, (906, 158), "lam_hat = +0.19", F_MONO_B, INK, anchor="ra")

        if t >= 0.46:
            rrect(d, (500, 208, 702, 278), 6, GREEN_BG, GREEN_BORDER)
            text(d, (510, 218), "h1: genuine_damage", F_MONO_B, GREEN)
            text(d, (510, 238), "Refund $185 is compliant", F_SMALL, MUTED)
            text(d, (510, 256), "margin = +0.45 >= +0.19", F_MONO_S, GREEN)

            rrect(d, (714, 208, 916, 278), 6, RED_BG, RED_BORDER)
            text(d, (724, 218), "h2: policy_abuse", F_MONO_B, RED)
            text(d, (724, 238), "Unverified >$150 leaks $185", F_SMALL, MUTED)
            text(d, (724, 256), "margin = -0.52 < +0.19", F_MONO_B, RED)

        if t >= 0.68:
            rrect(d, (500, 290, 916, 362), 6, BLUE_BG, BLUE, 1.8)
            text(d, (512, 302), "Verdict: PROBE -> request_photo_and_carrier_scan()", F_MONO_B, BLUE)
            text(d, (512, 323), "Zero dollar leakage during probe; once carrier scan", F_SMALL, INK)
            text(d, (512, 340), "confirms damage, h2 is ruled out -> CERTIFY $185.", F_SMALL, GREEN)

        text(
            d,
            (28, 398),
            "Across all 3 domains (Reactor, Kubernetes SRE, Refund Agent), RiskGate holds unsafe execution below alpha = 5% on 100% of random splits.",
            F_SMALL,
            MUTED,
        )

        frames.append(im)
        durations.append(110 if k < n_frames - 1 else 1500)

    save_gif(frames, durations, OUT / "4_refunds_multirisk.gif")


def main() -> None:
    build_gif_1()
    build_gif_2()
    build_gif_3()
    build_gif_4()


if __name__ == "__main__":
    main()
