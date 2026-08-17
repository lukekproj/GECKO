"""
Original schematic for the GECKO JOSS paper.

Panels A and B contrast standard frontoparallel stimulus presentation (constant
viewing depth, scalar visual angle) with KINARM transverse-plane presentation
(variable depth, eye-based 3D spherical coordinates). Panel C summarises the
processing chain GECKO implements on top of that geometry.

Geometry and equations follow Singh et al. (2016), J. NeuroEngineering and
Rehabilitation 13:10.

  Frontal plane (Eq. 1):   tan(beta/2) = a / (2 b)

  Eye-based Cartesian (Eq. 2):
      [x', y', z']^T = [0, 0, H]^T + R * [x, y, 0]^T
  i.e. the EYE is the origin of the X'Y'Z' frame, Z' points downward toward the
  workspace, and every point of the stimulus plane therefore has z' = H (the eye
  height above the plane). z' is NOT zero.

  Spherical (Eq. 3):
      rho   = sqrt(x'^2 + y'^2 + z'^2)      radial eye-to-POR distance
      theta = arctan(y'/x')                 azimuth in the X'Y' plane, from X'
      phi   = arccos(z'/rho)                elevation from the +Z' axis

Because theta is an azimuth in the horizontal plane, panel B is drawn in an
axonometric projection: a side elevation cannot display it. theta is drawn at
the foot point F (directly below the eye) rather than at the eye itself; the two
are identical because translation along Z' leaves x' and y' unchanged.

Deterministic vector drawing (no AI image generation): authorship is the team's.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc, FancyArrowPatch, Polygon

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "mathtext.fontset": "dejavusans",
})

C_PLANE = "#3a6ea5"      # stimulus plane / display
C_RHO   = "#2a9d8f"      # gaze vector
C_ANGLE = "#e76f51"      # angles
C_POR   = "#d62728"      # point of regard
C_AXIS  = "#22303a"      # text / axes
C_DASH  = "#8a8a8a"      # construction lines
C_BOX   = "#f4f1ea"      # equation box fill
EQ_BBOX = dict(boxstyle="round,pad=0.40", fc=C_BOX, ec=C_AXIS, lw=0.9)

fig = plt.figure(figsize=(11.0, 9.2))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.68, 0.52],
              hspace=0.10, wspace=0.06)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, :])

for ax in (axA, axB):
    ax.set_aspect("equal")
    ax.axis("off")
axC.axis("off")


def eye_marker(ax, p, s=150):
    ax.scatter([p[0]], [p[1]], s=s, color=C_AXIS, zorder=8)
    ax.scatter([p[0]], [p[1]], s=s * 0.26, color="white", zorder=9)


# =============================================================================
# Panel A: frontoparallel display -- constant depth, scalar visual angle
# =============================================================================
axA.set_xlim(-3.1, 7.7)   # same spans as panel B -> identical drawing scale
axA.set_ylim(-5.70, 5.10)

E = np.array([0.0, 1.4])
xs = 4.3                      # screen distance b
a_lo, a_hi = 0.55, 2.25       # stimulus extent on the screen

# vertical screen (frontal plane)
axA.plot([xs, xs], [-0.5, 3.2], color=C_PLANE, lw=3.2, solid_capstyle="round")
axA.text(xs + 0.20, 3.15, "frontoparallel\ndisplay", color=C_PLANE,
         fontsize=10, ha="left", va="top")

# line of sight + viewing distance b
axA.plot([E[0], xs], [E[1], E[1]], color=C_DASH, lw=1.4, ls=(0, (5, 4)))
axA.annotate("", xy=(xs, 0.05), xytext=(E[0], 0.05),
             arrowprops=dict(arrowstyle="<->", color=C_AXIS, lw=1.1))
axA.text(xs / 2, -0.16, r"$b$  (assumed fixed)", ha="center", va="top",
         color=C_AXIS, fontsize=11)

# rays subtending the stimulus + visual angle beta
for yy in (a_lo, a_hi):
    axA.plot([E[0], xs], [E[1], yy], color=C_RHO, lw=1.8)
ang_lo = np.degrees(np.arctan2(a_lo - E[1], xs - E[0]))
ang_hi = np.degrees(np.arctan2(a_hi - E[1], xs - E[0]))
axA.add_patch(Arc(tuple(E), 1.7, 1.7, angle=0, theta1=ang_lo, theta2=ang_hi,
                  color=C_ANGLE, lw=2.0, zorder=6))
axA.text(E[0] + 1.02, E[1] - 0.02, r"$\beta$", color=C_ANGLE, fontsize=15,
         ha="left", va="center")

# stimulus extent a
axA.annotate("", xy=(xs + 0.26, a_hi), xytext=(xs + 0.26, a_lo),
             arrowprops=dict(arrowstyle="<->", color=C_AXIS, lw=1.2))
axA.text(xs + 0.40, (a_lo + a_hi) / 2, r"$a$", ha="left", va="center",
         color=C_AXIS, fontsize=13)

eye_marker(axA, E)
axA.text(E[0] - 0.12, E[1] + 0.26, "Eye", ha="left", va="bottom",
         color=C_AXIS, fontsize=11)

axA.text(2.3, -1.75, "constant depth  \u2192  one scalar visual angle",
         ha="center", va="center", color=C_AXIS, fontsize=10.5)
axA.text(2.3, -3.30, r"$\tan\left(\dfrac{\beta}{2}\right)=\dfrac{a}{2b}$",
         ha="center", va="center", color=C_AXIS, fontsize=15, bbox=EQ_BBOX)
axA.set_title("(A) Frontoparallel display \u2014 standard eye tracking",
              fontsize=11.5, loc="left", color=C_AXIS, pad=6)

# =============================================================================
# Panel B: KINARM transverse plane -- eye-based 3D spherical coordinates
# =============================================================================
# axonometric projection: X' lateral-right, Y' anterior (into page), Z' downward
EX = np.array([1.00, 0.00])
EY = np.array([0.46, 0.33])
EZ = np.array([0.00, -1.00])


def proj(p):
    p = np.asarray(p, dtype=float)
    return p[0] * EX + p[1] * EY + p[2] * EZ


def slerp(u, v, t):
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    om = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    if om < 1e-9:
        return u
    return (np.sin((1 - t) * om) * u + np.sin(t * om) * v) / np.sin(om)


def arc3d(ax, center, u, v, radius, n=80, **kw):
    pts = np.array([proj(np.asarray(center, float) + radius * slerp(u, v, t))
                    for t in np.linspace(0, 1, n)])
    ax.plot(pts[:, 0], pts[:, 1], **kw)
    return pts


H = 3.05                                  # eye height above the stimulus plane
EYE = np.array([0.0, 0.0, 0.0])           # eye is the ORIGIN of X'Y'Z'
F = np.array([0.0, 0.0, H])               # foot of the perpendicular
POR = np.array([3.30, 2.05, H])           # main point of regard  (z' = H)

axB.set_xlim(-3.9, 6.9)
axB.set_ylim(-7.75, 3.05)

# --- transverse stimulus plane (a parallelogram in this projection) ----------
corners = [(-2.7, -1.25, H), (5.3, -1.25, H), (5.3, 3.45, H), (-2.7, 3.45, H)]
axB.add_patch(Polygon([proj(c) for c in corners], closed=True,
                      facecolor=C_PLANE, alpha=0.13, edgecolor=C_PLANE,
                      lw=2.2, zorder=1))
axB.text(*(proj((-2.7, -1.25, H)) + np.array([0.10, -0.34])),
         r"transverse stimulus plane  ($z'=H$)", color=C_PLANE,
         fontsize=10, ha="left", va="top")

# --- Z' axis: eye -> foot point -> beyond -----------------------------------
axB.plot(*zip(proj(EYE), proj(F)), color=C_DASH, lw=1.5, ls=(0, (5, 4)),
         zorder=3)
axB.add_patch(FancyArrowPatch(proj(F), proj((0, 0, H + 0.72)),
                              arrowstyle="-|>", mutation_scale=13, lw=1.5,
                              color=C_DASH, shrinkA=0, shrinkB=0, zorder=3))
axB.text(*(proj((0, 0, H + 0.79)) + np.array([0.10, -0.16])), r"$Z'$",
         color=C_DASH, fontsize=12.5, ha="left", va="top")

# H == z' annotation on the vertical drop
axB.annotate("", xy=proj(F) + np.array([-0.30, 0]),
             xytext=proj(EYE) + np.array([-0.30, 0]),
             arrowprops=dict(arrowstyle="<->", color=C_AXIS, lw=1.1))
axB.text(*(proj((0, 0, H / 2)) + np.array([-0.44, 0])),
         r"$H=z'$", color=C_AXIS, fontsize=12.5, ha="right", va="center")

# right-angle marker at the foot point
rp = [proj(F + np.array([0, 0, -0.32])), proj(F + np.array([0.30, 0, -0.32])),
      proj(F + np.array([0.30, 0, 0]))]
axB.plot([p[0] for p in rp], [p[1] for p in rp], color=C_AXIS, lw=1.0, zorder=4)

# --- X' and Y' axes, drawn at the foot point (translated copies) -------------
for vec, lab in (((2.05, 0, 0), r"$X'$"), ((0, 1.85, 0), r"$Y'$")):
    tip = F + np.array(vec)
    axB.add_patch(FancyArrowPatch(proj(F), proj(tip), arrowstyle="-|>",
                                  mutation_scale=12, lw=1.4, color=C_AXIS,
                                  shrinkA=0, shrinkB=0, alpha=0.85, zorder=4))
    off = np.array([0.12, -0.30]) if lab == r"$X'$" else np.array([0.10, 0.10])
    axB.text(*(proj(tip) + off), lab, color=C_AXIS, fontsize=12.5,
             ha="left", va="center")

# --- gaze vector rho, its horizontal projection, and a second POR ------------
axB.plot(*zip(proj(F), proj(POR)), color=C_DASH, lw=1.3, ls=(0, (3, 3)),
         zorder=4)
axB.add_patch(FancyArrowPatch(proj(EYE), proj(POR), arrowstyle="-|>",
                              mutation_scale=16, lw=2.6, color=C_RHO,
                              shrinkA=0, shrinkB=0, zorder=6))
axB.scatter(*proj(POR), s=78, color=C_POR, zorder=7)

axB.text(*(proj(POR * 0.55) + np.array([0.22, 0.20])), r"$\rho$",
         color=C_RHO, fontsize=15, ha="left", va="bottom")
axB.text(*(proj(POR) + np.array([0.20, 0.10])),
         "POR $(x,y)$\n" + r"$\rightarrow(x',y',z'\!=\!H)$",
         color=C_POR, fontsize=10, ha="left", va="bottom", linespacing=1.35)

# --- phi: from the +Z' axis to the gaze vector, at the eye -------------------
arc3d(axB, EYE, np.array([0, 0, 1.0]), POR, 1.30,
      color=C_ANGLE, lw=2.2, zorder=7)
mid_phi = slerp(np.array([0, 0, 1.0]), POR, 0.52) * 1.58
axB.text(*proj(mid_phi), r"$\varphi$", color=C_ANGLE, fontsize=15,
         ha="center", va="center", zorder=8)

# --- theta: azimuth in the X'Y' plane, drawn at the foot point ---------------
hor = np.array([POR[0], POR[1], 0.0])
arc3d(axB, F, np.array([1.0, 0, 0]), hor, 1.50,
      color=C_ANGLE, lw=2.2, zorder=5)
mid_th = F + slerp(np.array([1.0, 0, 0]), hor, 0.5) * 1.88
axB.text(*proj(mid_th), r"$\theta$", color=C_ANGLE, fontsize=15,
         ha="center", va="center", zorder=6)

eye_marker(axB, proj(EYE))
axB.text(*(proj(EYE) + np.array([0.16, 0.20])),
         "Eye = origin of $X'Y'Z'$", ha="left", va="bottom",
         color=C_AXIS, fontsize=10.5)

axB.text(1.3, -5.02, "variable depth  \u2192  eye-based 3D spherical coordinates",
         ha="center", va="center", color=C_AXIS, fontsize=10.5)
eq_txt = (r"$\rho=\sqrt{x'^2+y'^2+z'^2}$" + "\n"
          r"$\theta=\arctan(y'/x')$   (azimuth in $X'Y'$)" + "\n"
          r"$\varphi=\arccos(z'/\rho)$   (from $+Z'$)")
axB.text(1.3, -6.60, eq_txt, ha="center", va="center", color=C_AXIS,
         fontsize=11.0, linespacing=1.55, bbox=EQ_BBOX)
axB.set_title("(B) Transverse plane with depth \u2014 KINARM",
              fontsize=11.5, loc="left", color=C_AXIS, pad=6)

# =============================================================================
# Panel C: what GECKO computes on top of this geometry
# =============================================================================
axC.set_xlim(0, 1)
axC.set_ylim(0, 1)

steps = [
    (0.105, ".kinarm file\ngaze POR $(x,y)$\nin the transverse plane"),
    (0.335, "Eye-based spherical\ncoordinates $(\\rho,\\theta,\\varphi)$\nreferenced to the eye"),
    (0.578, "Gaze angular velocity\n"
            r"$v_{\|\varphi,\theta\|}=\sqrt{\left(\dot{\theta}\sin\varphi\right)^{2}+\dot{\varphi}^{2}}$"),
    (0.862, "Researcher-in-the-loop\nlabeling \u2192 per-frame\nsaccade / pursuit / fixation"),
]
for i, (x, label) in enumerate(steps):
    fc = "#eef3f7" if i == 0 else C_BOX
    axC.text(x, 0.42, label, ha="center", va="center", fontsize=10,
             color=C_AXIS, linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.52", fc=fc, ec=C_AXIS, lw=1.0))

for x0, x1 in ((0.209, 0.229), (0.438, 0.462), (0.704, 0.728)):
    axC.annotate("", xy=(x1, 0.42), xytext=(x0, 0.42),
                 arrowprops=dict(arrowstyle="-|>", color=C_AXIS, lw=1.5))

axC.plot([0.233, 0.988], [0.885, 0.885], color=C_RHO, lw=1.6,
         solid_capstyle="round")
for xt in (0.233, 0.988):
    axC.plot([xt, xt], [0.805, 0.885], color=C_RHO, lw=1.6,
             solid_capstyle="round")
axC.text(0.610, 0.912, "implemented by GECKO", ha="center", va="bottom",
         fontsize=10.5, color=C_RHO)
axC.set_title("(C) From transverse-plane gaze data to classified gaze events",
              fontsize=11.5, loc="left", color=C_AXIS, pad=12)

fig.savefig("kinarm_gaze_geometry.png", dpi=300, bbox_inches="tight",
            facecolor="white")
fig.savefig("kinarm_gaze_geometry.pdf", bbox_inches="tight", facecolor="white")
print("wrote kinarm_gaze_geometry.png and .pdf")
