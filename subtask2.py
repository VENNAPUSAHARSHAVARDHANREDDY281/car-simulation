

import argparse
import time

import cv2
import numpy as np
import pybullet as p

from simulation_setup import setup_simulation


# ─────────────────────────────────────────────────────────────────────────────
# Configuration & gains 
# ─────────────────────────────────────────────────────────────────────────────

# ── Camera / image ────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 320, 240          

# ── LK optical flow ──────────────────────────────────────────────────────────
LK_WIN        = (25, 25)          
LK_LEVELS     = 3
LK_CRITERIA   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
LK_MIN_FLOW   = 0.05              
FEAT_MAX      = 150
FEAT_QUALITY  = 0.3
FEAT_DIST     = 7
REDETECT_EVERY = 15               # re-detect every N steps (not every frame)
MIN_FEATURES  = 5

# ── Force gains (Eq. 8, 9) ───────────────────────────────────────────────────
ALPHA = 5.0    # attractive gain  α
BETA  = 30.0   # obstacle gain    β 
GAMMA = 30.0  # road gain        (Morse potential)

# ── Morse road potential (Table I) ───────────────────────────────────────────
ROAD_HALF_WIDTH = 1.16   # metres — from simulation_setup halfExtents
A_MORSE         = 0.5
B_MORSE         = 1.0

# ── Boundary condition zones ─────────────────────────────────────────────────
BC_WARN = 0.50   # warning zone  (m)
BC_CRIT = 0.25   # critical zone (m)
BC_EMRG = 0.10   # emergency zone (m)

# ── GTSMC (Eq. 28–30) ────────────────────────────────────────────────────────
U0      = 0.5    # max steering rate
EPSILON = 0.1    # boundary layer for saturation (snippet-3's smoothing idea)
CR      = 3.0    # rotational manifold gain  (Eq. 28)
ALPHA_F = 0.30   # IIR filter for steering

# ── Longitudinal ─────────────────────────────────────────────────────────────
TARGET_VEL = 15.0  # wheel velocity (rad/s) — paper ~20 km/h

# ── World / goal ─────────────────────────────────────────────────────────────
GOAL_X     = 30.0
GOAL_WORLD = np.array([GOAL_X, 0.0, 0.40])

# ── Display ───────────────────────────────────────────────────────────────────
DISPLAY_SCALE = 2              # upscale 640×480 → 1280×960
WINDOW_TITLE  = "VPF Navigation"


# ────────────────────────────────────────────────────────────────────────────
# On-board monocular camera model
#
# This class captures grayscale images from the car-mounted PyBullet camera.
# It also:
# - fixes projection aspect ratio,
# - hides the vehicle body during rendering to avoid self-generated features,
# - projects 3D world goal coordinates into image pixels so the goal remains
#   visually consistent while the car turns.
#
# This matches the paper requirement:
# "capture RGB frames from agent camera and compute optical flow."
# ─────────────────────────────────────────────────────────────────────────────
class CarCamera:
   

    CAM_FWD  = np.array([0.3, 0.0, 0.5])   # camera offset from CoM
    LOOK_FWD = np.array([1.3, 0.0, 0.5])   # look-at offset from CoM
    FOV      = 60.0
    NEAR     = 0.1
    FAR      = 80.0

    def __init__(self, car_id: int):
        self.car_id   = car_id 
        self.proj_mat = p.computeProjectionMatrixFOV(
            self.FOV,
            WIDTH / HEIGHT,    
            self.NEAR, self.FAR)

    def _pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        pos = np.array(pos)
        cam  = pos + R @ self.CAM_FWD
        look = pos + R @ self.LOOK_FWD
        return cam, look

    def _view_matrix(self):
        cam, look = self._pose()
        return p.computeViewMatrix(cam.tolist(), look.tolist(), [0, 0, 1])

    def get_frame(self) -> np.ndarray:
        """Returns (H,W) uint8 grayscale. Hides car body during capture."""
        view_mat   = self._view_matrix()
        num_joints = p.getNumJoints(self.car_id)
        all_links  = [-1] + list(range(num_joints))

        try:
            for lnk in all_links:
                p.changeVisualShape(self.car_id, lnk, rgbaColor=[0, 0, 0, 0])
        except Exception:
            pass

        _, _, rgb, _, _ = p.getCameraImage(
            WIDTH, HEIGHT, view_mat, self.proj_mat,
            renderer=p.ER_TINY_RENDERER)

        try:
            for lnk in all_links:
                p.changeVisualShape(self.car_id, lnk, rgbaColor=[1, 1, 1, 1])
        except Exception:
            pass

        rgb_arr = np.array(rgb, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3]
        return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

    def project_world_point(self, world_xyz: np.ndarray) -> np.ndarray:
        """
        Project a 3-D world point into pixel coordinates using the
        CURRENT view+projection matrices.  Keeps the goal reticle
        world-anchored as the car steers (fixes the rotating-goal bug).
        """
        V = np.array(self._view_matrix()).reshape(4, 4).T
        P = np.array(self.proj_mat).reshape(4, 4).T
        wp   = np.array([*world_xyz, 1.0])
        clip = P @ V @ wp
        if abs(clip[3]) < 1e-6:
            return np.array([WIDTH / 2.0, HEIGHT * 0.3])
        ndc = clip[:3] / clip[3]
        px  = ( ndc[0] + 1.0) * 0.5 * WIDTH
        py  = (-ndc[1] + 1.0) * 0.5 * HEIGHT
        return np.array([
            float(np.clip(px, 0, WIDTH  - 1)),
            float(np.clip(py, 0, HEIGHT - 1)),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Focus of Expansion (FOE) estimation
#
# FOE represents the apparent motion center of ego-vehicle movement.
# In pure forward motion, background optical-flow vectors diverge from FOE.
#
# Least-squares solution:
# FOE = (AᵀA)^(-1) Aᵀb
#
# Used to:
# - estimate vehicle heading in image plane
# - separate obstacle motion from background ego-motion
#
# Low-magnitude flow vectors are ignored because they are numerically unstable.
# ─────────────────────────────────────────────────────────────────────────────
def solve_foe(pts: np.ndarray, flow: np.ndarray):
    
    if len(pts) < MIN_FEATURES:
        return None

    A_rows, b_rows = [], []
    for pt, fl in zip(pts.reshape(-1, 2), flow.reshape(-1, 2)):
        vx, vy = float(fl[0]), float(fl[1])
        if np.hypot(vx, vy) < LK_MIN_FLOW:   # snippet-3 filter ✓
            continue
        x, y = float(pt[0]), float(pt[1])
        A_rows.append([vy, -vx])
        b_rows.append(vy * x - vx * y)

    if len(A_rows) < 4:
        return None

    foe, _, _, _ = np.linalg.lstsq(
        np.array(A_rows, dtype=np.float32),
        np.array(b_rows, dtype=np.float32),
        rcond=None)
    return np.clip(foe, [0, 0], [WIDTH, HEIGHT])


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle detection using sparse optical flow
#
# This stage identifies obstacle features by measuring deviation from
# ideal radial flow around the FOE.
#
# Steps:
# 1. Compute TTC (time-to-contact) for each tracked feature
# 2. Measure angular residual between actual flow and expected radial flow
# 3. Apply Otsu thresholding to separate obstacle points
# 4. Build binary obstacle plane O(x,y,t)
# 5. Apply Gaussian smoothing and spatial gradient to obtain g(x,y,t)
#
# This directly follows paper equations (4–6).
# ─────────────────────────────────────────────────────────────────────────────
def detect_obstacles(pts: np.ndarray, flow: np.ndarray,
                     foe: np.ndarray):
    """
    Classifies feature points as background / obstacle and returns the
    Gaussian-smoothed gradient map g(x,y,t) = ∇(G∗O) (Eq. 4–6).

    Replaces snippet 3's TTC threshold heuristic with the paper's
    Otsu-on-angular-residual method.

    Returns
    -------
    obs_mask : (N,) bool   — True = obstacle candidate
    ttc      : (N,) float  — time-to-contact  (Eq. 3)
    gx, gy   : (H,W) float32 gradient maps
    """
    N = len(pts)

    # TTC 
    dist_foe = np.linalg.norm(pts - foe, axis=1) + 1e-6
    flow_mag = np.linalg.norm(flow,      axis=1) + 1e-6
    ttc      = dist_foe / flow_mag

    # Angular residual from expected radial-outward pattern
    radial   = pts - foe
    r_unit   = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-6)
    f_unit   = flow   / flow_mag[:, None]
    cos_sim  = np.clip((r_unit * f_unit).sum(axis=1), -1.0, 1.0)
    residual = 1.0 - cos_sim                            # 0 = bg, 2 = obstacle

    res_u8   = (np.clip(residual / 2.0, 0, 1) * 255).astype(np.uint8)
    obs_mask = np.zeros(N, bool)
    if len(np.unique(res_u8)) >= 2:
        thresh, _ = cv2.threshold(res_u8, 0, 255, cv2.THRESH_OTSU)
        obs_mask  = res_u8 > max(15, int(thresh * 0.7))

    # Binary obstacle plane O(x,y,t) → G∗O → ∇(G∗O)
    O = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for pt in pts[obs_mask]:
        xi = int(np.clip(pt[0], 0, WIDTH  - 1))
        yi = int(np.clip(pt[1], 0, HEIGHT - 1))
        O[yi, xi] = 1.0

    G_O = cv2.GaussianBlur(O, (0, 0),WIDTH/2)
    gx  = cv2.Sobel(G_O, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(G_O, cv2.CV_32F, 0, 1, ksize=3)

    return obs_mask, ttc, gx, gy


# Attractive force generated by projected goal position.
#
# According to Eq. 8:
# F_att = α (goal_pixel - FOE)
#
# The force pulls the vehicle toward the target location visible in image space.
def attractive_force(foe: np.ndarray, goal_pixel: np.ndarray) -> np.ndarray:
    
    return ALPHA * (goal_pixel - foe)

# Repulsive obstacle force from optical-flow obstacle gradient.
#
# Sparse approximation of Eq. 9:
# F_rep = γ / |R| * Σ g(xi, yi) / Σ TTC_i
#
# Interpretation:
# - stronger gradient = stronger obstacle presence
# - smaller TTC = more urgent avoidance
#
# Obstacles close to collision generate larger repulsion.
def repulsive_force(pts: np.ndarray, flow: np.ndarray,
                    obs_mask: np.ndarray, ttc: np.ndarray,
                    gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
   
    n_obs = int(obs_mask.sum())
    if n_obs == 0:
        return np.zeros(2, dtype=np.float32)

    R       = float(HEIGHT * WIDTH)
    obs_pts = pts[obs_mask]
    sum_ttc = float(ttc[obs_mask].sum()) + 1e-6
    sum_gx  = 0.0
    sum_gy  = 0.0

    for pt in obs_pts:
        xi = int(np.clip(pt[0], 0, WIDTH  - 1))
        yi = int(np.clip(pt[1], 0, HEIGHT - 1))
        sum_gx += float(gx[yi, xi])
        sum_gy += float(gy[yi, xi])

    return np.array([
        GAMMA * sum_gx / (R * sum_ttc),
        GAMMA * sum_gy / (R * sum_ttc),
    ], dtype=np.float32)

# Road boundary force using Morse potential.
#
# Lane boundaries are treated as virtual repulsive walls:
# - approaching +Y wall pushes car right
# - approaching -Y wall pushes car left
#
# A small center pull term keeps the car near lane center.
#
# This replaces image-space road edges with world-space lane geometry.
def road_force(world_y: float) -> float:
   
    d_r = max( ROAD_HALF_WIDTH - world_y, 1e-3)   # dist to +y (left)  wall
    d_l = max( ROAD_HALF_WIDTH + world_y, 1e-3)   # dist to -y (right) wall

    def _morse_grad(d: float) -> float:
        e = np.exp(-B_MORSE * d)
        return 2.0 * A_MORSE * B_MORSE * e * (1.0 - e)

    # dU_sr/dy = -morse_grad(d_r)  [chain rule: d(d_r)/dy = -1]
    # dU_sl/dy = +morse_grad(d_l)  [chain rule: d(d_l)/dy = +1]
    dUsr_dy = -_morse_grad(d_r)
    dUsl_dy = +_morse_grad(d_l)
    # Add a gentle linear pull toward the center (y=0)
    # This prevents the car from getting "stuck" offset from the center
    K_center = 0.1 
    center_pull = -K_center * world_y

    return float(dUsr_dy + dUsl_dy+center_pull)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary condition enforcement  (3-zone hard safety layer)
# Hard safety override layer
#
# Three safety zones:
#
# Zone 1 WARN:
#   Gradually bias steering away from nearest wall
#
# Zone 2 CRITICAL:
#   Strong steering override + speed reduction
#
# Zone 3 EMERGENCY:
#   Maximum steering escape + full brake
#
# Applied after visual potential field to guarantee wall avoidance.
# ─────────────────────────────────────────────────────────────────────────────

def boundary_conditions(world_y: float,
                        steer_in: float,
                        speed_in: float):
   
    d_r = ROAD_HALF_WIDTH - world_y   # distance to +y (left)  wall
    d_l = ROAD_HALF_WIDTH + world_y   # distance to -y (right) wall

    if d_r <= d_l:
        d_min, escape = d_r, -1.0   # near +y wall → steer RIGHT (negative)
    else:
        d_min, escape = d_l,  1.0   # near -y wall → steer LEFT  (positive)

    if d_min < BC_EMRG:
        # ── BUG FIX ── was: np.clip(-escape, …)  →  steered into wall ✗
        return float(np.clip(escape, -1.0, 1.0)), 0.0, 3

    if d_min < BC_CRIT:
        # ── BUG FIX ── was: np.clip(-escape, …)  →  steered into wall ✗
        return float(np.clip(escape, -1.0, 1.0)), 0.18, 2

    if d_min < BC_WARN:
        blend  = 1.0 - (d_min - BC_CRIT) / (BC_WARN - BC_CRIT)
        # ── BUG FIX ── was: steer_in*(1-blend) + (-escape)*blend ✗
        biased = steer_in * (1.0 - blend) + escape * blend
        return float(np.clip(biased, -1.0, 1.0)), min(speed_in, 0.30), 1

    return steer_in, speed_in, 0


# ─────────────────────────────────────────────────────────────────────────────
# GTSMC  (snippet 3 structure + fixed formulation + saturation)
# Gradient Tracking Sliding Mode Controller (GTSMC)
#
# Steering objective:
# force vehicle heading to follow desired potential-field direction.
#
# Sliding manifold:
# s = c_r * heading_error + heading_rate_error
#
# Saturation function smooths steering to avoid oscillation.
#
# This converts visual force into steering command.
# ─────────────────────────────────────────────────────────────────────────────

class GTSMController:
    

    def __init__(self):
        self._prev_steer = 0.0
        self._prev_psi_e = 0.0
        self._speed      = 0.5

    def compute_steer(self, F_total: np.ndarray,
                      current_yaw: float, dt: float) -> float:
        
        # Desired heading (Eq. 27)  — FIX: correct (y, x) order
        psi_d = float(np.arctan2(F_total[1], F_total[0]))
        psi_d = (psi_d + np.pi) % (2.0 * np.pi) - np.pi

        psi_e     = current_yaw - psi_d
        psi_e     = (psi_e + np.pi) % (2.0 * np.pi) - np.pi
        psi_e_dot = (psi_e - self._prev_psi_e) / max(dt, 1e-6)

        
        sr  = CR * psi_e + psi_e_dot
        sat = float(np.clip(sr / EPSILON, -1.0, 1.0))
        u   = -U0 * sat

        # Integrate and smooth
        raw   = float(np.clip(self._prev_steer + u * dt, -1.0, 1.0))
        steer = ALPHA_F * raw + (1.0 - ALPHA_F) * self._prev_steer

        self._prev_steer = steer
        self._prev_psi_e = psi_e
        return steer

    def compute_speed(self, n_obs: int, steer: float,
                      world_y: float, dt: float) -> float:
        """Adaptive speed reduction near obstacles, walls, and in turns."""
        d_near    = min(ROAD_HALF_WIDTH - abs(world_y), ROAD_HALF_WIDTH)
        wall_prox = max(0.0, 1.0 - d_near / 0.5)
        v_des     = float(np.clip(
            0.75 - n_obs * 0.05 - abs(steer) * 0.15 - wall_prox * 0.20,
            0.20, 0.75))
        sl    = self._speed - v_des
        a     = -0.08 * float(np.sign(sl)) if abs(sl) > 1e-3 else 0.0
        speed = float(np.clip(self._speed + a * dt, 0.20, 0.55))
        self._speed = speed
        return speed


# ─────────────────────────────────────────────────────────────────────────────
# Overlay 
# ─────────────────────────────────────────────────────────────────────────────

BC_ZONE_COLOR = {1: (0, 165, 255), 2: (0, 0, 220), 3: (200, 0, 200)}
BC_ZONE_LABEL = {0: "SAFE", 1: "WARN", 2: "CRIT", 3: "EMRG"}
BC_ZONE_TXT   = {0: (40, 200, 40), 1: (0, 165, 255),
                 2: (0, 0, 220), 3: (200, 0, 200)}


def draw_overlay(frame_gray: np.ndarray,
                 good_prev: np.ndarray, flows: np.ndarray,
                 obs_mask: np.ndarray, foe: np.ndarray,
                 goal_px: np.ndarray,
                 steer: float, speed: float,
                 n_obs: int, bc_zone: int,
                 world_y: float,
                 F_att: np.ndarray,
                 F_rep: np.ndarray) -> np.ndarray:
    """Annotated BGR frame: flow vectors, FOE, goal, BC zone, HUD."""
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    H, W = vis.shape[:2]

    # ── Boundary-zone edge band ───────────────────────────────────────────
    if bc_zone in BC_ZONE_COLOR:
        col   = BC_ZONE_COLOR[bc_zone]
        bw    = max(8, int(W * 0.06))
        alpha = 0.55
        if world_y > 0:                          # near left wall
            band = vis[:, :bw].copy()
            cv2.rectangle(vis, (0, 0), (bw, H), col, -1)
            cv2.addWeighted(band, 1.0 - alpha,
                            vis[:, :bw], alpha, 0, vis[:, :bw])
        else:                                    # near right wall
            band = vis[:, W - bw:].copy()
            cv2.rectangle(vis, (W - bw, 0), (W, H), col, -1)
            cv2.addWeighted(band, 1.0 - alpha,
                            vis[:, W - bw:], alpha, 0, vis[:, W - bw:])
        cv2.putText(vis, BC_ZONE_LABEL[bc_zone],
                    (W // 2 - 22, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    col, 2, cv2.LINE_AA)

    # ── Optical flow vectors ──────────────────────────────────────────────
    for pt, fl, is_obs in zip(good_prev, flows, obs_mask):
        x,  y  = int(pt[0]),  int(pt[1])
        x2, y2 = (int(np.clip(pt[0] + fl[0] * 4, 0, W - 1)),
                  int(np.clip(pt[1] + fl[1] * 4, 0, H - 1)))
        col = (40, 40, 220) if is_obs else (40, 200, 40)
        cv2.arrowedLine(vis, (x, y), (x2, y2), col, 1, tipLength=0.35)
        cv2.circle(vis, (x, y), 2, col, -1)

    # ── FOE crosshair ─────────────────────────────────────────────────────
    fx = int(np.clip(foe[0], 0, W - 1))
    fy = int(np.clip(foe[1], 0, H - 1))
    cv2.circle(vis, (fx, fy), 12, (0, 220, 220), 2, cv2.LINE_AA)
    cv2.line(vis, (fx - 16, fy), (fx + 16, fy), (0, 220, 220), 1)
    cv2.line(vis, (fx, fy - 16), (fx, fy + 16), (0, 220, 220), 1)
    cv2.putText(vis, "FOE", (fx + 16, fy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 220), 1)

    # ── World-anchored goal reticle ────────────────────────────────────────
    gx = int(np.clip(goal_px[0], 0, W - 1))
    gy = int(np.clip(goal_px[1], 0, H - 1))
    cv2.circle(vis, (gx, gy), 16, (0, 200, 80), 2, cv2.LINE_AA)
    cv2.line(vis, (gx - 10, gy), (gx + 10, gy), (0, 200, 80), 1)
    cv2.line(vis, (gx, gy - 10), (gx, gy + 10), (0, 200, 80), 1)
    cv2.putText(vis, "GOAL", (gx + 18, gy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 80), 1)

    # ── Lateral position bar ──────────────────────────────────────────────
    lat   = world_y / ROAD_HALF_WIDTH
    bw_b, bh_b = 100, 10
    bx_b, by_b = W - bw_b - 8, 76
    cv2.rectangle(vis, (bx_b, by_b),
                  (bx_b + bw_b, by_b + bh_b), (60, 60, 60), -1)
    mid   = bx_b + bw_b // 2
    dot_x = int(np.clip(mid - lat * bw_b // 2, bx_b, bx_b + bw_b))
    dot_c = (40, 200, 40) if abs(lat) < 0.6 else (0, 80, 220)
    cv2.circle(vis, (dot_x, by_b + bh_b // 2), 6, dot_c, -1)
    cv2.line(vis, (mid, by_b), (mid, by_b + bh_b), (120, 120, 120), 1)
    cv2.putText(vis, "y-pos", (bx_b, by_b - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

    # ── Steering bar ──────────────────────────────────────────────────────
    sb_w  = int(abs(steer) * (bw_b // 2))
    sb_cx = mid
    sb_y  = by_b + 20
    cv2.rectangle(vis, (bx_b, sb_y),
                  (bx_b + bw_b, sb_y + bh_b), (60, 60, 60), -1)
    cv2.line(vis, (sb_cx, sb_y), (sb_cx, sb_y + bh_b), (120, 120, 120), 1)
    if steer > 0:
        cv2.rectangle(vis, (sb_cx, sb_y),
                      (sb_cx + sb_w, sb_y + bh_b), (200, 80, 40), -1)
    else:
        cv2.rectangle(vis, (sb_cx - sb_w, sb_y),
                      (sb_cx, sb_y + bh_b), (200, 80, 40), -1)
    cv2.putText(vis, "steer", (bx_b, sb_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

    # ── HUD bar ───────────────────────────────────────────────────────────
    lat_col = BC_ZONE_TXT.get(bc_zone, (40, 200, 40))
    cv2.rectangle(vis, (0, 0), (W, 66), (18, 18, 18), -1)
    cv2.putText(vis,
                f"steer={steer:+.2f}  spd={speed:.2f}  obs={n_obs}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(vis,
                f"lat={lat:+.2f}  y={world_y:+.3f}m",
                (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                lat_col, 1, cv2.LINE_AA)
    cv2.putText(vis,
                f"BC={BC_ZONE_LABEL[bc_zone]}  "
                f"grn=ego  red=obs  cyan=FOE  grn-ring=goal",
                (6, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(vis,
            f"Fatt=({F_att[0]:+.2f},{F_att[1]:+.2f})",
            (220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(vis,
                f"Frep=({F_rep[0]:+.2f},{F_rep[1]:+.2f})",
                (220, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 255), 1, cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# # Main control loop:
#
# 1. Capture current frame
# 2. Compute sparse optical flow
# 3. Estimate FOE
# 4. Detect obstacle points
# 5. Compute attractive + repulsive + road forces
# 6. Convert force to desired steering angle
# 7. Apply safety boundary conditions
# 8. Send steering and velocity commands to PyBullet
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(max_steps: int = 5000, visualise: bool = True):
    DT = 1.0 / 60.0

    car_id, steer_joints, motor_joints = setup_simulation(
        dt=DT, settle_frames=60, gui=True)

    cam  = CarCamera(car_id)
    ctrl = GTSMController()

    # ── Feature detection mask ────────────────────────────────────────────
    # Block: bottom 45% (car body), top 10% (sky), sides 10%
    feat_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    y_top = int(HEIGHT * 0.10)
    y_cut = int(HEIGHT * 0.55)
    x_lo  = int(WIDTH  * 0.10)
    x_hi  = int(WIDTH  * 0.90)
    feat_mask[y_top:y_cut, x_lo:x_hi] = 255

    feat_params = dict(maxCorners=FEAT_MAX, qualityLevel=FEAT_QUALITY,
                       minDistance=FEAT_DIST, blockSize=7)

    # ── Initial frame ─────────────────────────────────────────────────────
    prev_gray = cam.get_frame()
    prev_pts  = cv2.goodFeaturesToTrack(prev_gray, mask=feat_mask,
                                        **feat_params)

    # ── Display window ────────────────────────────────────────────────────
    if visualise:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE,
                         WIDTH  * DISPLAY_SCALE,
                         HEIGHT * DISPLAY_SCALE)

    foe     = np.array([WIDTH / 2.0, HEIGHT * 0.35])
    steer   = 0.0
    speed   = 0.5
    bc_zone = 0

    n_obs    = 0
    obs_mask = np.zeros(0, dtype=bool)
    F_att = np.zeros(2)
    F_rep = np.zeros(2)
    print("[Agent] Controller active. Press 'q' to abort.")

    try:
        for step in range(max_steps):
            p.stepSimulation()
            # time.sleep(DT)

            curr_gray = cam.get_frame()

            # ── Goal check ────────────────────────────────────────────────
            car_pos, car_orn = p.getBasePositionAndOrientation(car_id)
            R = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(3, 3)

            offset = np.array([0.45, 0.0, 0.0])

            car_pos = np.array(car_pos) + R @ offset
            if car_pos[0] >= GOAL_X:
                print(f"\n[SUCCESS] Goal reached at step {step}! "
                      f"pos=({car_pos[0]:.1f}, {car_pos[1]:+.2f}) m")
                for j in motor_joints:
                    p.setJointMotorControl2(car_id, j,
                                            p.VELOCITY_CONTROL,
                                            targetVelocity=0.0, force=800)
                break

            current_yaw = float(p.getEulerFromQuaternion(car_orn)[2])
            world_y     = float(car_pos[1])
            goal_px     = cam.project_world_point(GOAL_WORLD)

            # ── Lucas-Kanade optical flow ─────────────────────────────────
            if prev_pts is not None and len(prev_pts) >= MIN_FEATURES:
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, prev_pts, None,
                    winSize=LK_WIN, maxLevel=LK_LEVELS,
                    criteria=LK_CRITERIA)

                ok        = (status.ravel() == 1)
                good_prev = prev_pts.reshape(-1, 2)[ok]
                good_curr = curr_pts.reshape(-1, 2)[ok]
                flows     = good_curr - good_prev

                if ok.sum() >= MIN_FEATURES:
                    # ── FOE ───────────────────────────────────────────────
                    foe_new = solve_foe(good_prev, flows)
                    if foe_new is not None:
                        foe = 0.7 * foe + 0.3 * foe_new

                    # ── Obstacle detection ────────────────────────────────
                    obs_mask, ttc, gx, gy = detect_obstacles(
                        good_prev, flows, foe)

                    # ── Potential forces ──────────────────────────────────
                    F_att    = attractive_force(foe, goal_px)
                    F_rep    = repulsive_force(good_prev, flows,
                                               obs_mask, ttc, gx, gy)
                    F_road_Y = road_force(world_y)

                    # Total force: image-plane lateral + forward component
                    # image-x → world −Y  (flip sign for lateral)
                    F_lat   = -(F_att[0] - F_rep[0]) + 1.2 * F_road_Y
                    F_total = np.array([1.0, F_lat])   # [forward, lateral]

                    # ── GTSMC ─────────────────────────────────────────────
                    n_obs = int(obs_mask.sum())
                    steer = ctrl.compute_steer(F_total, current_yaw, DT)
                    speed = ctrl.compute_speed(n_obs, steer, world_y, DT)

                    # ── Hard boundary conditions ───────────────────────────
                    steer, speed, bc_zone = boundary_conditions(
                        world_y, steer, speed)

                prev_pts = curr_pts.reshape(-1, 1, 2)[ok]

                # ── Visualise ─────────────────────────────────────────────
                if visualise:
                    obs_draw = obs_mask if ok.sum() >= MIN_FEATURES \
                               else np.zeros(ok.sum(), dtype=bool)
                    overlay  = draw_overlay(
                        curr_gray, good_prev, flows,
                        obs_draw, foe, goal_px,
                        steer, speed, n_obs, bc_zone, world_y,F_att, F_rep)
                    disp = cv2.resize(
                        overlay,
                        (WIDTH * DISPLAY_SCALE, HEIGHT * DISPLAY_SCALE),
                        interpolation=cv2.INTER_LINEAR)
                    cv2.imshow(WINDOW_TITLE, disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Aborted by user.")
                        break

            # ── Redetect features ─────────────────────────────────────────
            if (step % REDETECT_EVERY == 0
                    or prev_pts is None
                    or len(prev_pts) < MIN_FEATURES):
                prev_pts = cv2.goodFeaturesToTrack(
                    curr_gray, mask=feat_mask, **feat_params)

            # ── Apply control ─────────────────────────────────────────────
            angle = float(np.clip(steer, -1.0, 1.0)) * 0.698  # ±40°
            vel   = speed * TARGET_VEL
            for j in steer_joints:
                p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL,
                                        targetPosition=angle, force=10.0)
            for j in motor_joints:
                p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL,
                                        targetVelocity=vel, force=800.0)

            prev_gray = curr_gray.copy()

            if step % 60 == 0:
                print(f"  step={step:4d}  x={car_pos[0]:.1f}m  "
                      f"y={world_y:+.2f}m  ψ={current_yaw:+.2f}  "
                      f"steer={steer:+.2f}  spd={speed:.2f}  "
                      f"obs={n_obs}  "           # BUG FIX: was 'n_obs if "n_obs" in dir()'
                      f"BC={BC_ZONE_LABEL[bc_zone]}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()
        try:
            p.disconnect()
        except Exception:
            pass
        print("Simulation ended.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VPF Navigation — Capito et al. (2020), snippet-3 base")
    parser.add_argument("--steps",  type=int, default=5000)
    parser.add_argument("--no-vis", action="store_true",
                        help="Disable OpenCV overlay window")
    args = parser.parse_args()
    run_agent(max_steps=args.steps, visualise=not args.no_vis)
