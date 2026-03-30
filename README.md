# Task 1 — Optical Flow Based Perception and Navigation

This submission contains implementations for:

* **Task 1 Subtask 1:** Sparse Lucas-Kanade Optical Flow
* **Task 1 Subtask 2:** Optical Flow Based Visual Potential Field Navigation

Both tasks are based on optical flow and follow the AGV software task requirements.

---

# Task 1 Subtask 1 — Sparse Optical Flow

## Objective

Implement the Lucas-Kanade sparse optical flow algorithm manually for feature tracking in video frames.

## Method

The implemented pipeline is:

1. Read input video frame
2. Convert frame to grayscale
3. Detect Shi-Tomasi corner features
4. Compute spatial gradients Ix and Iy
5. Build Lucas-Kanade matrix M
6. Solve local displacement iteratively
7. Update tracked feature positions
8. Draw motion trajectories

## Key Features

* Manual Lucas-Kanade implementation
* Shi-Tomasi corner detection
* Iterative convergence
* Dynamic corner refresh during scene changes

## Output

* Sparse motion vectors
* Feature trajectory visualization
* Active feature count display

---

# Task 1 Subtask 2 — Visual Potential Field Navigation

## Objective

Use sparse optical flow inside a PyBullet simulator to drive the vehicle toward the goal while avoiding obstacles.

---

## Method

Pipeline:

1. Capture grayscale image from onboard camera
2. Compute sparse optical flow between consecutive frames
3. Estimate Focus of Expansion (FOE)
4. Detect obstacle points using TTC and angular residual
5. Generate visual potential field
6. Compute steering command
7. Apply control in PyBullet

---

# Main Components

## 1. Sparse Optical Flow

Sparse Lucas-Kanade optical flow tracks feature motion across frames.

## 2. Focus of Expansion (FOE)

FOE estimates ego-motion center using:

FOE = (AᵀA)^(-1) Aᵀb

## 3. Obstacle Detection

Obstacle regions are identified using:

* Time-to-contact (TTC)
* Angular residual
* Otsu thresholding

## 4. Visual Potential Field

Three force components:

### Attractive Force

Goal-directed force.

### Repulsive Force

Obstacle avoidance force.

### Road Force

Lane boundary repulsion using Morse potential.

## 5. Controller

Total force is converted into steering angle.

---

# Files Included

* task1_subtask1_optical_flow.py
* task1_subtask2_visual_potential_field.py
* simulation_setup.py
* requirements.txt

---

# Environment

## Python Version

Python 3.10

## Dependencies

* numpy==1.24.4
* opencv-python==4.8.1.78
* pybullet==3.2.6

Install:

pip install -r requirements.txt

---

# Run Instructions

## Subtask 1

python task1_subtask1_optical_flow.py --video input.mp4

## Subtask 2

python task1_subtask2_visual_potential_field.py

---

# Output Summary

## Subtask 1

* Sparse feature tracking
* Optical flow trajectories

## Subtask 2

* Obstacle avoidance
* Lane following
* Goal reaching

---

# Limitations

* Sparse optical flow weak in low-texture scenes
* FOE unstable with few features
* Fixed road geometry assumption

---

# Future Improvements

* Dense optical flow
* Adaptive obstacle weighting
* Dynamic velocity control

---

# References

1. Lucas-Kanade Optical Flow
2. Shi-Tomasi Corner Detection
3. Optical Flow based Visual Potential Field for Autonomous Driving
