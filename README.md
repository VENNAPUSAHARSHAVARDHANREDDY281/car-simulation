# car-simulation

Main components:

* Monocular camera capture from PyBullet vehicle
* Sparse optical flow using Lucas-Kanade method
* Focus of Expansion (FOE) estimation for ego-motion analysis
* Obstacle detection using TTC and angular residuals
* Visual potential field generation:

  * attractive force toward target
  * repulsive force from obstacles
  * road boundary force using Morse potential
* Gradient-based steering control for obstacle avoidance and lane keeping

The vehicle navigates toward the goal while avoiding obstacles using only monocular visual motion cues.
