# RRT-Connect — literature grounding for the feasibility oracle

Backs the scene builder's use of RRT-Connect as an independent, non-shield
feasibility oracle (`shortstop/calvin/feasibility_planner.py`). Written for direct reuse in the paper's
methodology / related-work section — citations and claims below are
sourced, not assumed.

## Original algorithm

**Kuffner, J. J., & LaValle, S. M. (2000). RRT-Connect: An efficient
approach to single-query path planning.** *Proceedings 2000 ICRA.
Millennium Conference. IEEE International Conference on Robotics and
Automation. Symposia Proceedings*, Vol. 2, pp. 995–1001.
DOI: 10.1109/ROBOT.2000.844730.

Directly relevant, not just adjacent: the original paper demonstrates the
algorithm on a **7-DOF kinematic chain modeling a human arm**, for
collision-free grasping/manipulation motion generation — the same DOF
count and problem class (redundant-arm reach-to-grasp under obstacles) as
this project's Panda arm setting, not a mobile-robot or point-mass case
being awkwardly repurposed.

## Adoption — this is the standard tool, not a niche pick

- RRT-Connect is **one of the default planners shipped in MoveIt**, the
  most widely used ROS manipulation planning framework, via **OMPL** (Open
  Motion Planning Library), the standard sampling-based planning library
  MoveIt integrates.
- Characterized in current sources as "commonly used and best known for
  its fast convergence, even in high-dimensional spaces" — the
  high-dimensional case is exactly a 7-DOF arm's joint space.
- **Probabilistically complete**: guaranteed to find a solution if one
  exists, given enough runtime — the property this project's feasibility
  gate actually needs (a pass/fail existence proof), not path optimality.
- **Non-optimizing by design**: "simply tries to find ANY path without
  taking path qualities such as length or collision clearance into
  account" — this is a feature for our use case, not a limitation: the
  oracle only needs to answer "does a path exist," never "what is the
  best path" (the path itself is discarded either way).
- Still an active research base 25 years on, not superseded/obsolete:
  e.g. **pRRTC** (GPU-parallel RRT-Connect, 2025) extends it for
  faster, more consistent motion planning, evidence the core algorithm
  remains the reference point new work builds on rather than replaces.

## One-line justification (paper-ready)

> Scene feasibility is verified offline with RRT-Connect
> [Kuffner & LaValle, 2000], the sampling-based planner underlying
> MoveIt/OMPL's default manipulation planning and originally demonstrated
> on a 7-DOF arm reach-to-grasp setting; its probabilistic completeness
> gives a pass/fail existence check independent of any shield baseline
> under comparison, and its non-optimizing nature is sufficient since the
> found path itself is discarded, never exposed to the policy or shield.

## Sources

- [RRT-connect: An efficient approach to single-query path planning (Semantic Scholar)](https://www.semanticscholar.org/paper/RRT-connect:-An-efficient-approach-to-single-query-Kuffner-LaValle/0f98f000da79132ef80a521ab8fad0396b82c7f5)
- [RRT-Connect (ResearchGate)](https://www.researchgate.net/publication/221075120_RRT-Connect_An_Efficient_Approach_to_Single-Query_Path_Planning)
- [ompl: The Open Motion Planning Library](https://ompl.kavrakilab.org/)
- [pRRTC: GPU-Parallel RRT-Connect for Fast, Consistent, and Low-Cost Motion Planning (2025)](https://arxiv.org/pdf/2503.06757)
