# RS LiDAR Odometry

LiDAR-based odometry using **ICP** to estimate motion between consecutive scans.

Built mainly to understand how scan matching actually works instead of just using existing SLAM packages.

---

## What This Is

* **Takes:** LiDAR scans
* **Uses:** ICP to align consecutive frames
* **Estimates:** How the robot moved

---

## Implementation

`Scan (t) + Scan (t-1) → ICP → (R, t) → Pose Update`

- Current scan is aligned with the previous scan using ICP.
- ICP outputs a rigid transform:
    - Rotation matrix **R**
    - Translation vector **t**

### Pose Update

The pose is updated incrementally using the estimated transform.

**If previous pose is:**
$R_{prev}, t_{prev}$

**And ICP gives:**
$R_{icp}, t_{icp}$

**Then:**
$$R_{new} = R_{icp} \cdot R_{prev}$$
$$t_{new} = R_{icp} \cdot t_{prev} + t_{icp}$$

- Rotation is updated by applying the matrix to the current pose matrix (product).
- Translation is rotated into the global frame and then updated.

Pose is accumulated over time by repeatedly applying these transforms.

---

## Author

**Akshat Sharma**
