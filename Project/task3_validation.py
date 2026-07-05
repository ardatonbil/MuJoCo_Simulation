"""
task3_validation.py  —  Task 3 Validation
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt
import os

from src.dynamics import apply_gravity_compensation, validate_against_mujoco, gravity_torques

XML_PATH = os.path.join("z1_files", "z1_joint_torque_mode", "scene_torque.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

n_joints     = model.nu
dt           = model.opt.timestep
jnt_range    = model.jnt_range[:n_joints]
ctrl_min     = model.actuator_ctrlrange[:n_joints, 0]
ctrl_max     = model.actuator_ctrlrange[:n_joints, 1]

# ─────────────────────────────────────────────────────────────────────────────
# Generate 20 configs where gravity torque stays within actuator limits
# ─────────────────────────────────────────────────────────────────────────────
rng     = np.random.default_rng(seed=42)
configs = []
attempts = 0
while len(configs) < 20:
    attempts += 1
    q = np.array([rng.uniform(jnt_range[i, 0], jnt_range[i, 1]) for i in range(n_joints)])
    tau = gravity_torques(q, model) 
    
    # Check if torque is within limits
    if np.all(tau >= ctrl_min) and np.all(tau <= ctrl_max):
        data.qpos[:n_joints] = q
        mujoco.mj_forward(model, data)
        
        # If there are 0 contacts, the configuration is safe!
        if data.ncon == 0:
            configs.append(q)

configs = np.array(configs)
print(f"Found 20 valid configs in {attempts} attempts\n")

# ─────────────────────────────────────────────────────────────────────────────
# Torque validation vs qfrc_bias
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Torque validation vs MuJoCo qfrc_bias (zero velocity)")
print("=" * 60)
for i in range(3):
    print(f"\n--- Config {i+1} ---")
    validate_against_mujoco(configs[i], model, data)

# ─────────────────────────────────────────────────────────────────────────────
# Drift simulation
# ─────────────────────────────────────────────────────────────────────────────
SIM_SECONDS  = 3.0
RECORD_HZ    = 100
steps_sim    = int(SIM_SECONDS / dt)
record_every = max(1, int(1.0 / (RECORD_HZ * dt)))

all_times  = []
all_drifts = []

print(f"\nSimulating 20 configurations x {SIM_SECONDS} s each ...")

for cfg_idx, q0 in enumerate(configs):
    mujoco.mj_resetData(model, data)
    data.qpos[:n_joints] = q0
    data.qvel[:n_joints] = 0.0
    mujoco.mj_forward(model, data)

    times  = []
    drifts = []

    for step in range(steps_sim):
        apply_gravity_compensation(model, data)

        if step % record_every == 0:
            times.append(data.time)
            drifts.append(data.qpos[:n_joints].copy() - q0)

        mujoco.mj_step(model, data)

    all_times.append(np.array(times))
    all_drifts.append(np.array(drifts))

    drift_norm = np.linalg.norm(all_drifts[-1], axis=1)
    max_drift  = np.max(drift_norm)
    print(f"  Config {cfg_idx+1:2d}: max drift norm = {max_drift:.2e} rad")

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
axes_flat  = axes.flatten()
cmap       = plt.cm.tab20
colors     = [cmap(i / 20) for i in range(20)]

for j in range(n_joints):
    ax = axes_flat[j]
    for cfg_idx in range(20):
        ax.plot(all_times[cfg_idx], all_drifts[cfg_idx][:, j],
                color=colors[cfg_idx], linewidth=0.8, alpha=0.7)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_title(f"Joint {j+1}", fontsize=10)
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel("Drift [rad]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))

fig.suptitle(
    "Task 3 - Gravity Compensation Validation\n"
    "Joint angle drift from q0  (20 configs within actuator limits)",
    fontsize=12,
)

handles = [plt.Line2D([0], [0], color=colors[i], linewidth=1.5, label=f"Config {i+1}")
           for i in range(20)]
fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=6,
           title="Configuration", bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/task3_dynamics_validation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to plots/task3_dynamics_validation.png")

# ─────────────────────────────────────────────────────────────────────────────
# Additional plot — drift norm over time for all 20 configs on one axis
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
for cfg_idx in range(20):
    drift_norm = np.linalg.norm(all_drifts[cfg_idx], axis=1)
    ax2.plot(all_times[cfg_idx], drift_norm,
             color=colors[cfg_idx], linewidth=0.8, alpha=0.8, label=f"Config {cfg_idx+1}")
ax2.axhline(0, color="k", linewidth=0.5, linestyle="--")
ax2.set_xlabel("Time [s]", fontsize=10)
ax2.set_ylabel("||q(t) - q0|| [rad]", fontsize=10)
ax2.set_title("Task 3 - Configuration Space Drift Norm (20 configs)", fontsize=12)
ax2.legend(ncol=5, fontsize=6, loc="upper right")
plt.tight_layout()
plt.savefig("plots/task3_dynamics_validation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved to plots/task3_drift_validation.png")