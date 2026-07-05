import numpy as np
import mujoco
import matplotlib.pyplot as plt
import os

from src.controller import computed_torque_control

XML_PATH = os.path.join("z1_files", "z1_joint_torque_mode", "scene_torque.xml")

SETTLE_THRESHOLD = 0.05   # rad


def first_reach_time(times, error_norms, threshold=SETTLE_THRESHOLD):
    times  = np.array(times)
    errors = np.array(error_norms)
    idx = np.argmax(errors < threshold)
    if errors[idx] < threshold:
        return times[idx]
    return None


def generate_safe_configurations(model, data, num_cases=3):
    rng = np.random.default_rng(seed=104)
    n_joints = model.nu
    jnt_range = model.jnt_range[:n_joints]

    cases = []
    attempts = 0
    while len(cases) < num_cases:
        attempts += 1
        q_init = np.array([rng.uniform(jnt_range[i, 0], jnt_range[i, 1]) for i in range(n_joints)])
        q_targ = np.array([rng.uniform(jnt_range[i, 0], jnt_range[i, 1]) for i in range(n_joints)])

        data.qpos[:n_joints] = q_init
        mujoco.mj_forward(model, data)
        init_ok = (data.ncon == 0)

        data.qpos[:n_joints] = q_targ
        mujoco.mj_forward(model, data)
        targ_ok = (data.ncon == 0)

        if init_ok and targ_ok:
            cases.append({"q0": q_init, "qd": q_targ})

    print(f"Generated {num_cases} safe cases in {attempts} attempts.")
    return cases


def run_task5_validation():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    n_joints = model.nu
    dt = model.opt.timestep

    q_dot_des  = np.zeros(n_joints)
    q_ddot_des = np.zeros(n_joints)

    cases = generate_safe_configurations(model, data, num_cases=3)

    SIM_SECONDS = 5.0
    steps_sim   = int(SIM_SECONDS / dt)
    gain_sets   = ["fast_reach", "lowest_error"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    os.makedirs("plots", exist_ok=True)

    print("-" * 60)
    print("Running Task 5 Torque Control Simulations...")
    print("-" * 60)
    print(f"{'Case':<6} {'Tuning':<14} {'First Reach':>12} {'Mean Error':>12}")
    print("-" * 60)

    for c_idx, case in enumerate(cases):
        q0 = case["q0"]
        qd = case["qd"]

        for obj_idx, gain_name in enumerate(gain_sets):
            mujoco.mj_resetData(model, data)
            data.qpos[:n_joints] = q0
            data.qvel[:n_joints] = 0.0
            mujoco.mj_forward(model, data)

            times          = []
            error_norms    = []
            e_int          = np.zeros(n_joints)

            for step in range(steps_sim):
                q     = data.qpos[:n_joints].copy()
                q_dot = data.qvel[:n_joints].copy()

                error  = qd - q
                e_int += error * dt
                e_int  = np.clip(e_int, -1.5, 1.5)

                error_norms.append(np.linalg.norm(error))
                times.append(data.time)

                tau_cmd = computed_torque_control(
                    q=q,
                    q_dot=q_dot,
                    q_des=qd,
                    q_dot_des=q_dot_des,
                    q_ddot_des=q_ddot_des,
                    e_int=e_int,
                    model=model,
                    gain_set=gain_name
                )

                ctrl_min = model.actuator_ctrlrange[:n_joints, 0]
                ctrl_max = model.actuator_ctrlrange[:n_joints, 1]
                data.ctrl[:n_joints] = np.clip(tau_cmd, ctrl_min, ctrl_max)
                mujoco.mj_step(model, data)

            mean_error = np.mean(error_norms)
            t_metric   = first_reach_time(times, error_norms)
            reach_str  = f"{t_metric:.3f} s" if t_metric is not None else ">5 s"

            print(f"{c_idx+1:<6} {gain_name:<14} {reach_str:>12} {mean_error:>11.4f} rad")

            ax    = axes[c_idx, obj_idx]
            color = 'teal' if obj_idx == 0 else 'indigo'
            ax.plot(times, error_norms, color=color, linewidth=1.5)

            # always show both metrics on the plot
            if t_metric is not None:
                ax.axvline(t_metric, color='red', linewidth=1.0, linestyle='--',
                           label=f"First reach: {t_metric:.2f}s")
            ax.axhline(mean_error, color='orange', linewidth=1.2, linestyle='--',
                       label=f"Mean error: {mean_error:.4f} rad")
            ax.legend(fontsize=8, loc='upper right')

            ax.axhline(SETTLE_THRESHOLD, color='gray', linewidth=0.8,
                       linestyle=':', alpha=0.8)
            ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
            ax.grid(True, linestyle=":", alpha=0.7)

            if c_idx == 0:
                title = "Minimize Time" if obj_idx == 0 else "Minimize Mean Error"
                ax.set_title(f"Objective: {title}", fontsize=12, fontweight='bold')
            if obj_idx == 0:
                ax.set_ylabel(f"Case {c_idx+1}\n||q_d - q|| [rad]", fontsize=10)
            if c_idx == 2:
                ax.set_xlabel("Time [s]", fontsize=10)

    fig.suptitle("Task 5 - Computed Torque Position Control",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("plots/task5_validation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved to plots/task5_validation.png")


if __name__ == "__main__":
    run_task5_validation()