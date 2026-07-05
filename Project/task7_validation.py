import numpy as np
import mujoco
import matplotlib.pyplot as plt
import os

from src.controller import impedance_control

XML_PATH = os.path.join("z1_files", "z1_joint_torque_mode", "scene_torque.xml")

def run_task7_impedance_multi():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    n_joints = model.nu
    dt = model.opt.timestep

    # Get the ID for the end-effector site
    site_name = "attachment_site"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    # All simulations start from this curled-up posture
    q_initial = np.array([0.0, 0.785, -0.261, -0.523, 0.0, 0.0])

    target_poses = [
        {"name": "Overhead Arc",     "pos": np.array([-0.2, -0.3, 0.6])},
        {"name": "Forward Reach",    "pos": np.array([0.5, 0.0, 0.25])},
        {"name": "Tabletop Hover",   "pos": np.array([0.3, 0.3, 0.15])}
    ]

    SIM_SECONDS = 4.0
    steps_sim = int(SIM_SECONDS / dt)

    # Setup the massive 3-row, 2-column plot
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle("Task 7 - Impedance Control Across Multiple Arbitrary Poses", fontsize=16, fontweight="bold")

    print("-" * 60)
    print("Running Task 7: Multi-Pose Cartesian Impedance Control...")
    print("-" * 60)

    # Loop through each target pose
    for idx, target_info in enumerate(target_poses):
        x_target = target_info["pos"]
        case_name = target_info["name"]
        
        print(f"Simulating Case {idx+1}/3: [{case_name}] at XYZ = {x_target}")

        # 1. Hard reset the physics engine for the new simulation run
        mujoco.mj_resetData(model, data)
        data.qpos[:n_joints] = q_initial
        data.qvel[:n_joints] = 0.0
        mujoco.mj_forward(model, data)

        jacp = np.zeros((3, model.nv))
        times = []
        error_log = []
        path_log = []

        # 2. Run the 4-second simulation
        for step in range(steps_sim):
            t = data.time

            q = data.qpos[:n_joints].copy()
            q_dot = data.qvel[:n_joints].copy()
            x_actual = data.site_xpos[site_id].copy()

            mujoco.mj_jacSite(model, data, jacp, None, site_id)

            error_x = x_target - x_actual
            times.append(t)
            error_log.append(np.linalg.norm(error_x))
            path_log.append(x_actual)

            # Request Torque from your custom controller
            tau_cmd = impedance_control(
                q=q, 
                q_dot=q_dot, 
                jacp=jacp, 
                x_actual=x_actual, 
                x_target=x_target, 
                model=model
            )

            ctrl_min, ctrl_max = model.actuator_ctrlrange[:n_joints, 0], model.actuator_ctrlrange[:n_joints, 1]
            data.ctrl[:n_joints] = np.clip(tau_cmd, ctrl_min, ctrl_max)
            mujoco.mj_step(model, data)

        # 3. Plot this specific case onto the current row of the figure
        path_log = np.array(path_log)
        
        # 3D Path Plot (Left Column)
        ax1 = fig.add_subplot(3, 2, (2 * idx) + 1, projection='3d')
        ax1.plot(path_log[:, 0], path_log[:, 1], path_log[:, 2], 'b-', linewidth=2, label="EE Path")
        ax1.scatter(x_target[0], x_target[1], x_target[2], color='red', s=100, label="Target Point", marker='*')
        ax1.scatter(path_log[0, 0], path_log[0, 1], path_log[0, 2], color='green', s=50, label="Start Point")
        ax1.set_title(f"Case {idx+1}: {case_name} - 3D Path", fontweight="bold")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_zlabel("Z [m]")
        ax1.legend()

        # Error Plot (Right Column)
        ax2 = fig.add_subplot(3, 2, (2 * idx) + 2)
        ax2.plot(times, error_log, 'g-', linewidth=2)
        ax2.axhline(0.0, color='k', linestyle='--')
        ax2.set_title(f"Case {idx+1} Error: ||x_target - x_actual||", fontweight="bold")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Error [m]")
        ax2.grid(True, linestyle=":", alpha=0.7)

    # Save the final massive figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("plots", exist_ok=True)
    save_path = "plots/task7_impedance_multi.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nAll Simulations Complete! Mega-plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_task7_impedance_multi()