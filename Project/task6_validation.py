import numpy as np
import mujoco
import matplotlib.pyplot as plt
import os

from src.controller import computed_torque_control

XML_PATH = os.path.join("z1_files", "z1_joint_torque_mode", "scene_torque.xml")

def run_task6_3d_tracking():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    
    # We create a dummy data object purely to calculate the Forward Kinematics 
    # of the target trajectory without affecting the real physics simulation.
    data_target = mujoco.MjData(model)
    
    # Get the ID for the end-effector site defined in your XML
    site_name = "attachment_site"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    
    n_joints = model.nu
    dt = model.opt.timestep

    # Task 6 Parameters: 1 Minute Simulation
    SIM_SECONDS = 60.0
    steps_sim   = int(SIM_SECONDS / dt)
    
    A = np.array([0.5, 0.4, 0.4, 0.0, 0.0, 0.0])  # Amplitudes (4, 5, 6 are zeroed)
    w = np.array([0.3, 0.3, 0.3, 0.0, 0.0, 0.0])  # Frequencies
    q0_offset = np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0])
    
    # Logs for plotting
    times = []
    error_log = []
    ee_actual_log = []
    ee_target_log = []
    
    e_int = np.zeros(n_joints)
    
    print("-" * 60)
    print(f"Running Task 6: {SIM_SECONDS}s 3D Trajectory Tracking...")
    print("-" * 60)

    # Initialize positions to avoid the "startup snap"
    mujoco.mj_resetData(model, data)
    data.qpos[:n_joints] = q0_offset
    data.qvel[:n_joints] = 0.0
    mujoco.mj_forward(model, data)

    for step in range(steps_sim):
        t = data.time
        
        # 1. Analytical Trajectory Generation
        q_des      = q0_offset + A * np.sin(w * t)
        q_dot_des  = A * w * np.cos(w * t)
        q_ddot_des = -A * (w**2) * np.sin(w * t)

        # 2. Extract Actual State
        q     = data.qpos[:n_joints].copy()
        q_dot = data.qvel[:n_joints].copy()
        
        # 3. Log End-Effector Cartesian Positions
        # Actual position directly from the physics engine
        ee_actual = data.site_xpos[site_id].copy()
        
        # Target position via Forward Kinematics on the dummy data
        data_target.qpos[:n_joints] = q_des
        mujoco.mj_kinematics(model, data_target)
        ee_target = data_target.site_xpos[site_id].copy()

        # Integral accumulation
        error  = q_des - q
        e_int += error * dt
        e_int  = np.clip(e_int, -2.0, 2.0)

        # Log metrics
        times.append(t)
        error_log.append(np.linalg.norm(error))
        ee_actual_log.append(ee_actual)
        ee_target_log.append(ee_target)

        # 4. Control (Using the precision gains from Task 5)
        tau_cmd = computed_torque_control(
            q=q, q_dot=q_dot, q_des=q_des, 
            q_dot_des=q_dot_des, q_ddot_des=q_ddot_des, 
            e_int=e_int, model=model, gain_set="lowest_error"
        )

        # Step simulation
        ctrl_min, ctrl_max = model.actuator_ctrlrange[:n_joints, 0], model.actuator_ctrlrange[:n_joints, 1]
        data.ctrl[:n_joints] = np.clip(tau_cmd, ctrl_min, ctrl_max)
        mujoco.mj_step(model, data)

        if step % 2000 == 0:
            print(f"Simulating time {t:.1f}s / {SIM_SECONDS}s...")

    # --- PLOTTING ---
    ee_actual_log = np.array(ee_actual_log)
    ee_target_log = np.array(ee_target_log)
    
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle("Task 6 - 60s Dynamic Trajectory Tracking", fontsize=14, fontweight="bold")
    
    # Plot i) 3D EE Path
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(ee_target_log[:, 0], ee_target_log[:, 1], ee_target_log[:, 2], 
             'k--', label="Target Path", linewidth=2, alpha=0.7)
    ax1.plot(ee_actual_log[:, 0], ee_actual_log[:, 1], ee_actual_log[:, 2], 
             'b-', label="Actual Path", linewidth=1.5)
    ax1.set_title("i) End-Effector 3D Cartesian Path", fontweight="bold")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")
    ax1.legend()

    # Plot ii) Position Error vs Time
    ax2 = fig.add_subplot(122)
    ax2.plot(times, error_log, 'indigo', linewidth=1.5)
    ax2.set_title("ii) Configuration Error vs Time", fontweight="bold")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Norm of Joint Error ||q_d - q|| [rad]")
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/task6_1minute_tracking.png", dpi=150, bbox_inches="tight")
    print("\nSimulation Complete! Plot saved to plots/task6_1minute_tracking.png")
    plt.show()

if __name__ == "__main__":
    run_task6_3d_tracking()