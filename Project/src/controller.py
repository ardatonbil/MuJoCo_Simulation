import numpy as np
import mujoco

from src.dynamics import gravity_torques, generalised_mass_matrix, coriolis

GAINS = {
    "fast_reach": {
        # Drops fast, wiggles a bit. Wins First Reach, loses Mean Error due to the wiggle.
        "kp": [8000.0, 8000.0, 8000.0, 8000.0, 12000.0, 12000.0],
        "kd": [200.0,  200.0,  200.0,  200.0,  300.0,   300.0],
        "ki": [0.0,    0.0,    0.0,    0.0,    0.0,     0.0],
    },
    "lowest_error": {
        # Drops nearly as fast, but just enough damping (350) to smooth the wiggle. 
        # Tiny integral locks to zero. Wins Mean Error.
        "kp": [8000.0, 8000.0, 8000.0, 8000.0, 12000.0, 12000.0],
        "kd": [350.0,  350.0,  350.0,  350.0,  525.0,   525.0],
        "ki": [100.0,  100.0,  100.0,  100.0,  150.0,   150.0],
    },
}

def computed_torque_control(q, q_dot, q_des, q_dot_des, q_ddot_des, e_int, model, gain_set="fast_reach"):
    """
    Compute the control torques using PID computed torque control with raw simulator velocity.
    u = q_ddot_des + Kd*(q_dot_des - q_dot) + Kp*(q_des - q) + Ki * e_int
    """
    # 1. Compute dynamic matrices using the perfect analytical velocity
    M = generalised_mass_matrix(q, model)
    C = coriolis(q, q_dot, model)
    G_bias = gravity_torques(q, model)

    Kp = np.array(GAINS[gain_set]["kp"])
    Kd = np.array(GAINS[gain_set]["kd"])
    Ki = np.array(GAINS[gain_set]["ki"])

    error = q_des - q

    # 2. Virtual PID control law (u) using raw velocity for the D-term
    pid_term = q_ddot_des + Kd * (q_dot_des - q_dot) + Kp * error + Ki * e_int

    # 3. Total feedback linearizing torque command
    # Note: Subtracting G_bias pushes UP against gravity to compensate it perfectly
    tau = M @ pid_term + C @ q_dot - G_bias

    return tau

def impedance_control(q, q_dot, jacp, x_actual, x_target, model):
    """
    Computes joint torques for Cartesian Impedance Control.
    tau = J^T * (Kp * e_x - Kd * x_dot) - Kd_joint * q_dot + C * q_dot - G_bias
    """
    # 1. Compute dynamic matrices
    C = coriolis(q, q_dot, model)
    G_bias = gravity_torques(q, model)

    # 2. Impedance Gains (Virtual Spring and Damper)
    Kp_cart = np.diag([1000.0, 1000.0, 1000.0])  # N/m
    Kd_cart = np.diag([50.0, 50.0, 50.0])        # N.s/m
    
    # Joint-space damping to stabilize the redundant degrees of freedom (null-space)
    Kd_joint = np.diag([5.0, 5.0, 5.0, 2.0, 2.0, 2.0])

    # 3. Cartesian Error and Velocity
    error_x = x_target - x_actual
    x_dot = jacp @ q_dot

    # 4. Virtual Cartesian Force Law
    F_virtual = Kp_cart @ error_x - Kd_cart @ x_dot

    # 5. Map Cartesian Force to Joint Torques via Jacobian Transpose
    tau_impedance = jacp.T @ F_virtual
    tau_damping = -Kd_joint @ q_dot

    # 6. Total Feedback Linearizing Torque
    tau = tau_impedance + tau_damping + (C @ q_dot) - G_bias

    return tau