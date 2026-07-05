"""
dynamics.py  —  Task 3 & 4
Gravity compensation and full dynamics (M, C, G) for the Unitree Z1.
"""

import numpy as np
import mujoco

from src.jacobian import jacobian

LINK_NAMES = [f"link0{i}" for i in range(1, 7)]


def _get_link_body_ids(model: mujoco.MjModel) -> list[int]:
    ids = []
    for name in LINK_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            ids.append(body_id)
    return ids


def gravity_torques(q: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    g        = np.array(model.opt.gravity)
    link_ids = _get_link_body_ids(model)
    tau_g    = np.zeros(len(q))

    for link_idx, body_id in enumerate(link_ids):
        m_k        = model.body_mass[body_id]
        p_com_body = model.body_ipos[body_id].copy()
        J          = jacobian(link_idx + 1, p_com_body, q, model)
        tau_g     += J[:3, :].T @ (m_k * g)

    return -tau_g   # negate to match MuJoCo's qfrc_bias convention


def apply_gravity_compensation(model, data):
    q = data.qpos[:model.nu].copy()
    data.ctrl[:model.nu] = gravity_torques(q, model)  # no minus, tau_g already negated


def validate_against_mujoco(q: np.ndarray,
                             model: mujoco.MjModel,
                             data: mujoco.MjData) -> None:
    """
    Compare applied_tau gravity_torques against MuJoCo's qfrc_bias at rest.
    At zero velocity qfrc_bias equals the gravity generalised force exactly.
    """
    data.qpos[:model.nu] = q
    data.qvel[:model.nu] = 0.0
    mujoco.mj_forward(model, data)
    mujoco_g = data.qfrc_bias[:model.nu].copy()

    applied_tau_g = gravity_torques(q, model)

    print(f"{'Joint':<8} {'MuJoCo qfrc_bias':>18} {'applied_tau tau_g':>18} {'Error':>12}")
    print("-" * 60)
    for i in range(model.nu):
        err = applied_tau_g[i] - mujoco_g[i]
        print(f"  q{i+1}    {mujoco_g[i]:>18.6f} {applied_tau_g[i]:>18.6f} {err:>12.2e}")
        
def generalised_mass_matrix(q: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """
    Compute the generalized mass matrix M(q) using the COM Jacobians.
    """
    n = len(q)
    M = np.zeros((n, n))
    
    # 1. Create a temporary data object to avoid altering the main simulation state
    #    during the finite difference perturbations.
    data_temp = mujoco.MjData(model)
    data_temp.qpos[:n] = q
    
    # 2. Update the spatial kinematics to calculate the correct rotation matrices 
    #    for this specific configuration 'q'.
    mujoco.mj_kinematics(model, data_temp)
    
    link_ids = _get_link_body_ids(model)
    
    for link_idx, body_id in enumerate(link_ids):
        m_i = model.body_mass[body_id]
        
        # Local COM position and Inertia
        p_com_local = model.body_ipos[body_id]
        I_local_diag = model.body_inertia[body_id]
        I_local = np.diag(I_local_diag)
        
        # 3. Retrieve the correctly updated rotation matrix for the link
        R_i = data_temp.xmat[body_id].reshape(3, 3) 
        
        # 4. Safely handle the Jacobian output (whether it's a tuple or a single 6xn array)
        J_result = jacobian(link_idx + 1, p_com_local, q, model)
        if isinstance(J_result, tuple):
            J_v, J_w = J_result
        else:
            J_v = J_result[:3, :]
            J_w = J_result[3:, :]
        
        # Rotate local inertia into global frame
        I_global = R_i @ I_local @ R_i.T
        
        # Add contribution to the total mass matrix
        M += m_i * (J_v.T @ J_v) + (J_w.T @ I_global @ J_w)
        
    return M


def _partial_derivative_M(q: np.ndarray, k: int, model: mujoco.MjModel, eps=1e-5) -> np.ndarray:
    """
    Helper function to compute dM/dq_k using central finite differences.
    """
    q_plus = q.copy()
    q_minus = q.copy()
    
    q_plus[k] += eps
    q_minus[k] -= eps
    
    M_plus = generalised_mass_matrix(q_plus, model)
    M_minus = generalised_mass_matrix(q_minus, model)
    
    return (M_plus - M_minus) / (2 * eps)

def coriolis(q: np.ndarray, q_dot: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """
    Compute the Coriolis matrix C(q, q_dot) using Christoffel symbols.
    """
    n = len(q)
    C = np.zeros((n, n))
    
    # Precompute the partial derivatives of M w.r.t all joint angles
    dM_dq = np.zeros((n, n, n))
    for k in range(n):
        dM_dq[k] = _partial_derivative_M(q, k, model)
        
    # Calculate Christoffel symbols and populate C matrix
    for i in range(n):
        for j in range(n):
            c_ij = 0
            for k in range(n):
                # Christoffel symbol c_ijk
                c_ijk = 0.5 * (dM_dq[k][i, j] + dM_dq[j][i, k] - dM_dq[i][j, k])
                c_ij += c_ijk * q_dot[k]
            C[i, j] = c_ij
            
    return C