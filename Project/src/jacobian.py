import numpy as np
from src.kinematics import forward_kinematics, make_transform, quaternion_to_rotation


def compute_all_joint_world_frames(q, model):
    """
    Returns joint origins and axes in world frame for all 6 joints,
    computed from the FK chain.

    For joint i living in body B_i:
      - origin  = position of B_i frame in world
      - axis    = R_0_Bi @ local_axis
    """
    # Body names in chain order
    body_names = ['link00', 'link01', 'link02', 'link03', 'link04', 'link05', 'link06']
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

    # Accumulate transforms link by link
    T_world = np.eye(4)
    body_world_transforms = {}

    for name in body_names:
        body_id = model.body(name).id
        p  = model.body_pos[body_id].copy()
        R  = quaternion_to_rotation(model.body_quat[body_id])
        T_fixed = make_transform(R, p)
        T_world = T_world @ T_fixed

        # Apply joint rotation if this body has one
        joint_idx = body_names.index(name) - 1  # link01 has joint1, etc.
        if 1 <= body_names.index(name) <= 6:
            jname    = joint_names[joint_idx]
            jid      = model.joint(jname).id
            axis_loc = model.jnt_axis[jid].copy()
            axis_loc /= np.linalg.norm(axis_loc)

            # Joint origin and axis in world frame (before joint rotation)
            origin_world = T_world[:3, 3].copy()
            axis_world   = T_world[:3, :3] @ axis_loc

            body_world_transforms[jname] = {
                'origin': origin_world,
                'axis':   axis_world,
            }

            # Now apply the joint rotation to continue the chain
            c, s  = np.cos(q[joint_idx]), np.sin(q[joint_idx])
            K     = np.array([[         0, -axis_loc[2],  axis_loc[1]],
                               [ axis_loc[2],          0, -axis_loc[0]],
                               [-axis_loc[1],  axis_loc[0],          0]])
            R_jnt = np.eye(3) + s * K + (1 - c) * (K @ K)
            T_world = T_world @ make_transform(R_jnt, np.zeros(3))

    return body_world_transforms


def jacobian(link_index, p_body, q, model):
    """
    Compute the 6xN Jacobian for a point on a link.

    Parameters
    ----------
    link_index : int
        1-based index of the link (1 = link01, 2 = link02, ... 6 = link06).
        Only joints 1..link_index affect this point.
    p_body : array-like, shape (3,)
        Position of the point in the body frame of the given link.
    q : array-like, shape (6,)
        Current joint configuration.
    model : mujoco.MjModel

    Returns
    -------
    J : np.ndarray, shape (6, 6)
        J[:3, :] = linear velocity Jacobian
        J[3:, :] = angular velocity Jacobian
    """
    q        = np.asarray(q, dtype=float)
    p_body   = np.asarray(p_body, dtype=float)
    n_joints = 6

    # Get the world transform of the target link
    body_name   = f'link0{link_index}'
    T_0_link    = _body_world_transform(q, model, body_name)

    # Express p in world frame
    p_world = (T_0_link @ np.append(p_body, 1.0))[:3]

    # Get all joint world frames
    joint_frames = compute_all_joint_world_frames(q, model)
    joint_names  = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

    J = np.zeros((6, n_joints))
    for i, jname in enumerate(joint_names):
        joint_i = i + 1   # 1-based
        if joint_i <= link_index:
            z_i = joint_frames[jname]['axis']
            o_i = joint_frames[jname]['origin']
            J[:3, i] = np.cross(z_i, p_world - o_i)   # linear
            J[3:, i] = z_i                              # angular
        # else: column stays zero

    return J


def _body_world_transform(q, model, body_name):
    """
    Get world transform of a specific body by walking the chain up to that body.
    Reuses the same logic as forward_kinematics but stops at the requested body.
    """
    body_names  = ['link00', 'link01', 'link02', 'link03', 'link04', 'link05', 'link06']
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

    T_world = np.eye(4)
    for idx, name in enumerate(body_names):
        body_id = model.body(name).id
        p = model.body_pos[body_id].copy()
        R = quaternion_to_rotation(model.body_quat[body_id])
        T_world = T_world @ make_transform(R, p)

        joint_idx = idx - 1
        if 1 <= idx <= 6:
            jid      = model.joint(joint_names[joint_idx]).id
            axis_loc = model.jnt_axis[jid].copy()
            axis_loc /= np.linalg.norm(axis_loc)
            c, s     = np.cos(q[joint_idx]), np.sin(q[joint_idx])
            K        = np.array([[          0, -axis_loc[2],  axis_loc[1]],
                                  [ axis_loc[2],           0, -axis_loc[0]],
                                  [-axis_loc[1],  axis_loc[0],           0]])
            R_jnt    = np.eye(3) + s * K + (1 - c) * (K @ K)
            T_world  = T_world @ make_transform(R_jnt, np.zeros(3))

        if name == body_name:
            return T_world

    raise ValueError(f"Body '{body_name}' not found in chain.")