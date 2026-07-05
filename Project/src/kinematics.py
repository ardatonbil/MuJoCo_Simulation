import numpy as np

def rotation_to_quaternion(R):
    """Rotation matrix to [w, x, y, z] quaternion, numerically stable."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s,
                         (R[2,1]-R[1,2])*s,
                         (R[0,2]-R[2,0])*s,
                         (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                         (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                         0.25*s,             (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                         (R[1,2]+R[2,1])/s, 0.25*s])


def quaternion_to_rotation(q):
    """[w, x, y, z] quaternion to rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([[1 - 2*(y**2 + z**2),  2*(x*y - w*z),       2*(x*z + w*y)      ],
                     [2*(x*y + w*z),         1 - 2*(x**2 + z**2), 2*(y*z - w*x)      ],
                     [2*(x*z - w*y),         2*(y*z + w*x),       1 - 2*(x**2 + y**2)]])


def quaternion_error_angle(q1, q2):
    """Scalar geodesic angular error in radians between two [w,x,y,z] quaternions."""
    q1 = np.asarray(q1, dtype=float);  q1 /= np.linalg.norm(q1)
    q2 = np.asarray(q2, dtype=float);  q2 /= np.linalg.norm(q2)
    return 2.0 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))


def make_transform(R, p):
    """Build a 4x4 homogeneous transform from rotation matrix R and translation p."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = p
    return T


def forward_kinematics(q, model):
    """
    Compute end-effector transform T_0_EE using homogeneous transformation matrices.
    Parameters are read directly from the MuJoCo model object.

    Chain:
        T_0_EE = T_00_01 @ T_01_02 @ T_02_03 @ T_03_04 @ T_04_05 @ T_05_06 @ T_06_EE

    Each T_i is built from:
        - body_pos[i]:  translation from parent body frame to this body frame
        - body_quat[i]: orientation of this body frame relative to parent (default identity)
        - jnt_axis[i]:  rotation axis of the joint in the local body frame
        - q[i]:         joint angle
    """

    # --- collect body ids ---
    b = {name: model.body(name).id
         for name in ['link00', 'link01', 'link02', 'link03',
                      'link04', 'link05', 'link06']}

    # --- collect joint ids (joint lives in the child body) ---
    j = {name: model.joint(name).id
         for name in ['joint1', 'joint2', 'joint3',
                      'joint4', 'joint5', 'joint6']}

    # --- attachment site ---
    site_id = model.site('attachment_site').id

    def body_fixed_transform(body_id):
        """Fixed transform from parent frame to this body frame (ignores joint)."""
        p = model.body_pos[body_id].copy()
        R = quaternion_to_rotation(model.body_quat[body_id])
        return make_transform(R, p)

    def joint_rotation_transform(joint_id, angle):
        """Rotation transform for a revolute joint about its local axis."""
        axis = model.jnt_axis[joint_id].copy()
        axis /= np.linalg.norm(axis)
        c, s = np.cos(angle), np.sin(angle)
        # Rodrigues formula
        K = np.array([[    0, -axis[2],  axis[1]],
                      [ axis[2],     0, -axis[0]],
                      [-axis[1],  axis[0],     0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        return make_transform(R, np.zeros(3))

    # --- build the chain ---
    # link00 is the fixed base (no joint)
    T_world_00 = body_fixed_transform(b['link00'])

    # Each subsequent link: fixed body offset, then joint rotation
    T_00_01 = body_fixed_transform(b['link01']) @ joint_rotation_transform(j['joint1'], q[0])
    T_01_02 = body_fixed_transform(b['link02']) @ joint_rotation_transform(j['joint2'], q[1])
    T_02_03 = body_fixed_transform(b['link03']) @ joint_rotation_transform(j['joint3'], q[2])
    T_03_04 = body_fixed_transform(b['link04']) @ joint_rotation_transform(j['joint4'], q[3])
    T_04_05 = body_fixed_transform(b['link05']) @ joint_rotation_transform(j['joint5'], q[4])
    T_05_06 = body_fixed_transform(b['link06']) @ joint_rotation_transform(j['joint6'], q[5])

    # Fixed offset from link06 to the attachment site (no joint)
    site_pos = model.site_pos[site_id].copy()   # [0.051, 0, 0]
    T_06_EE  = make_transform(np.eye(3), site_pos)

    T_0_EE = T_world_00 @ T_00_01 @ T_01_02 @ T_02_03 @ T_03_04 @ T_04_05 @ T_05_06 @ T_06_EE
    return T_0_EE


def end_effector_position(q, model):
    return forward_kinematics(q, model)[:3, 3]


def end_effector_orientation(q, model):
    """Returns [w,x,y,z] quaternion of end-effector orientation."""
    return rotation_to_quaternion(forward_kinematics(q, model)[:3, :3])