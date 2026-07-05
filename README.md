ME 403 Project: Unitree Z1 Robotic Arm Simulation 

*This project was developed by a team of 3 members.*

## Overview

This project focuses on applying robotics principles and control methods to a simulated Unitree Z1 robotic arm. The Z1 is a commercial robotic arm featuring a base rotator, a two-link elbow, and three tightly-packed wrist joints.

The simulation environment is built on MuJoCo  and utilizes Python. While the provided simulation supports both position and torque modes , the core of this project is implemented in torque mode.

## Features & Implementation

The repository contains the code and mathematical models to achieve the following:

* 
**Kinematics Modeling:** Forward kinematics of the end-effector using homogeneous transformation matrices.


* 
**Jacobian Derivation:** A programmatic approach to derive the Jacobian matrix for any given point on the robot's body, utilizing the link index and local position.


* 
**Gravity Compensation:** Calculation of the generalized gravitational forces applied to the structure and the application of necessary joint torques to successfully cancel gravity across the workspace.


* 
**System Dynamics:** Computation of the generalized mass matrix (using the Jacobian of each link's Center of Mass) and the Coriolis matrix (derived via Christoffel Symbols).


* 
**Static Position Control:** A configuration space position controller designed to drive the end-effector to arbitrary static points.


* 
**Trajectory Tracking:** A position controller capable of tracking dynamic target configurations formulated as time-dependent sine waves:



$$q_{i}=A_{i}sin(w_{i}t)$$





* 
**Impedance Control:** An impedance controller designed to dynamically drive the end-effector to target static points using impedance forces.



## Project Structure

The core files required to run the simulation are contained within the working directory:

* 
`z1_torque.xml`: The XML description of the physical robot parameters (inertia, dimensions, COM, etc.).


* 
`scene_torque.xml`: The environment setup that references the robot description.


* 
`sample_z1_torque.py`: The main Python execution script containing the simulation loop.



## Usage

1. Ensure MuJoCo and the necessary Python bindings are installed on your system.
2. Run `sample_z1_torque.py`.


3. The MuJoCo viewport will open. In torque mode, you can expand the "Control" panel on the right side to manually assign joint torques, or let the implemented controllers handle the joint states.
