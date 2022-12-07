# Distributionally Robust Model Predictive Control for Nonlinear Process Systems

This repository includes the code for the paper "[Tube-based Distributionally Robust Model Predictive Control for Nonlinear Process Systems via Linearization](https://arxiv.org/abs/2211.14595)".

## Abstract

Model predictive control (MPC) is an effective approach to control multivariable dynamic systems with constraints. Most real dynamic models are however affected by plant-model mismatch and process uncertainties, which can lead to closed-loop performance deterioration and constraint violations. Methods such as stochastic MPC (SMPC) have been proposed to alleviate these problems; however, the resulting closed-loop state trajectory might still significantly violate the prescribed constraints if the real system deviates from the assumed disturbance distributions made during the controller design. In this work we propose a novel data-driven distributionally robust MPC scheme for nonlinear systems. Unlike SMPC, which requires the exact knowledge of the disturbance distribution, our scheme decides the control action with respect to the worst distribution from a distribution ambiguity set. This ambiguity set is defined as a Wasserstein ball centered at the empirical distribution. Due to the potential model errors that cause off-sets, the scheme is also extended by leveraging an offset-free method. The favorable results of this control scheme are demonstrated and empirically verified with a nonlinear mass spring system and a nonlinear CSTR case study.

## How to Run Experiments

``python CSTR.py `` or ``python academic_model.py`` to run the simulation for one realization.

## Simulation results

### Simulation 1: Comparison between feedback linearization and successive linearization for nonlinear mass-spring systems

Simulation results of DRMPC using successive linearization averaged from 500 realizations with one sample and ball radius ranging from $0.001$ to $5$ on the nonlinear mass spring system. Solid lines are the expected trajectories and  shaded areas represent $15-75\%$ percentile of trajectories.
<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/SucL_baseline.png" width="50%">
</P>

Simulation results of DRMPC using feedback linearization averaged from 500 realizations with one sample and ball radius ranging from $0.001$ to $5$ on the nonlinear mass spring system. Solid lines are the expected trajectories and  shaded areas represent $15-75\%$ percentile of trajectories.
<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/FeedL.png" width="50%">
</P>

### Simulation 2: Comparison between two conic formulations applying successive linearization for case study 1

Illustration of the comparison between two conic formulations applying successive linearization.

<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/two_constraint_types.png" width="50%">
</P>


### Simulation 3: Comparison between successive linearization and polynomial chaos SMPC for case study 1

Simulation results comparing DRMPC using successive linearization and PC-based SMPC in the nominal scenario averaged from 500 realizations with one sample and ball radius ranging from $0.001$ to $5$ on the nonlinear mass spring system. Solid lines are the expected trajectories and  shaded areas represent $15-75\%$ percentile of trajectories.
<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/SucL_PolyC.png" width="50%">
</P>

Simulation results comparing DRMPC using successive linearization and PC-based SMPC under modified distribution  averaged from 500 realizations. DRMPC collects five samples online initially from one sample and ball radius ranges from $0.001$ to $5$ on the nonlinear mass spring system. Solid lines are the expected trajectories and  shaded areas represent $15-75\%$ percentile of trajectories.
<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/SucL_PolyC_shifted_learning.png" width="50%">
</P>


### Simulation 4: Comparison between successive linearization and polynomial chaos SMPC for case study 2

Simulation results comparing DRMPC using successive linearization and PC-based SMPC in the nominal scenario averaged from 500 realizations with one sample and ball radius ranging from $0.001$ to $5$ on the CSTR system. Solid lines are the expected trajectories and  shaded areas represent $15-75\%$ percentile of trajectories.

<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/CSTR_SucL_PolyC1.png" width="50%">
</P>

Simulation results comparing DRMPC using successive linearization and PC-based SMPC under modified distribution  averaged from 500 realizations. DRMPC collects five samples online initially from one sample and ball radius ranges from $0.001$ to $5$ on the CSTR system. Solid lines are the expected trajectories and  shaded areas represent $15-75\%$ percentile of trajectories.

<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/CSTR_SucL_PolyC_shifted_learning1.png" width="50%">
</P>

Constraint violation rate for the case study 1. Relation between the ball radius and constraint violations, averaged from $500$ realizations of trajectories
<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/constraint_vio1.png" width="50%">
</P>

Constraint violation rate for the case study 2. Relation between the ball radius and constraint violations, averaged from $500$ realizations of trajectories.
<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/constraint_vio2.png" width="50%">
</P>



### Simulation 5: Offset-free tracking

Simulation results for offset-free DRMPC using successive linearization in the nominal scenario averaged from 500 realizations with one sample and ball radius ranging from 0.001 to 5 on the CSTR system. Solid lines are the expected trajectories and shaded areas represent 15 − 75% percentile of trajectories.

<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/CSTR_SucL_PolyC1_Offset_free.png" width="50%">
</P>

Simulation results for offset-free DRMPC using successive linearization under modified distribution averaged from 500 realizations with one sample and ball radius ranges from 0.001 to 5 on the CSTR system. Solid lines are the expected trajectories and shaded areas represent 15 − 75% percentile of trajectories.

<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/CSTR_SucL_PolyC_shifted_learning_offset_free.png" width="50%">
</P>

Constraint violation rate for the case study 3. Relation between the ball radius and constraint violations, averaged from 500 realizations of trajectories.

<p align="center">
  <img src="https://github.com/zhengang-zhong/DRMPC_nonlinear_system_via_linearization/blob/main/figs/constraint_vio1_offset_free.png" width="50%">
</P>