import threading
import numpy as np
import cv2
import os

from train.utils import preprocess_image
import math
import time
import argparse

from process import postprocess
import matplotlib.pyplot as plt

#LGSVL planner
from selfdrive.controls.lib.lane_planner import LanePlanner
from polyfuzz.polyfuzz import PolyFuzz, LaneLine, ModelOutput, VehicleState

PLAN_PER_PREDICT = 25

v_ego = 28.256250

def update_trajectory(model_outs, max_steering_angle_increase=0.25, start_steering_angle=None, add_noise=False):
        vehicle_state = VehicleState()

        if start_steering_angle is None:
            current_steering_angle = 0
        else:
            current_steering_angle = start_steering_angle

        vehicle_state.update_steer(current_steering_angle)

        total_lateral_shift = 0
        lateral_shift = 0
        lateral_shift_openpilot = 0

        yaw = 0
        yaw_diff = 0
        long_noise = 0
        logger.info("current_steering_angle: {}".format(current_steering_angle))

        list_total_lateral_shift = []
        list_lateral_shift_openpilot = []
        list_yaw = []
        list_desired_steering_angle = []
        list_current_steering_angle = []
        list_state = []
        ###

        PF = PolyFuzz()
        for i in range(self.n_frames):  # loop on 20Hz
            # update vehicle state
            v_ego = self.df_sensors.loc[i, "speed"]
            vehicle_state.update_velocity(v_ego)
            vehicle_state.update_steer(current_steering_angle)

            model_out = model_outs[i]

            valid, cost, angle, angle_steers_des_mpc = get_steer_angle(
                PF, model_out, v_ego, current_steering_angle
            )
            if i == 0 and start_steering_angle is None:
                current_steering_angle = angle_steers_des_mpc

            logger.info(
                "{}: valid: {}, cost: {}, angle: {}".format(i, valid, cost, angle)
            )

            logger.info(
                "desired steering angle: {}, current steering angle: {}".format(
                    angle_steers_des_mpc, current_steering_angle
                )
            )
            # update steering angle
            budget_steering_angle = angle_steers_des_mpc - current_steering_angle
            for _ in range(PLAN_PER_PREDICT):  # loop on 100Hz
                logger.debug(f"current_steering_angle 100Hz: {current_steering_angle}")
                angle_change = np.clip(
                    budget_steering_angle,
                    -max_steering_angle_increase,
                    max_steering_angle_increase,
                )
                current_steering_angle += angle_change
                budget_steering_angle -= angle_change
                if angle_steers_des_mpc - current_steering_angle > 0:
                    budget_steering_angle = max(budget_steering_angle, 0)
                else:
                    budget_steering_angle = min(budget_steering_angle, 0)

                state = vehicle_state.apply_plan(current_steering_angle)

            total_lateral_shift = state.y
            yaw = state.yaw

            list_state.append(state)
            list_yaw.append(yaw)
            list_total_lateral_shift.append(total_lateral_shift)
            list_lateral_shift_openpilot.append(lateral_shift_openpilot)

            list_desired_steering_angle.append(angle_steers_des_mpc)
            list_current_steering_angle.append(current_steering_angle)

        self.list_state = list_state
        self.list_yaw = list_yaw
        self.list_total_lateral_shift = list_total_lateral_shift
        self.list_desired_steering_angle = list_desired_steering_angle
        self.list_current_steering_angle = list_current_steering_angle
        self.list_lateral_shift_openpilot = list_lateral_shift_openpilot


def plan_and_control(model_outs, max_steering_angle_increase=0.25, start_steering_angle=None):  # loop on 20 hz?
    v_ego = self.df_sensors.loc[i, "speed"]
    vehicle_state.update_velocity(v_ego)
    vehicle_state.update_steer(current_steering_angle)

    model_out = model_outs

    PF = PolyFuzz()
    valid, cost, angle, angle_steers_des_mpc = get_steer_angle(
        PF, model_out, v_ego, current_steering_angle
    )

    print("{}: valid: {}, cost: {}, angle: {}".format(i, valid, cost, angle))
    print("desired steering angle: {}, current steering angle: {}".format(angle_steers_des_mpc, current_steering_angle))

    budget_steering_angle = angle_steers_des_mpc - current_steering_angle
    
    for _ in range(PLAN_PER_PREDICT):  # loop on 100Hz
        print("current_steering_angle 100Hz: %f" % (current_steering_angle))
        angle_change = np.clip(
            budget_steering_angle,
            -max_steering_angle_increase,
            max_steering_angle_increase,
        )
        current_steering_angle += angle_change
        budget_steering_angle -= angle_change
        if angle_steers_des_mpc - current_steering_angle > 0:
            budget_steering_angle = max(budget_steering_angle, 0)
        else:
            budget_steering_angle = min(budget_steering_angle, 0)

        state = vehicle_state.apply_plan(current_steering_angle)

    total_lateral_shift = state.y
    yaw = state.yaw



# needs polyfuzz
def get_steer_angle(PF, model_output, v=29, steering_angle=4):
    # PF = PolyFuzz()
    PF.update_state(v, steering_angle)
    path_poly, left_poly, right_poly, left_prob, right_prob = postprocess(model_output)
    valid, cost, angle = PF.run(left_poly, right_poly, path_poly, left_prob, right_prob)
    angle_steers_des = PF.angle_steers_des_mpc
    return valid, cost, angle, angle_steers_des


left_poly = [-3.45917902e-07, -1.00115515e-04, 1.66893143e-02, 1.53663266e+00] 
right_poly = [ 2.61395518e-07, -2.10013468e-04, 1.20487389e-02, -1.80241835e+00] 
path_poly = [-3.44784759e-07, 8.59623129e-05, 7.66798126e-03, -8.51272792e-03]
left_prob = 0.2426844673294238 
right_prob = 0.20274714555223178

LP = LanePlanner()
md = ModelOutput(left_poly, right_poly, path_poly, left_prob, right_prob)
LP.update(v_ego, md)

print("Working")