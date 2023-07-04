""" This module record code removed from original macad-gym and carla/PythonAPI in case 
further references are needed. """


# Removed from macad_gym.core.multi_env.MultiCarlaEnv
def _step(self, actor_id, action):
    """Perform the actual step in the CARLA environment

    Applies control to `actor_id` based on `action`, process measurements,
    compute the rewards and terminal state info (dones).

    Args:
        actor_id(str): Actor identifier
        action: Actions to be executed for the actor.

    Returns
        obs (obs_space): Observation for the actor whose id is actor_id.
        reward (float): Reward for actor. None for first step
        done (bool): Done value for actor.
        info (dict): Info for actor.
    """

    if self._discrete_actions:
        action = DISCRETE_ACTIONS[int(action)]
    assert len(action) == 2, "Invalid action {}".format(action)
    if self._squash_action_logits:
        forward = 2 * float(sigmoid(action[0]) - 0.5)
        throttle = float(np.clip(forward, 0, 1))
        brake = float(np.abs(np.clip(forward, -1, 0)))
        steer = 2 * float(sigmoid(action[1]) - 0.5)
    else:
        throttle = float(np.clip(action[0], 0, 0.6))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
    reverse = False
    hand_brake = False
    if self._verbose:
        print(
            "steer", steer, "throttle", throttle, "brake", brake, "reverse", reverse
        )

    config = self._actor_configs[actor_id]
    if config["manual_control"]:
        self._control_clock.tick(60)
        self._manual_control_camera_manager._hud.tick(
            self.world,
            self._actors[actor_id],
            self._collisions[actor_id],
            self._control_clock,
        )
        self._manual_controller.parse_events(self, self._control_clock)

        # TODO: consider move this to Render as well
        self._manual_control_camera_manager.render(
            Render.get_screen(), self._manual_control_render_pose
        )
        self._manual_control_camera_manager._hud.render(
            Render.get_screen(), self._manual_control_render_pose
        )
        pygame.display.flip()
    elif config["auto_control"]:
        if getattr(self._actors[actor_id], "set_autopilot", 0):
            self._actors[actor_id].set_autopilot(
                True, self._traffic_manager.get_port()
            )
    else:
        # TODO: Planner based on waypoints.
        # cur_location = self.actor_list[i].get_location()
        # dst_location = carla.Location(x = self.end_pos[i][0],
        # y = self.end_pos[i][1], z = self.end_pos[i][2])
        # cur_map = self.world.get_map()
        # next_point_transform = get_transform_from_nearest_way_point(
        # cur_map, cur_location, dst_location)
        # the point with z = 0, and the default z of cars are 40
        # next_point_transform.location.z = 40
        # self.actor_list[i].set_transform(next_point_transform)

        agent_type = config.get("type", "vehicle")
        # TODO: Add proper support for pedestrian actor according to action
        # space of ped actors
        if agent_type == "pedestrian":
            rotation = self._actors[actor_id].get_transform().rotation
            rotation.yaw += steer * 10.0
            x_dir = math.cos(math.radians(rotation.yaw))
            y_dir = math.sin(math.radians(rotation.yaw))

            self._actors[actor_id].apply_control(
                carla.WalkerControl(
                    speed=3.0 * throttle,
                    direction=carla.Vector3D(x_dir, y_dir, 0.0),
                )
            )

        # TODO: Change this if different vehicle types (Eg.:vehicle_4W,
        #  vehicle_2W, etc) have different control APIs
        elif "vehicle" in agent_type:
            self._actors[actor_id].apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=steer,
                    brake=brake,
                    hand_brake=hand_brake,
                    reverse=reverse,
                )
            )
    # Asynchronosly (one actor at a time; not all at once in a sync) apply
    # actor actions & perform a server tick after each actor's apply_action
    # if running with sync_server steps
    # NOTE: A distinction is made between "(A)Synchronous Environment" and
    # "(A)Synchronous (carla) server"
    if self._sync_server:
        if self._render:
            spectator = self.world.get_spectator()
            transform = self._actors[actor_id].get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=80),
                                                    carla.Rotation(pitch=-90)))
        self.world.tick()
        # `wait_for_tick` is no longer needed, see https://github.com/carla-simulator/carla/pull/1803
        # self.world.wait_for_tick()

    # Process observations
    py_measurements = self._read_observation(actor_id)
    if self._verbose:
        print("Next command", py_measurements["next_command"])
    # Store previous action
    self._previous_actions[actor_id] = action
    if type(action) is np.ndarray:
        py_measurements["action"] = [float(a) for a in action]
    else:
        py_measurements["action"] = action
    py_measurements["control"] = {
        "steer": steer,
        "throttle": throttle,
        "brake": brake,
        "reverse": reverse,
        "hand_brake": hand_brake,
    }

    # Compute truncated
    truncated = (
        False
    )

    # Compute done
    done = (
        self._num_steps[actor_id] > self._scenario_map["max_steps"]
        or py_measurements["next_command"] == "REACH_GOAL"
        or (
            config["early_terminate_on_collision"]
            and collided_done(py_measurements)
        )
    )
    py_measurements["done"] = done
    py_measurements["truncated"] = truncated

    # Compute reward
    config = self._actor_configs[actor_id]
    flag = config["reward_function"]
    reward = self._reward_policy.compute_reward(
        self._prev_measurement[actor_id], py_measurements, flag
    )

    self._previous_rewards[actor_id] = reward
    if self._total_reward[actor_id] is None:
        self._total_reward[actor_id] = reward
    else:
        self._total_reward[actor_id] += reward

    py_measurements["reward"] = reward
    py_measurements["total_reward"] = self._total_reward[actor_id]

    # End iteration updating parameters and logging
    self._prev_measurement[actor_id] = py_measurements
    self._num_steps[actor_id] += 1

    if config["log_measurements"] and CARLA_OUT_PATH:
        # Write out measurements to file
        if not self._measurements_file_dict[actor_id]:
            self._measurements_file_dict[actor_id] = open(
                os.path.join(
                    CARLA_OUT_PATH,
                    "measurements_{}.json".format(self._episode_id_dict[actor_id]),
                ),
                "w",
            )
        self._measurements_file_dict[actor_id].write(json.dumps(py_measurements))
        self._measurements_file_dict[actor_id].write("\n")
        if done:
            self._measurements_file_dict[actor_id].close()
            self._measurements_file_dict[actor_id] = None
            # if self.config["convert_images_to_video"] and\
            #  (not self.video):
            #    self.images_to_video()
            #    self.video = Trueseg_city_space
    original_image = self._cameras[actor_id].image

    return (
        self._encode_obs(actor_id, original_image, py_measurements),
        reward,
        done,
        py_measurements,
    )
