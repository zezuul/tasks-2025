from gymnasium.envs.registration import register

register(
    id="OctoSpace-v0",
    entry_point="task_5.octospace.octospace.envs:OctoSpaceEnv",
)
