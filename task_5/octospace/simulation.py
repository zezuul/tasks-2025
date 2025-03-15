import gymnasium as gym
import numpy as np
import os

import octospace
import pygame

from dummy_agent import Agent


def setup_agent(agent: Agent, player_id: int):
    agent.load(os.path.abspath(f"agents/{player_id}/"))
    agent.eval()


def simulate_game(
    player_1_id: int,
    player_2_id: int,
    player_1_agent_class: Agent.__class__,
    player_2_agent_class: Agent.__class__,
    n_games: int = 3,
    render_mode: str = "human",
    verbose: bool = False,
    turn_on_music: bool = False,
):
    if not verbose:
        gym.logger.min_level = 40

    env = gym.make('OctoSpace-v0', player_1_id=player_1_id, player_2_id=player_2_id, max_steps=1000,
                   render_mode=render_mode, debug=False, turn_on_music=turn_on_music, volume=0.1)
    obs, info = env.reset()

    agent_1 = player_1_agent_class()
    agent_2 = player_2_agent_class()

    terminated = False
    reward = {}

    score = np.array([0, 0], dtype=float)
    curr_round = 0

    while curr_round / 2 != n_games:
        if terminated or sum(reward.values()) != 0:
            curr_round += 1
            score += np.array(list(reward.values()))
            obs, info = env.reset()
            agent_1 = player_1_agent_class()
            agent_2 = player_2_agent_class()

        env.render()

        action_1 = agent_1.get_action(obs["player_1"])
        action_2 = agent_2.get_action(obs["player_2"])

        obs, reward, terminated, _, info = env.step(
            {
                "player_1": action_1,
                "player_2": action_2
            }
        )

        if render_mode is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return -1

    return score


