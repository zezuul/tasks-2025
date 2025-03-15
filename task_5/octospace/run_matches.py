import argparse
import gymnasium as gym
import numpy as np
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from importlib.machinery import SourceFileLoader

# from database.utils import task_5_update
from task_5.octospace.matches_config import TEAMS
from task_5.octospace.simulation import simulate_game
# from database.db_config import DB_USER, DB_PASSWORD
from tqdm import tqdm


"""
Score:
    - (+) time for agent decision
    - (+) fewer number of steps in match
"""
LIST_OF_PLAYERS_FULL = os.listdir('task_5/octospace/agents/') # only for time testing, you can remove later
LIST_OF_PLAYERS = LIST_OF_PLAYERS_FULL[:]


def get_parser():
    parser = argparse.ArgumentParser(description='Run matches between agents')
    parser.add_argument('--n_matches', type=int, default=3, help='Number of matches to run')
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    parser.add_argument('--render_mode', type=str, default=None, help='Render mode')
    parser.add_argument('--turn_on_music', type=bool, default=False, help='Music')
    return parser


# async def save_to_db(scores: np.ndarray[float]):
#     for i in range(len(TEAMS)):
#         response = await task_5_update(db_user=DB_USER, db_password=DB_PASSWORD, team_name=TEAMS[i] , rating=scores[i])
#         print(response)
#     print('Data saved to the database')


def get_ratings(scores: np.ndarray[float]) -> np.ndarray[float]:
    # TODO
    pass


def run_matches(n_matches: int, render_mode: str = None, verbose: bool = False, turn_on_music: bool = False) -> np.ndarray[float]:
    # Disable warnings in the gym
    if not verbose:
        gym.logger.min_level = 40

    list_of_players = os.listdir('task_5/octospace/agents/')

    scores = np.zeros(len(TEAMS), dtype=int)

    for i in range(len(list_of_players)):
        for j in range(i+1, len(list_of_players)):
            player_1_id = int(list_of_players[i])
            player_2_id = int(list_of_players[j])
            agent_1 = SourceFileLoader('agent_1', f'task_5/octospace/agents/{list_of_players[i]}/agent.py').load_module()
            agent_2 = SourceFileLoader('agent_2', f'task_5/octospace/agents/{list_of_players[j]}/agent.py').load_module()

            score = simulate_game(player_1_id=player_1_id, player_2_id=player_2_id, player_1_agent_class=agent_1.Agent,
                                player_2_agent_class=agent_2.Agent, n_games=n_matches, render_mode=render_mode, verbose=False,
                                  turn_on_music=turn_on_music)

            scores[player_1_id] += score[0]
            scores[player_2_id] += score[1]
            print(f'{TEAMS[player_1_id]} vs {TEAMS[player_2_id]}: {score}')

    return scores


def run_matches_async(n_matches: int, render_mode: str = None, verbose: bool = False) -> np.ndarray[float]:
    games = [(i, j, n_matches) for i in range(len(LIST_OF_PLAYERS)) for j in range(i + 1, len(LIST_OF_PLAYERS))]
    scores = np.zeros(len(TEAMS), dtype=int)

    # For now I tested it for CPU count - you can adjust it for the Athena
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(run_single_match, games), total=len(games)))

    for player_1_id, player_2_id, score in results:
        scores[player_1_id] += score[0]
        scores[player_2_id] += score[1]

    return scores


def run_single_match(args):
    player_1_id, player_2_id, n_matches = args

    agent_1 = SourceFileLoader('agent_1', f'task_5/octospace/agents/{LIST_OF_PLAYERS[player_1_id]}/agent.py').load_module()
    agent_2 = SourceFileLoader('agent_2', f'task_5/octospace/agents/{LIST_OF_PLAYERS[player_2_id]}/agent.py').load_module()

    score = simulate_game(
        player_1_id=player_1_id, player_2_id=player_2_id,
        player_1_agent_class=agent_1.Agent, player_2_agent_class=agent_2.Agent,
        n_games=n_matches, render_mode=None, verbose=False
    )

    return player_1_id, player_2_id, score


if __name__ == '__main__':

    parse = get_parser()
    args = parse.parse_args()

    standard_times = list()
    multiprocess_times = list()

    # standard
    scores = run_matches(n_matches=args.n_matches, render_mode=args.render_mode, verbose=args.verbose, turn_on_music=args.turn_on_music)
    #
    # print(scores)

    # async
    # scores = run_matches_async(n_matches=args.n_matches, render_mode=args.render_mode, verbose=args.verbose)

    print(scores)
    # asyncio.run(save_to_db(scores))

    # """
    # TIME COMPARISION BETWEEN STANDARD AND ASYNC
    # """
    #
    # import timeit
    # import matplotlib.pyplot as plt
    #
    # for i in range(2, 10):
    #     LIST_OF_PLAYERS = LIST_OF_PLAYERS_FULL[:i]
    #     print(LIST_OF_PLAYERS)
    #
    #     start = timeit.default_timer()
    #     scores = run_matches(n_matches=args.n_matches, render_mode=args.render_mode, verbose=args.verbose)
    #     stop = timeit.default_timer()
    #     standard_times.append(stop-start)
    #
    #     start = timeit.default_timer()
    #     scores = run_matches_async(n_matches=args.n_matches, render_mode=args.render_mode, verbose=args.verbose)
    #     stop = timeit.default_timer()
    #     multiprocess_times.append(stop-start)
    #
    # print(standard_times)
    # print(multiprocess_times)
    #
    # plt.plot(standard_times)
    # plt.plot(multiprocess_times)
    # plt.show()