"""
we create the training data here for our network to be trained on.
we let the network and mcts work together and play multiple games and we will record
the triplets (state, policy - pi, z - winner value)
since we wont know the winner on each state, for each state instead of z we fill in player to move
and at the end of the game whoever wins, we go and put that value for all z's of that game
"""

import othello_env
import network
import mcts
import numpy as np

def play_one_selfplay_game(net, n=8, sims=25):
    env = othello_env.OthelloEnv(n)
    trajectory = [] #triplet list (state, pi, player to move)

    while not env.done:
        legal = env.legal_actions()
        if not legal:
            env.step(env.PASS)
            continue
    
        pi = mcts.run_mcts(env, net, sims, add_Noise=True) #running mcts to get the improved policy

        state = env.state() # (2,n,n) from current player's pov
        trajectory.append((state,pi,env.to_move))

        pi = pi / pi.sum()
        action =  np.random.choice(len(pi), p = pi)
        env.step(int(action))

    result_current = env.outcome_for_current_player()
    absolute_result = result_current * env.to_move

    examples = []

    for (state, pi, to_move) in trajectory:
        z = absolute_result * to_move
        examples.append((state, pi, z))
    
    return examples

def generate_selfplay_data(net, num_games, n=8, sims=25):
    all_examples = []
    for g in range(num_games):
        all_examples.extend(play_one_selfplay_game(net, n, sims))
    return all_examples

if __name__ == "__main__":
    """
    generating one game to test and see the no of examples it generates and whether it generates the z values properly
    """
    net = network.Othello_Net(6)
    test_examples = play_one_selfplay_game(net)
    print(f"No. of examples generated: {len(test_examples)}")
    for i, (state,pi,z) in enumerate(test_examples):
        print(f'Example {i}) {z}')
