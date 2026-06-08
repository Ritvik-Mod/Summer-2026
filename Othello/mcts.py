import math
import torch
import numpy as np
import othello_env
import network

class Node:
    def __init__(self, env):
        self.env = env
        self.is_expanded = False

        self.children = {} #maps action --> child node

        #stats based on actions from a state (keys are actions)
        self.N = {} #visit count per action
        self.W = {} #total value per action
        self.Q = {} #mean value per action - W/N
        self.P = {} #prior probability per action - from network

#PUCT - Predictor + Upper Confidence Bound Applied to Trees

def select_action(node, c_puct = 1.0):
    total_N = sum(node.N.values()) # -> total visits from all actions at this node
    sqrt_total = math.sqrt(total_N)

    best_score = -math.inf
    best_action = None

    #looping over all legal actions of this node. (probabilities are from the network and in the networks logic we alrdy mask the output to make sure the network outputs only legal actions probability logits)
    for a in node.P:
        Q = node.Q[a]
        P = node.P[a]
        N = node.N[a]
    
        #PUCT formula
        u = c_puct * P * sqrt_total / (1 + N)
        score = Q + u

        if score > best_score:
            best_score = score
            best_action = a
    
    return best_action

def search(node, net):
    #if terminal node
    if node.env.done:
        return -node.env.outcome_for_current_player()
    
    #if leaf node(not expanded yet) - so we will expand and evaluate
    if not node.is_expanded:
        value = expand(node, net) #calling the network, storing the prior probabs, creating edges
        return -value #since when we go up for the next player the signs will be opposite
    
    #alrdy expanded - so we will select it and recurse
    a = select_action(node)

    if a not in node.children:
        child_env = node.env.clone()
        child_env.step(a)
        node.children[a] = Node(child_env)

    child = node.children[a]

    value = search(child, net) # --> recursion here. this will return the value from the child's perspective

    node.N[a] += 1
    node.W[a] += value
    node.Q[a] = node.W[a] / node.N[a]

    return -value # --> negating the value as we move up another level

def expand(node, net):
    state = node.env.state()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(net.parameters()).device)

    net.eval()
    with torch.no_grad():
        policy_logits, value = net(state_tensor)
    
    mask = node.env.legal_mask()
    logits = policy_logits[0]
    mask_t = torch.from_numpy(mask).to(logits.device)
    logits = logits.masked_fill(~mask_t, -math.inf)
    priors = torch.softmax(logits, dim=0)

    legal_indices = np.where(mask)[0]
    for a in legal_indices:
        a = int(a)
        node.P[a] = priors[a].item()
        node.N[a] = 0
        node.W[a] = 0.0
        node.Q[a] = 0.0

    node.is_expanded = True
    return value.item()

def run_mcts(env, net, num_sims, c_puct=1.0, add_Noise=False, alpha=0.3, eps=0.25):
    root = Node(env.clone())

    search(root, net) #expanding the root once before we add noise

    if add_Noise and root.P:
        actions = list(root.P.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            root.P[a] = (1 - eps) * root.P[a] + eps * n

    for _ in range(num_sims - 1):
        search(root, net)
    
    counts = root.N #counts is a dict action --> visit count
    total = sum(counts.values())

    pi = np.zeros(env.action_size)
    for a in counts:
        pi[a] = counts[a] / total

    return pi

def play_game(player_fn_black, player_fn_white, n=8):
    env = othello_env.OthelloEnv(n)
    while not env.done:
        legal = env.legal_actions()
        if not legal:
            env.step(env.PASS)
            continue
        if env.to_move == 1: #black
            a = player_fn_black(env)
        else:
            a = player_fn_white(env)
        env.step(a)
        
    result_current = env.outcome_for_current_player()
    absolute_result = result_current * env.to_move
    return absolute_result

def random_player(env):
    legal = env.legal_actions()
    return int(np.random.choice(legal))

def mcts_player(env, net, sims):
    pi = run_mcts(env, net, sims)
    return int(np.argmax(pi))

def evaluate_vs_random(net, sims, games=20,n=8):
    mcts_wins = 0
    mcts_fun = lambda env : mcts_player(env,net,sims)
    rand_fn = lambda env : random_player(env)
    for i in range(games):
        if not (i&1): #mcts plays black and random plays white
            result = play_game(mcts_fun, rand_fn,n)
            if result == 1: mcts_wins += 1
        else: #mcts plays white and random plays black
            result = play_game(rand_fn, mcts_fun,n)
            if result == -1: mcts_wins += 1
    return mcts_wins/games

if __name__ == "__main__":

    """
    The Algo to test whether the above mcts logic is working or has some flaw

    if we get mcts win rate of more than 50% and if the rate climbs up as we 
    increase the sim count, then our mcts logic is fine

    --> this is cause the other player we made is a random player who will pick random moves
    --> so if our mcts player was also just randomly picking moves the rate would stay around 50
    --> but our mcts search does look up for future moves (even tho untrained net rn but some looking happens)
    --> which provides some insights into which moves leads to win and which dont
    --> hence the win rate for mcts should be more than 50% for our test to pass
    """

    win_rate = evaluate_vs_random(network.Othello_Net(6), 100, 40)

    print(f'MCTS win rate vs random: {win_rate:.2f}')
