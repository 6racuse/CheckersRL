

from Environment import CheckersEnv,is_move_vulnerable
import numpy as np
import random 
import matplotlib.pyplot as plt
class MonteCarloAgent:
    def __init__(self, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, gamma=0.97):
        self.Q = {}       
        self.returns = {}
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

    def get_state_hash(self, state):
        return tuple(tuple(row) for row in state)

    def policy(self, state):
        state_hash = self.get_state_hash(state)
        env = CheckersEnv(stalemate_threshold=120)
        moves = env.get_available_moves(state, env.WHITE_PAWN)
        if not moves:
            return None, None
        # Filtrer les coups vulnérables : si des coups sûrs existent, on les privilégie
        safe_moves = [move for move in moves if not is_move_vulnerable(move, state, env, env.WHITE_PAWN)]
        if safe_moves:
            moves = safe_moves
        if state_hash not in self.Q:
            self.Q[state_hash] = [0.0] * len(moves)
        if np.random.rand() < self.epsilon:
            action_index = random.randrange(len(moves))
        else:
            action_index = int(np.argmax(self.Q[state_hash]))
        return moves[action_index], action_index

    def update(self, episode):
        G = 0.0
        visited = set()
        for state, action_index, reward in reversed(episode):
            G = self.gamma * G + reward
            state_hash = self.get_state_hash(state)
            if (state_hash, action_index) not in visited:
                if (state_hash, action_index) not in self.returns:
                    self.returns[(state_hash, action_index)] = []
                self.returns[(state_hash, action_index)].append(G)
                self.Q[state_hash][action_index] = np.mean(self.returns[(state_hash, action_index)])
                visited.add((state_hash, action_index))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon
    
    
    

def compute_q_variance(agent):
    """
    Calcule la variance de toutes les Q-valeurs dans la Q-table de l'agent.
    """
    values = []
    for state_key in agent.Q:
        values.extend(agent.Q[state_key])
    if values:
        return np.var(values)
    else:
        return 0.0
def plot_q_convergence(q_variances):
    plt.figure(figsize=(8, 6))
    plt.plot(q_variances, label="Q-Value Variance")
    plt.xlabel("Episode")
    plt.ylabel("Variance of Q-values")
    plt.title("Convergence of Q-table (Variance over Episodes)")
    plt.legend()
    plt.show()