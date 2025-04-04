import math
import random 
from copy import deepcopy


class MCTSNode:
    """
    Monte-Carlo Tree Search Node
    """
    def __init__(self, state, player, env, parent=None, action=None):
        """
        Initializes the MCTS node

        Input : 
        - state: The provided state
        - player: The current player
        - env: The CheckersRL environment
        - parent: The parent node
        - action: The provided action
        """
        self.state = state
        self.player = player # (BLACK/WHITE)
        self.env = env
        self.parent = parent #This is the information linking each node
        self.action = action #The action information is used to expand the tree
        self.children = []
        # we set up the various leaves of the current node : it will be used to create the branches
        self.untried_actions = env.get_available_moves(self.state, self.player) 
        self.n = 0
        self.w = 0
        self.is_terminal = env.is_game_finished(self.state)[0]
    
    def fully_expanded(self):
        """
        Checks if all possible actions have been tried
        """
        return len(self.untried_actions) == 0
    

    def expand(self):
        """
        Picks an untried action, evaluates it, generates the node for 
        the resulting state (also add it to the children) and returns it.
        """
        action = self.untried_actions.pop()
        # we simulate the next state given the action applied to the environment
        next_state, next_player,_ = self.env.apply_action_to_environment(deepcopy(self.state), action, self.player)
        # this is where the recursion part of the MCTS algorithm is : we create the child node, with parent state as the current state 
        child_node = MCTSNode(next_state, next_player, self.env, parent=self, action=action)
        self.children.append(child_node)

        return child_node
    

    def random_simulation(self):
        """
        Performs a random simulation 
        """
        
        state = deepcopy(self.state)
        #first we check if the current state is not a terminal one
        done, result = self.env.is_game_finished(state)
        # if not, we select a random move from available ones, until the end of the simulated game
        while not done:
            available_actions = self.env.get_available_moves(state, self.player)
            if not available_actions:
                break
            action = random.choice(available_actions)
            state, self.player,_ = self.env.apply_action_to_environment(deepcopy(state), action, self.player)
            done, result = self.env.is_game_finished(state)

        return result #result is the winning side (BLACK_PAWN,WHITE_PAWN,None)
    

    def backpropagate(self, result):
        """
        Backpropagates the simulation result to the root, 
        adding winrate points to the best simulated scenarios (the closer to the root, the greater the winrate scenarios)
        """
        self.n += 1
        if self.parent: #if it's not the root
            if result == self.parent.player:
                self.w += 1
            elif result is not None:
                self.w -= 1
            self.parent.backpropagate(result) 


    def propagate(self):
        """
        Explore the tree until non-fully expanded node is reached
        """
        node = self

        while node.fully_expanded() and not node.is_terminal:
            #we retrive the best Upper confidence bound applied to trees child of the current node
            
            node = node.Get_best_uct() 
            
        # if the selected node is terminal, it return it, otherwise, it expands the node
        if node.is_terminal: 
            return node
        
        return node.expand()
    

    def winrate(self):
        """
        Returns the winrate the provided node
        """
        # self.w is incremented in the backpropagation phase, so is self.n
        return self.w/self.n if self.n > 0 else 0 


    def uct(self):
        """
        Retturns the UCT value
        """
        
        return self.winrate() + math.sqrt(2*math.log(self.parent.n)/self.n)
    

    def best_child(self):
        """
        Returns the child who has the best winrate
        """
        return max(self.children, key=lambda child: child.winrate())
    

    def Get_best_uct(self):
        """
        Returns child which has the greater UCT value (win rate + C*sqrt(ln(t)/s))
        """
        return max(self.children, key=lambda child: child.uct())
    

def mcts(state, player, env,iters=5000):
    """
    Runs the MCTS algorithm.
    
    The MCTS algorithm is composed of nodes, linked to one another with the 'parent' information (default : None). 
    The Algorithm propagates (simulates outcome of the game on children of current node) 
    and back-propagates the information of winrate to the init node.
    The next action chosen is the action entailing the best win_rate 
    
    In the particular case of Checkers, the tree is not that big given that the game is usually 30 actions long, 
    and the number of actions per node is limited, given that (according to checkers rules) a capture must occur if it is available   
    
    There sometimes is a problem if we have a low iters number, because the current state might be too far from a terminal state 
    (according to our experiments, it happens when there are too little pawns left on the board, and they opponents are far from each otehr )  
    
    Inputs : 
    - state : The current state (game board) 
    - player: The current player (WHITE/BLACK) 
    - env   : The environment 
    - iters : The number of simulation iterations
    
    Returns : 
    - best_action : the action leading to the best winrate
    - root : the current node
    
    """
    root = MCTSNode(deepcopy(state), player, env)
    # we set up the root of the MCTS to the current environment state
    
    for _ in range(iters):
        # first step of MCTS : we explore the tree whose root is the current state of the board
        # leaf is either the current node (if it is terminal), or one of its children
        leaf = root.propagate() 
        # second step of MCTS : get the best uct child of the current node, and hopefully converge to a winning state
        simulation_result = leaf.random_simulation()
        # simulation result returns the winning side of the random tree search (BLACK_PAWN,WHITE_PAWN,None)
        # third step of MCTS : we perform a back-propagation which computes the winrate of the nodes back to the children of the init node
        leaf.backpropagate(simulation_result)
    
    best_action  = root.best_child().action
    return best_action, root

        