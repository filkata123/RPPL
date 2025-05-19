import networkx as nx
import random

# Q-learning parameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.1    # Exploration rate
episodes = 1000  # Number of learning episodes
max_steps = 100  # Max steps per episode

def q_learning(graph, goal):
    # Initialize Q-values for each edge
    Q = {}
    for u, v in graph.edges:
        Q[(u, v)] = 0.0

    nodes = list(graph.nodes)

    for episode in range(episodes):
        state = random.choice(nodes)
        for step in range(max_steps):
            # Get all possible actions (neighbors)
            actions = list(graph.successors(state))
            if not actions:
                break

            # Choose action using epsilon-greedy
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                action = max(actions, key=lambda a: Q[(state, a)])

            # Get reward and next state
            next_state = action
            reward = -graph[state][action]['weight']
            
            # If reached goal
            if next_state == goal:
                Q[(state, action)] = Q[(state, action)] + alpha * (reward - Q[(state, action)])
                break

            # Update Q-value
            next_actions = list(graph.successors(next_state))
            if next_actions:
                max_q_next = max([Q[(next_state, a)] for a in next_actions])
            else:
                max_q_next = 0.0

            Q[(state, action)] = Q[(state, action)] + alpha * (reward + gamma * max_q_next - Q[(state, action)])
            state = next_state

        # Optionally print every N episodes
        if episode % 100 == 0:
            print(f"Episode {episode}")

    # Derive state values from Q-values
    V = {}
    for state in graph.nodes:
        actions = list(graph.successors(state))
        if actions:
            V[state] = max(Q[(state, a)] for a in actions)
        else:
            V[state] = 0.0

    print("Learned state values:", V)

# Replace value iteration with Q-learning
g2_length = 20
G2 = nx.DiGraph()
G2.add_nodes_from([i for i in range(g2_length)])

for i in range(g2_length-1):
    G2.add_edge(i, i+1, weight=1)
for i in range(1, g2_length):
    G2.add_edge(i, i-1, weight=1)

q_learning(G2, g2_length-1)
