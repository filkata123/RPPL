import random
from prob_model import probability_model

# Compute solution path from Q-table
def q_learning_stochastic_path(graph, init, goal_region, episodes=1000, max_steps=500, alpha=1, gamma=1, initial_epsilon=1):
    # Add an edge from the goal state to itself with 0 weight (termination action)
    for goal in goal_region:
        graph.add_edge(goal, goal, weight=0.0)
    
    # Populate Q-table with zeros - not a proper Q-table, since it's technically [state,state]
    Q = {}
    for n in graph.nodes:
        for m in graph.neighbors(n):
            Q[(n, m)] = -1.0E7 # TODO: investigate value, very sensitive to it
    
    # Epsilon decay
    epsilon = 0.1 # = initial_epsilon

    # Convergence criterion
    convergence_threshold = 1e-4

    # Iteratively update Q-table values
    for episode in range(episodes):
        state = init
        max_delta = 0
        
        for _ in range(max_steps):
            neighbors = list(graph.neighbors(state))
            if not neighbors:
                print("No neighbors found.")
                break

            if random.random() < epsilon:
                action = random.choice(neighbors) # in this case, the random action takes into account the stochasticity of the environment
            else:
                action = min(neighbors, key=lambda a: Q.get((state, a), 0.0))
            cost = graph[state][action]['weight']
            next_state = action

            next_neighbors = list(graph.neighbors(next_state))
            min_q_next = min([Q.get((next_state, a), 1.0E10) for a in next_neighbors]) if next_neighbors else 0

            old_q = Q[(state, action)]
            Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha * (cost + gamma * min_q_next)

            # Track maximum absolute change in Q-values per episodes
            delta = abs(Q[(state, action)] - old_q)
            if delta > max_delta:
                max_delta = delta

            state = next_state
            if state in goal_region:
                break
        
        # If the values in the Q-table haven't changed by a lot, some sort of soft convergence has been reached
        if max_delta < convergence_threshold:
            #print(f"Q-learning converged at episode {episode}")
            break

    # Extract path from learned Q-values
    path = [init]
    current = init
    while current not in goal_region:
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            print("No neighbors found.")
            break
        desired = min(neighbors, key=lambda a: Q.get((current, a), float('inf')))
        prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(current)))) # get probabilities
        choice = random.random()
        if choice <= prob_success:
            next_node = desired # successful transition
        elif choice > prob_success and choice <= prob_success + prob_stay:
            next_node = current # stay
        else:
            current_range = prob_success + prob_stay
            for o in graph.neighbors(current):
                if o != desired: # make sure that the desired node is not taken into account
                    if choice > current_range and choice <= current_range + prob_other:
                        next_node = o
                        break
                    else: current_range += prob_other
        path.append(next_node)
        current = next_node
    for goal in goal_region:
        graph.remove_edge(goal, goal) # clean up self-loop at goal
    return path if current in goal_region else []

# Compute solution path from Q-table
def q_learning_dc_path(graph, init, goal_region, episodes=15000, max_steps=5000, initial_epsilon=1):
    # Add an edge from the state to itself with 0 weight (stay cost)
    for n in graph.nodes:
        graph.add_edge(n, n, weight=0.0)
    
    # Populate Q-table with zeros - not a proper Q-table, since it's technically [state,state]
    Q = {}
    for n in graph.nodes:
        for m in graph.neighbors(n):
            Q[(n, m)] = None # don't care value
    for goal in goal_region:
        Q[(goal, goal)] = 0.0 # goal state attractor. Works only in DC q-learning here since it's better than all other values.

    
    # Epsilon decay
    epsilon = 0.1 # = initial_epsilon

    # Iteratively update Q-table values
    for episode in range(episodes):
        state = init
        
        for _ in range(max_steps):
            neighbors = list(graph.neighbors(state))
            if not neighbors:
                print("No neighbors found.")
                break
            
            #TODO: Use digits of Pi in base n, where n = |actions| (sagemath)
            # if random.random() < epsilon:
            #     action = random.choice(neighbors)
            # else:
            valid_neighbors = [a for a in neighbors if Q.get((state, a)) is not None]
            action = min(valid_neighbors, key=lambda a: Q.get((state, a), 0.0)) if valid_neighbors else random.choice(neighbors)

            cost = graph[state][action]['weight']
            next_state = action

            next_neighbors = list(graph.neighbors(next_state))
            valid_q_values = [Q[(next_state, a)] for a in next_neighbors if Q[(next_state, a)] is not None]
            min_q_next = min(valid_q_values) if valid_q_values else None

            if min_q_next is not None:
                Q[(state, action)] = cost + min_q_next

            state = next_state
            if state in goal_region:
                break
    
    #print(Q)
    # Extract path from learned Q-values
    path = [init]
    current = init
    visited = set()
    while current not in goal_region:
        visited.add(current)
        neighbors = list(graph.neighbors(current))
        valid_neighbors = [a for a in neighbors if Q.get((current, a)) is not None]
        if not valid_neighbors:
            print("No valid neighbors found in Q-table. No path to goal available.")
            break
        next_node = min(valid_neighbors, key=lambda a: Q.get((current, a), float('inf')))
        if next_node in visited:
            print("Loop detected in Q-table. No path to goal available.")
            break # avoid loops
        path.append(next_node)
        current = next_node

    for n in graph.nodes:
        graph.remove_edge(n, n) # clean up self-loops

    return path if current in goal_region else []

# Compute solution path from Q-table
def q_learning_path(graph, init, goal_region, episodes=1000, max_steps=500, alpha=1, gamma=1, initial_epsilon=1):
    # Add an edge from the goal state to itself with 0 weight (termination action)
    for goal in goal_region:
        graph.add_edge(goal, goal, weight=0.0)
    
    # Populate Q-table with zeros - not a proper Q-table, since it's technically [state,state]
    Q = {}
    for n in graph.nodes:
        for m in graph.neighbors(n):
            Q[(n, m)] = 0 # TODO: investigate value, very sensitive to it
    
    # Epsilon decay
    epsilon = 0.1 # = initial_epsilon

    # Convergence criterion
    convergence_threshold = 1e-4

    # Iteratively update Q-table values
    for episode in range(episodes):
        state = init
        max_delta = 0
        
        for _ in range(max_steps):
            neighbors = list(graph.neighbors(state))
            if not neighbors:
                print("No neighbors found.")
                break

            if random.random() < epsilon:
                action = random.choice(neighbors)
            else:
                action = min(neighbors, key=lambda a: Q.get((state, a), 0.0))
            #action = min(neighbors, key=lambda a: Q.get((state, a), 0.0))

            cost = graph[state][action]['weight']
            next_state = action

            next_neighbors = list(graph.neighbors(next_state))
            min_q_next = min([Q.get((next_state, a), 1.0E10) for a in next_neighbors]) if next_neighbors else 0

            old_q = Q[(state, action)]
            Q[(state, action)] = (1-alpha)*Q[(state, action)] + alpha * (cost + gamma * min_q_next)

            # Track maximum absolute change in Q-values per episodes
            delta = abs(Q[(state, action)] - old_q)
            if delta > max_delta:
                max_delta = delta

            state = next_state
            if state in goal_region:
                break
        
        # If the values in the Q-table haven't changed by a lot, some sort of soft convergence has been reached
        # if max_delta < convergence_threshold:
        #     print(f"Q-learning converged at episode {episode}")
        #     break

    #print(Q)
    # Extract path from learned Q-values
    path = [init]
    current = init
    visited = set()
    while current not in goal_region:
        visited.add(current)
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        next_node = min(neighbors, key=lambda a: Q.get((current, a), float('inf')))
        if next_node in visited:
            print("Loop detected in Q-table. No path to goal available.")
            break # avoid loops
        path.append(next_node)
        current = next_node
    for goal in goal_region:
        graph.remove_edge(goal, goal) # clean up self-loop at goal
    return path if current in goal_region else []