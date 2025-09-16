#! /usr/bin/env python3

# Robot Planning Python Library (RPPL)
# Copyright (c) 2021 Alexander J. LaValle. All rights reserved.
# This software is distributed under the simplified BSD license.

from networkx.classes.function import get_node_attributes, set_node_attributes
import pygame, time
from pygame.locals import *
import networkx as nx
from tkinter import *
from rppl_globals import *
from rppl_util import *
from ast import literal_eval
from collections import defaultdict, Counter

dims = 20 # number of samples per axis
radius = 1 # neightborhood radius (1 = four-neighbors)
exnum = 2 # example number
xmax = 800 # force a square environment

screen = pygame.display.set_mode([xmax,ymax])
use_qlearning = False
pygame.display.set_caption('Grid Planner')

# value iteration constants
failure_cost = 1.0E30
max_valits = 1000


def find_closest_node(mpos,nodes):
    a = [dist2(mpos, nodes[0]['point']),0]
    for i in nodes:
        if i > 0:
            b = [dist2(mpos, nodes[i]['point']),i]
            if a[0] > b[0]:
                a = [dist2(mpos, nodes[i]['point']),i]
    return a[1]

def generate_neighborhood_indices(radius):
    neighbors = []
    k = int(radius+2)
    for i in range(-k,k):
        for j in range(-k,k):
            if 0 < vlen([i,j]) <= radius:
                neighbors.append([i,j])
    return neighbors

# Compute solution path from Q-table
def q_learning_dc_path(graph, init, goal, episodes=10000, max_steps=5000, initial_epsilon=1):
    # Add an edge from the state to itself with 0 weight (stay cost)
    for n in graph.nodes:
        graph.add_edge(n, n, weight=0.0)
    
    # Populate Q-table with zeros - not a proper Q-table, since it's technically [state,state]
    Q = {}
    for n in graph.nodes:
        for m in graph.neighbors(n):
            Q[(n, m)] = None # don't care value
    Q[(goal, goal)] = 0.0 # termination action

    
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
            if random.random() < epsilon:
                action = random.choice(neighbors)
            else:
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
            if state == goal:
                break
    
    #print(Q)
    # Extract path from learned Q-values
    path = [init]
    current = init
    visited = set()
    while current != goal:
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
        graph.remove_edge(n, n) # clean up self-loops TODO: test!

    return path if current == goal else []

# Compute solution path from Q-table
def q_learning_path(graph, init, goal, episodes=1000, max_steps=500, alpha=1, gamma=1, initial_epsilon=1):
    # Add an edge from the goal state to itself with 0 weight (termination action)
    graph.add_edge(goal, goal, weight=0.0)
    
    # Populate Q-table with zeros - not a proper Q-table, since it's technically [state,state]
    Q = {}
    for n in graph.nodes:
        for m in graph.neighbors(n):
            Q[(n, m)] = 1.0E7 # TODO: investigate value, very sensitive to it
    
    # Epsilon decay
    epsilon = 0.1 # = initial_epsilon
    teleporting = False # TODO if true
    # decay_rate = 0.9999

    # Convergence criterion
    convergence_threshold = 1e-4

    # Iteratively update Q-table values
    for episode in range(episodes):
        state = init
        max_delta = 0
        
        for _ in range(max_steps):
            if(teleporting):
                if random.random() < 0.01:
                    state = random.choice(list(graph.nodes))

                neighbors = list(graph.neighbors(state))
                while not neighbors:
                    print("No neighbors found.")
                    state = random.choice(list(graph.nodes))
                    print(state)
                    neighbors = list(graph.neighbors(state))
            else:
                neighbors = list(graph.neighbors(state))
                if not neighbors:
                    print("No neighbors found.")
                    break

            if random.random() < epsilon:
                action = random.choice(neighbors)
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
            if state == goal:
                break
        
        # If the values in the Q-table haven't changed by a lot, some sort of soft convergence has been reached
        # if max_delta < convergence_threshold:
        #     print(f"Q-learning converged at episode {episode}")
        #     break
        
        # for epsilon decay
        # epsilon = max(0.05, initial_epsilon * decay_rate**episode)

    #print(Q)
    # Extract path from learned Q-values
    path = [init]
    current = init
    visited = set()
    while current != goal:
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

    graph.remove_edge(goal, goal) # clean up self-loop at goal TODO: test!
    return path if current == goal else []

# Compute the stationary cost-to-go function and return a solution path.
def valit_path(graph, init, goal):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
    set_node_attributes(graph, {goal:0.0}, 'value') # This is the termination action, although it is not an action to speak of.
    
    # main loop
    i = 0
    max_change = failure_cost
    while i < max_valits and max_change > 0.0:
        max_change = 0.0
        for m in graph.nodes:
            best_cost = failure_cost
            best_n = m
            for n in graph.neighbors(m):
                step_cost = graph.get_edge_data(n,m)['weight']
                cost = graph.nodes[n]['value'] + step_cost
                if cost < best_cost:
                    best_cost = cost
                    best_n = n
            stay_cost = graph.nodes[m]['value']
            if best_cost < stay_cost:
                if stay_cost - best_cost > max_change:
                    max_change = stay_cost - best_cost
                set_node_attributes(graph, {m:best_cost}, 'value')
                set_node_attributes(graph, {m:best_n}, 'next')
        i += 1
    path = []
    if graph.nodes[init]['value'] < failure_cost:
        path.append(init)
        goal_reached = False
        current_node = init
        while not goal_reached:
            nn = graph.nodes[current_node]['next']
            path.append(nn)
            current_node = nn
            if nn == goal:
                goal_reached = True
    print("Stages: " + str(i))
    return path

def random_valit_path(graph, init, goal):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
        set_node_attributes(graph, {n:False}, 'updated') # has the node been visited
    set_node_attributes(graph, {goal:0.0}, 'value') # This is the termination action, although it is not an action to speak of.
    
    # main loop
    i = 0
    nodes_updated = nx.get_node_attributes(graph, "updated")
    while i < max_valits:
        for m in graph.nodes:
            if not list(graph.neighbors(m)):
                continue 
            n = random.choice(list(graph.neighbors(m)))
            step_cost = graph.get_edge_data(n,m)['weight']
            cost = graph.nodes[n]['value'] + step_cost

            best_cost = failure_cost
            best_n = m
            if cost < best_cost:
                best_cost = cost
                best_n = n
            stay_cost = graph.nodes[m]['value']
            if best_cost < stay_cost:
                current = graph.nodes[m].get('updated', None)
                set_node_attributes(graph, {m:not current}, 'updated')
                set_node_attributes(graph, {m:best_cost}, 'value')
                set_node_attributes(graph, {m:best_n}, 'next')
        if i != 0 and i % 50 == 0: # TODO: Can be optimized with statistics? If the node has 4 actions how many times do we need to visit it so that we are sure that all actions have been tried?
            new_nodes_updated =  nx.get_node_attributes(graph, "updated")
            if nodes_updated == new_nodes_updated:
                break
            nodes_updated = new_nodes_updated
        i += 1
    path = []
    if graph.nodes[init]['value'] < failure_cost:
        path.append(init)
        goal_reached = False
        current_node = init
        while not goal_reached:
            nn = graph.nodes[current_node]['next']
            path.append(nn)
            current_node = nn
            if nn == goal:
                goal_reached = True
    print("Stages: " + str(i))
    return path

def prob_valit(graph, init, goal, prob):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
    set_node_attributes(graph, {goal:0.0}, 'value')
    
    # main loop
    i = 0
    max_change = failure_cost
    while i < max_valits and max_change > 1e-6:
        max_change = 0.0
        for m in graph.nodes:
            best_cost = failure_cost
            best_n = m
            
            for n in graph.neighbors(m):
                cost = graph.nodes[n]['value'] * prob # multiply by chosen probability

                # # Get edge count minus first action
                k = len(list(graph.neighbors(m))) - 1
                # sum up expected costs and multiply by updated chosen probability
                for o in graph.neighbors(m):
                    if o != n: #make sure that the current node is not taken into account
                        cost = cost + graph.nodes[o]['value'] * (1-prob) / k

                # add weight to summed up cost    
                step_cost = graph.get_edge_data(n,m)['weight']
                cost = cost + step_cost

                if cost < best_cost:
                    best_cost = cost
                    best_n = n
            stay_cost = graph.nodes[m]['value']
            if best_cost < stay_cost:
                if stay_cost - best_cost > max_change:
                    max_change = stay_cost - best_cost
                set_node_attributes(graph, {m:best_cost}, 'value')
                set_node_attributes(graph, {m:best_n}, 'next')
        i += 1
    path = []
    #TODO: currently I apply probability to the value iteration, but there is no probability in the path extraction. Since the actual graph is deterministic, what am I doing then?
    if graph.nodes[init]['value'] < failure_cost:
        path.append(init)
        goal_reached = False
        current_node = init
        while not goal_reached:
            nn = graph.nodes[current_node]['next']
            path.append(nn)
            current_node = nn
            if nn == goal:
                goal_reached = True
    print("Stages: " + str(i))
    return path

# This corresponds to GUI button 'Draw' that runs the example.
def Draw():
    obstacles = literal_eval(problines[exnum*3])
    initial = literal_eval(problines[exnum*3+1])
    goal = literal_eval(problines[exnum*3+2])

    global G
    arr = [[0 for i in range(dims)] for j in range(dims)]
    actions = generate_neighborhood_indices(radius)
    G = nx.Graph()
    pygame.init()
    screen.fill(black)
    incrementy = 0
    i = 0
    length = 0
    
    # construct grid
    for y in range(dims):
        if y > 0:
            incrementy += ymax/dims + (ymax/dims)/(dims-1)
        incrementx = 0
        for x in range(dims):
            G.add_node(i, point=(incrementx,incrementy))
            incrementx += xmax/dims + (xmax/dims)/(dims-1)
            arr[y][x] = i
            i += 1
    for x in range(dims):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                quit()
        for y in range(dims):
            for u in actions:
                if (0 <= x + u[0] <= dims-1 and 0 <= y + u[1] <= dims-1 
                    and safe(G.nodes[arr[y][x]]['point'],G.nodes[arr[y+u[1]][x+u[0]]]['point'],obstacles) 
                    and not G.has_edge(arr[y][x],arr[y+u[1]][x+u[0]])
                    ):
                    G.add_edge(arr[y][x],arr[y+u[1]][x+u[0]],weight=dist2(G.nodes[arr[y][x]]['point'],G.nodes[arr[y+u[1]][x+u[0]]]['point']))
    # The next three lines delete the obstacle nodes (optional).
    #for i in range(len(G.nodes)):
    #        if point_inside_discs(G.nodes[i]['point'],obstacles):
    #            G.remove_node(i)
    draw_discs(obstacles, screen)
    draw_graph_edges(G, screen)
    p1index = find_closest_node(initial,G.nodes)
    p2index = find_closest_node(goal,G.nodes)
    # Print edge cost/weight
    # for (u,v,c) in G.edges().data():
    #     print("Edge (" + str(u) + ", " + str(v) +"): " + str(c))

    if nx.has_path(G,p1index,p2index):
        t = time.time()
        if use_qlearning:
            path = q_learning_dc_path(G, p1index, p2index)
            print('Q-learning:   time elapsed:     ' + str(time.time() - t) + ' seconds')
        else:
            path = prob_valit(G,p1index,p2index, 0.8)
            print('value iteration: time elapsed: ' + str(time.time() - t) + ' seconds')
        print("Shortest path: " + str(len(path)))
        for l in range(len(path)):
            if l > 0:
                pygame.draw.line(screen,green,G.nodes[path[l]]['point'],G.nodes[path[l-1]]['point'],5)
                length += G.get_edge_data(path[l],path[l-1])['weight']
        pygame.display.set_caption('Grid Planner, Euclidean Distance: ' +str(length))
    else:
        print('Path not found')
        pygame.display.set_caption('Grid Planner')
    pygame.draw.circle(screen,green,initial,10)
    pygame.draw.circle(screen,red,goal,10)
    pygame.display.update()

# get example list
problem = open('problem_circles.txt')
problines = problem.readlines()
problem.close()
num_of_ex = len(problines)/3
# The rest is for the GUI.

def SwitchType():
    global use_qlearning
    if use_qlearning:
        use_qlearning = False
        print("Switched to Value Iteration")
    else:
        use_qlearning = True
        print("Switched to Q-learning")
    Draw()

def SetDims(val):
    global dims
    dims = int(val)

def SetRadius(val):
    global radius
    radius = float(val)

def SetExNum(val):
    global exnum
    exnum = int(val) - 1

def Exit():
    master.destroy()

def SaveData():
    data = open('valit_data.txt','w')
    data.write(str(get_node_attributes(G, 'value')) + '\n' + str(get_node_attributes(G, 'point')) + '\n' + str(dims))
    data.close()

master = Tk()
master.title('Grid-Planner GUI')
master.geometry("630x80")

m1 = PanedWindow(master,borderwidth=10,bg="#000000")
m1.pack(fill = BOTH,expand = 1)

exitbutton = Button(m1, text='     Quit     ',command=Exit,fg='red')
m1.add(exitbutton)

savebutton = Button(m1, text='     Save     ',command=SaveData,fg='blue')
m1.add(savebutton)

switchbutton = Button(m1, text='   Change   \n   Planner   ',command=SwitchType,fg='brown')
m1.add(switchbutton)

drawbutton = Button(m1, text='     Draw     ',command=Draw,fg='green')
m1.add(drawbutton)

dimsscale = Scale(m1, orient = HORIZONTAL, from_=2, to=200, resolution=1, command=SetDims, label='Resolution (n * n)')
dimsscale.set(dims)
m1.add(dimsscale)

radscale = Scale(m1, orient = HORIZONTAL, from_=1, to=10, resolution=0.1, command=SetRadius, label='Neighbor Radius')
radscale.set(radius)
m1.add(radscale)

exscale = Scale(m1, orient = HORIZONTAL, from_=1, to=num_of_ex, resolution=1, command=SetExNum, label='Example Number')
exscale.set(int(exnum))
m1.add(exscale)

master.mainloop()