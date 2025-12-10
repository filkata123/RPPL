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
from q_learning_functions import *
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


# Compute the stationary cost-to-go function and return a solution path.
def valit_path(graph, init, goal_region):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
    for goal in goal_region:
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
            if nn in goal_region:
                goal_reached = True
    print("Stages: " + str(i))
    return path

def random_valit_path(graph, init, goal_region, epsilon_greedy = False):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
        set_node_attributes(graph, {n:False}, 'updated') # has the node been visited
    for goal in goal_region:
        set_node_attributes(graph, {goal:0.0}, 'value') # This is the termination action, although it is not an action to speak of.
    
    # main loop
    i = 0
    nodes_updated = nx.get_node_attributes(graph, "updated")
    while i < max_valits:
        for m in graph.nodes:
            if not list(graph.neighbors(m)):
                continue 
            if not epsilon_greedy:
                chosen_n = random.choice(list(graph.neighbors(m)))
            else:
                if random.random() < 0.1:
                    chosen_n = random.choice(list(graph.neighbors(m)))
                else:
                    chosen_n = min(
                        graph.neighbors(m),
                        key=lambda n: graph.nodes[n]["value"] + graph.get_edge_data(n,m)['weight']
                    )
            step_cost = graph.get_edge_data(chosen_n,m)['weight']
            cost = graph.nodes[chosen_n]['value'] + step_cost

            best_cost = failure_cost
            best_n = m
            if cost < best_cost:
                best_cost = cost
                best_n = chosen_n
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
            if nn in goal_region:
                goal_reached = True
    print("Stages: " + str(i))
    return path

def prob_valit(graph, init, goal_region):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
    for goal in goal_region:
        set_node_attributes(graph, {goal:0.0}, 'value')
    
    # main loop
    i = 0
    max_change = failure_cost
    while i < max_valits and max_change > 0.0:
        max_change = 0.0
        for m in graph.nodes:
            best_cost = failure_cost
            best_n = m
            
            for n in graph.neighbors(m):
                # Get edge count
                prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(m))))
                
                cost = graph.nodes[n]['value'] * prob_success # multiply by success probability
                cost = cost + graph.nodes[m]['value'] * prob_stay # multiply by stay probability

                # sum up expected costs and multiply by updated chosen probability
                for o in graph.neighbors(m):
                    if o != n: #make sure that the current node is not taken into account
                        cost = cost + graph.nodes[o]['value'] * prob_other

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
    
    #print(nx.get_node_attributes(graph, 'value'))
    path = []
    if graph.nodes[init]['value'] < failure_cost:
        path.append(init)
        goal_reached = False
        current_node = init
        while not goal_reached:
            desired = graph.nodes[current_node]['next'] # select our desired node
            prob_success, prob_stay, prob_other = probability_model(len(list(graph.neighbors(current_node)))) # get probabilities
            choice = random.random()
            if choice <= prob_success:
                nn = desired # successful transition
            elif choice > prob_success and choice <= prob_success + prob_stay:
                nn = current_node # stay
            else:
                current_range = prob_success + prob_stay
                for o in graph.neighbors(current_node):
                    if o != desired: # make sure that the desired node is not taken into account
                        if choice > current_range and choice <= current_range + prob_other:
                            nn = o
                            break
                        else: current_range += prob_other
            path.append(nn)
            current_node = nn
            if nn in goal_region:
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

    # Use a radius parameter to find the neighbors that will define the goal region
    goal_radius = 0
    goal_indices = list(nx.single_source_shortest_path_length(G, p2index, cutoff=goal_radius).keys())

    #Since the graph is undirected, this is equivalent to checking if there is a path from p1index to any of the goal_indices
    if nx.has_path(G,p1index,p2index):
        t = time.time()
        if use_qlearning:
            path = q_learning_dc_path(G, p1index, goal_indices)
            print('Q-learning:   time elapsed:     ' + str(time.time() - t) + ' seconds')
        else:
            path = random_valit_path(G,p1index,goal_indices, True)
            print('value iteration: time elapsed: ' + str(time.time() - t) + ' seconds')
        print("Shortest path: " + str(len(path)))
        for l in range(len(path)):
            if l > 0:
                pygame.draw.line(screen,green,G.nodes[path[l]]['point'],G.nodes[path[l-1]]['point'],5)
                if G.get_edge_data(path[l],path[l-1]) is not None: # When there are loops, there is no weight in some cases
                    length += G.get_edge_data(path[l],path[l-1])['weight']
        pygame.display.set_caption('Grid Planner, Euclidean Distance: ' +str(length))
    else:
        print('Path not found')
        pygame.display.set_caption('Grid Planner')
    pygame.draw.circle(screen,green,G.nodes[p1index]['point'],10)
    for g in goal_indices:
        pygame.draw.circle(screen,red,G.nodes[g]['point'],10)
    # Old implementation of visualization
    #pygame.draw.circle(screen,green,initial,10)
    #pygame.draw.circle(screen,red,goal,10)
    pygame.display.update()
    #pygame.image.save(screen, "screenshot.png")

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