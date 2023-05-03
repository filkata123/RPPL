#! /usr/bin/env python3

# Robot Planning Python Library (RPPL)
# Copyright (c) 2021 Alexander J. LaValle. All rights reserved.
# This software is distributed under the simplified BSD license.

import networkx as nx
from networkx.classes.function import get_node_attributes, set_node_attributes


# value iteration constants
failure_cost = 10000#1.0E30
max_valits = 1000


def valit(graph, goal):
    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
    set_node_attributes(graph, {goal:0.0}, 'value')
    print('Iteration: 0  ', get_node_attributes(graph, 'value'))
    
    # main loop
    i = 0
    max_change = failure_cost
    while i < max_valits and max_change > 0.0:
        update_list = []
        max_change = 0.0
        for m in graph.nodes:
            best_cost = failure_cost
            for n in graph.neighbors(m):
                step_cost = graph.get_edge_data(m,n)['weight']
                cost = graph.nodes[n]['value'] + step_cost
                if cost < best_cost:
                    best_cost = cost
            stay_cost = graph.nodes[m]['value']
            if best_cost < stay_cost:
                if stay_cost - best_cost > max_change:
                    max_change = stay_cost - best_cost
                update_list.append([m,best_cost])
        for u in update_list:
            set_node_attributes(graph, {u[0]:u[1]}, 'value')
        i += 1
        print('Iteration: ' +str(i), ' ', get_node_attributes(graph, 'value'))
        
def prob_valit(graph, goal, p):

    # initialize values
    for n in graph.nodes:
        set_node_attributes(graph, {n:failure_cost}, 'value')
    set_node_attributes(graph, {goal:0.0}, 'value')
    print('Iteration: 0  ', get_node_attributes(graph, 'value'))
    
    # main loop
    i = 0
    max_change = failure_cost
    while i < max_valits and max_change > 0.0:
        update_list = []
        max_change = 0.0
        for m in graph.nodes:
            best_cost = failure_cost

            #ensure probability is one when only one edge is present
            prob = p
            if graph.in_degree(m) == 1:
                prob = 1
            
            for n in graph.neighbors(m):
                    
                cost = graph.nodes[n]['value'] * prob # multiply by chosen probability

                # Get edge count minus first action
                k = len(list(graph.neighbors(m))) - 1
                # sum up expected costs and multiply by updated chosen probability
                for o in graph.neighbors(m):
                    if o != n: #make sure that the current node is not taken into account
                        cost = cost + graph.nodes[o]['value'] * (1-prob) / k

                # add weight to summed up cost    
                step_cost = graph.get_edge_data(m,n)['weight']
                cost = cost + step_cost

                if cost < best_cost:
                    best_cost = cost
            stay_cost = graph.nodes[m]['value']
            if best_cost < stay_cost:
                if stay_cost - best_cost > max_change:
                    max_change = stay_cost - best_cost
                update_list.append([m,best_cost])
        for u in update_list:
            set_node_attributes(graph, {u[0]:u[1]}, 'value')
        i += 1
        print('Iteration: ' +str(i), ' ', get_node_attributes(graph, 'value'))

# First, create the graph.
# This graph is from Figure 2.8 from Planning Algorithms, 2006 (http://lavalle.pl/planning/).
G = nx.DiGraph()
G.add_nodes_from([0, 1, 2, 3, 4])
G.add_edge(0, 1, weight=2)
G.add_edge(1, 0, weight=1)
G.add_edge(1, 2, weight=4)
G.add_edge(2, 3, weight=3)
G.add_edge(2, 4, weight=7)
G.add_edge(3, 2, weight=1)
G.add_edge(3, 3, weight=1)
G.add_edge(3, 4, weight=1)

#This example is a linear bidirectional graph of length g2_length.
g2_length = 20
G2 = nx.DiGraph()
G2.add_nodes_from([i for i in range(g2_length)])

for i in range(g2_length-1):
    G2.add_edge(i, i+1, weight=1)
for i in range(1, g2_length):
    G2.add_edge(i, i-1, weight=1)


# Compute the optimal cost to go.
#valit(G, 4)
#valit(G2, g2_length-1)
prob_valit(G2, g2_length-1, 0.8)