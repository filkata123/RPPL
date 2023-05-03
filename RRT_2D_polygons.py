#! /usr/bin/env python3

# Robot Planning Python Library (RPPL)
# Copyright (c) 2021 Alexander J. LaValle. All rights reserved.
# This software is distributed under the simplified BSD license.

from networkx.algorithms.shortest_paths.generic import shortest_path
import pygame, time, random
from pygame.locals import *
import networkx as nx
from rppl_util import *
from rppl_globals import *
from polygon_triangulate import polygon_triangulate
from ast import literal_eval

bidirectional = True
bias = 6
stepsize = 5

Open = True
stepping = True
restart = True


def add_next_node(rp,closest,g):
    d = dist2(rp,g.nodes[closest]['point'])
    if d == 0:
        d = 1.0E-20
    diff = stepsize / d
    newpoint = ((rp[0] - g.nodes[closest]['point'][0]) * diff + g.nodes[closest]['point'][0],(rp[1] - g.nodes[closest]['point'][1]) * diff + g.nodes[closest]['point'][1])
    return newpoint

def step_to_config(t,q):
    stepping = True
    closest = find_closest_node(q, t.nodes)
    t.add_node(len(t.nodes),point=add_next_node(q,closest,t))
    if not point_in_triangles(t.nodes[len(t.nodes)-1]['point'],tlist):
        t.add_edge(len(t.nodes)-1,closest)
        if t == I:
            pygame.draw.line(screen,white,t.nodes[closest]['point'],t.nodes[len(t.nodes)-1]['point'],2)
        else:
            pygame.draw.line(screen,red,t.nodes[closest]['point'],t.nodes[len(t.nodes)-1]['point'],2)
        while stepping:
            for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        quit()
            t.add_node(len(t.nodes),point=add_next_node(q,len(t.nodes)-1,t))
            if not point_in_triangles(t.nodes[len(t.nodes)-1]['point'],tlist):
                t.add_edge(len(t.nodes)-1,len(t.nodes)-2)
                if t == I:
                    pygame.draw.line(screen,white,t.nodes[len(t.nodes)-2]['point'],t.nodes[len(t.nodes)-1]['point'],2)
                else:
                    pygame.draw.line(screen,red,t.nodes[len(t.nodes)-2]['point'],t.nodes[len(t.nodes)-1]['point'],2)
                if dist2(t.nodes[len(t.nodes)-1]['point'],q) <= stepsize:
                    stepping = False
            else:
                t.remove_node(len(t.nodes)-1)
                stepping = False
    else:
        t.remove_node(len(t.nodes)-1)
    pygame.display.update()

def find_closest_node(mpos,nodes):
    a = [dist2(mpos, nodes[0]['point']),0]
    for i in nodes:
        if i > 0:
            b = [dist2(mpos, nodes[i]['point']),i]
            if a[0] > b[0]:
                a = [dist2(mpos, nodes[i]['point']),i]
    return a[1]

problem = open('problem_polygonal.txt')
problines = problem.readlines()
problem.close()
obstacles = literal_eval(problines[0])
initial = literal_eval(problines[1])
goal = literal_eval(problines[2])
tilist = []
tlist = []

for o in obstacles:
    tilist.append(polygon_triangulate(len(o), [i[0] for i in o], [j[1] for j in o]))

# convert triangulation output to list of triangles
for o in range(len(obstacles)):
        for t in tilist[o]:
            tlist.append([obstacles[o][t[0]],obstacles[o][t[1]],obstacles[o][t[2]]])

print('\n*CONTROLS*\nR = Re-Run\nSPACEBAR = Alternate RRT type\nESC = Stop all\n')

while Open:
    I = nx.Graph()
    G = nx.Graph()
    I.add_node(0, point=initial)
    G.add_node(0, point=goal)
    pygame.init()
    screen = pygame.display.set_mode([xmax,ymax])
    pygame.display.set_caption('RRT Plan')
    screen.fill(black)
    pygame.draw.circle(screen,green,initial,5)
    pygame.draw.circle(screen,red,goal,5)
    draw_polys(tlist,screen)
    pygame.display.update()
    pstat = 0
    restart = False
    i = 0
    t = time.time()
    step_to_config(I,goal) # Check for straight line

    if bidirectional:
        print("Bidirectional RRT")
        while dist2(I.nodes[len(I.nodes)-1]['point'],G.nodes[len(G.nodes)-1]['point']) > stepsize:
            rc = [random.randint(0,xmax),random.randint(0,ymax)]
            step_to_config(I,rc)
            step_to_config(G,I.nodes[len(I.nodes)-1]['point'])
            if dist2(I.nodes[len(I.nodes)-1]['point'],G.nodes[len(G.nodes)-1]['point']) > stepsize:
                rc = [random.randint(0,xmax),random.randint(0,ymax)]
                step_to_config(G,rc)
                step_to_config(I,G.nodes[len(G.nodes)-1]['point'])
        
    else:
        print("Goal-biased RRT")
        while dist2(I.nodes[len(I.nodes)-1]['point'], goal) > stepsize:
            if i % bias == 0:
                rc = goal
            else:
                rc = [random.randint(0,xmax),random.randint(0,ymax)]
            step_to_config(I,rc)
            i += 1

    print('planning time: ' + str(time.time() - t) + ' seconds')
    pygame.display.set_caption('RRT Plan, Total nodes: ' + str(len(I.nodes)+len(G.nodes)))
    if bidirectional:
        path = shortest_path(I, source=0, target=len(I.nodes)-1)
        for p in range(len(path)-1):
            pygame.draw.line(screen,green,I.nodes[path[p]]['point'],I.nodes[path[p+1]]['point'],3)
        path = shortest_path(G, source=0, target=len(G.nodes)-1)
        for p in range(len(path)-1):
            pygame.draw.line(screen,green,G.nodes[path[p]]['point'],G.nodes[path[p+1]]['point'],3)
    else:
        path = shortest_path(I, source=0, target=len(I.nodes)-1)
        for p in range(len(path)-1):
            pygame.draw.line(screen,green,I.nodes[path[p]]['point'],I.nodes[path[p+1]]['point'],3)

    while restart == False:
        time.sleep(0.02)
        pygame.display.update()
        mpos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if bidirectional:
                    bidirectional = False
                else:
                    bidirectional = True
                restart = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                restart = True