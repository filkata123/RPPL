#! /usr/bin/env python3

# Robot Planning Python Library (RPPL)
# Copyright (c) 2021 Alexander J. LaValle. All rights reserved.
# This software is distributed under the simplified BSD license.

import pygame, time, random
from pygame.locals import *
import networkx as nx
from networkx import shortest_path
from math import *
from rppl_util import *
from rppl_globals import *


show_rrt_progress = True
bidirectional = True
print_status = False
numlinks = 6 # number of links
numobst = 6  # number of obstacles

links = [1000/numlinks for i in range(numlinks)]
base = [xmax/2,ymax/2]
config = [2*pi-2*pi/numlinks for i in range(numlinks)]
stepsize = 1.0 / numlinks # radians per link
goal = [2*pi/numlinks for i in range(numlinks)]
bias = 2

Open = True


def transform_robot(l, b, q):
    cp = b
    cangle = 0.0
    tl = [b]
    for i in range(len(l)):
        npx = cp[0] + l[i] * cos(cangle + q[i])
        npy = cp[1] + l[i] * sin(cangle + q[i])
        cp = [npx,npy]
        tl.append(cp)
        cangle += q[i]
    return tl

def config_distance(q, r):
    d = 0.0
    for i in range(len(q)):
        d += sqr(min(abs(q[i] - r[i]), 2.0 * pi - abs(q[i] - r[i])))
    return sqrt(d)

def calc_new_config(q,closest,t):
    newconfig = []
    c = t.nodes[closest]['config']
    d = config_distance(q,c)
    diff = stepsize / d
    for i in range(len(q)):
        s = abs(q[i] - c[i])
        di = min(s, 2.0 * pi - s)
        if (q[i] > c[i] and s < pi) or (q[i] < c[i] and s > pi):
            newconfig.append(fix_angle(c[i] + di * diff))
        else:
            newconfig.append(fix_angle(c[i] - di * diff))
    return newconfig

def find_closest_node(rc,nodes):
    a = [config_distance(rc, nodes[0]['config']),0]
    for i in nodes:
        if i > 0:
            b = [config_distance(rc, nodes[i]['config']),i]
            if a[0] > b[0]:
                a = [config_distance(rc, nodes[i]['config']),i]
    return a[1]

def step_to_config(t,q):
    stepping = True
    closest = find_closest_node(q,t.nodes)
    t.add_node(len(t.nodes),config=calc_new_config(q,closest,t))
    if safe_segments(transform_robot(links, base, t.nodes[len(t.nodes)-1]['config']),obstacles):
        t.add_edge(len(t.nodes)-1,closest)
        while stepping:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit()
            if show_rrt_progress:
                screen.fill(black)
                draw_arm(transform_robot(links, base, I.nodes[len(I.nodes)-1]['config']),screen,white)
                draw_discs(obstacles,screen)
                if bidirectional:
                    draw_arm(transform_robot(links, base, G.nodes[len(G.nodes)-1]['config']),screen,red)
                pygame.display.update()
            t.add_node(len(t.nodes),config=calc_new_config(q,len(t.nodes)-1,t))
            if safe_segments(transform_robot(links, base, t.nodes[len(t.nodes)-1]['config']),obstacles):
                t.add_edge(len(t.nodes)-1,len(t.nodes)-2)
                if config_distance(t.nodes[len(t.nodes)-1]['config'],q) <= stepsize:
                    stepping = False
            else:
                t.remove_node(len(t.nodes)-1)
                stepping = False
    else:
        t.remove_node(len(t.nodes)-1)

while Open:
    I = nx.Graph()
    G = nx.Graph()
    I.add_node(0, config=config)
    G.add_node(0, config=goal)
    pygame.init()
    screen = pygame.display.set_mode([xmax,ymax])
    screen.fill(black)
    unobstructed = False
    while not unobstructed:
        obstacles = create_random_discs(numobst,base)
        if safe_segments(transform_robot(links, base, config),obstacles) and safe_segments(transform_robot(links, base, goal),obstacles):
            unobstructed = True
    draw_discs(obstacles,screen)
    draw_arm(transform_robot(links, base, config),screen,white)
    pygame.display.update()
    time.sleep(0.5)
    pstat = 0
    restart = False
    pygame.display.set_caption('RRT Line Segment Robot - Controls: [R] = Run again, [ESC] = Quit')
    t = time.time()
    i = 0

    if bidirectional:
        if print_status: print('check for straight line')
        step_to_config(I,goal) # Check for straight line
        
        while config_distance(I.nodes[len(I.nodes)-1]['config'],G.nodes[len(G.nodes)-1]['config']) > stepsize:
            rc = [random.uniform(0.0, 2 * pi) for i in range(numlinks)]
            if print_status: print('connect initial tree to random point')
            step_to_config(I,rc)
            if print_status: print('connect goal tree to new initial tree vertex')
            step_to_config(G,I.nodes[len(I.nodes)-1]['config'])
            if config_distance(I.nodes[len(I.nodes)-1]['config'],G.nodes[len(G.nodes)-1]['config']) > stepsize:
                rc = [random.uniform(0.0, 2 * pi) for i in range(numlinks)]
                if print_status: print('connect goal tree to random point')
                step_to_config(G,rc)
                if print_status: print('connect initial tree to new goal vertex')
                step_to_config(I,G.nodes[len(G.nodes)-1]['config'])

    else:
        while config_distance(I.nodes[len(I.nodes)-1]['config'], goal) > stepsize:
            if i % bias == 0:
                rc = goal
            else:
                rc = [random.uniform(0.0, 2 * pi) for i in range(numlinks)]
            step_to_config(I,rc)
            i += 1
    
    print('solved! planning time: ' + str(time.time() - t) + ' seconds')
    if show_rrt_progress:
        screen.fill(black)
        draw_discs(obstacles,screen)
        draw_arm(transform_robot(links, base, config),screen,white)
        pygame.display.update()
        time.sleep(0.6)
    
    while not restart:
        screen.fill(black)
        draw_discs(obstacles,screen)
        draw_arm(transform_robot(links, base, config),screen,white)
        pygame.display.update()
        time.sleep(1)
        path = shortest_path(I, source=0, target=len(I.nodes)-1)
        for s in path:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    restart = True
                    break
            if restart:
                break
            time.sleep(0.002)
            screen.fill(black)
            draw_discs(obstacles,screen)
            draw_arm(transform_robot(links, base, I.nodes[s]['config']),screen,white)
            pygame.display.update()
        path = shortest_path(G, source=len(G.nodes)-1, target=0)
        for s in path:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    restart = True
                    break
            if restart:
                break
            time.sleep(0.002)
            screen.fill(black)
            draw_discs(obstacles,screen)
            draw_arm(transform_robot(links, base, G.nodes[s]['config']),screen,white)
            pygame.display.update()
        if not restart:
            time.sleep(1)