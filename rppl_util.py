#! /usr/bin/env python3

# Robot Planning Python Library (RPPL)
# Copyright (c) 2021 Alexander J. LaValle. All rights reserved.
# This software is distributed under the simplified BSD license.

import networkx as nx
from math import sqrt, pi
import time, random, pygame
from rppl_globals import *


def right_turn(a, b, c):
    return (a[0] - b[0]) * (a[1] - c[1]) < (a[1] - b[1]) * (a[0] - c[0])

def left_turn(a, b, c):
    return (a[0] - b[0]) * (a[1] - c[1]) > (a[1] - b[1]) * (a[0] - c[0])

def point_in_triangle(p, t):
    return left_turn(t[0], t[1] ,p) and left_turn(t[1], t[2] ,p) and left_turn(t[2], t[0] ,p)

def point_in_triangles(p, tlist):
    for t in tlist:
        if left_turn(t[0], t[1] ,p) and left_turn(t[1], t[2] ,p) and left_turn(t[2], t[0] ,p):
            return True
    return False

def dist2(p, q):
    a = sqrt(sqr(p[0] - q[0]) + sqr(p[1] - q[1]))
    return a

def sqr(a):
    return a * a

# avoid = coordinates of point to avoid
def create_random_discs(num, avoid):
    i = 0
    o = []
    while i < num:
        o.append([random.randint(150,xmax-150),random.randint(150,ymax-150),random.randrange(50,150,25)])
        if dist2([o[i][0],o[i][1]],avoid) < o[i][2] + 25:
            del o[i]
        else:
            i += 1   
    return o

def pick_random_goal(dlist):
    p = [random.randint(0,xmax),random.randint(0,ymax)]
    while point_inside_discs(p, dlist):
        p = [random.randint(0,xmax),random.randint(0,ymax)]
    return p

def point_inside_discs(p, dlist):
    for d in dlist:
        if dist2(p,d) < d[2]:
            return True
    return False

def points_inside_discs(plist, dlist):
    for d in dlist:
        for p in plist:
            if dist2(p,d) < d[2]:
                return True
    return False

def draw_graph_edges(g,screen):
    for i,j in g.edges:
        pygame.draw.line(screen,white,g.nodes[i]['point'],g.nodes[j]['point'],2)

def draw_discs(dlist,screen):
    for d in dlist:
        pygame.draw.circle(screen,grey,[d[0],d[1]],d[2])

def draw_polys(polys,screen):
    for p in polys:
        pygame.draw.polygon(screen,light_grey,p)

def draw_arm(tlinks,screen,color):
    for i in range(len(tlinks) - 1):
        pygame.draw.line(screen, color, tlinks[i], tlinks[i+1])

def vlen(v):
    return sqrt(v[0]*v[0]+v[1]*v[1])

def detect( A, B, C, r ):#https://stackoverflow.com/questions/26725842/how-to-pick-up-line-segments-which-is-inside-or-intersect-a-circle
    AB = (B[0] - A[0], B[1]-A[1])
    AC = (C[0] - A[0], C[1]-A[1])
    BC = (C[0]-B[0], C[1]-B[1])

    if vlen(BC) < r or vlen(AC) < r: return True
    
    abl = vlen(AB)
    AB_normalized = (AB[0] / abl , AB[1] / abl)
    AP_distance = AC[0] * AB_normalized[0]  +  AC[1] * AB_normalized[1]
    AP = (AP_distance * AB_normalized[0], AP_distance * AB_normalized[1])
    
    AP_proportion = AP_distance / vlen( AB )   
    in_segment =   0 <= AP_proportion <= 1

    CP = (AP[0] - AC[0], AP[1]-AC[1])
    in_circle = vlen( CP ) < r

    return in_circle and in_segment

def safe(a,b,obsts):
    for i in range(len(obsts)):
        if detect(a,b,(obsts[i][0],obsts[i][1]),obsts[i][2]):
            return False
    return True

def safe_segments(segments, obsts):
    for o in obsts:
        for p in range(len(segments) - 1):
            if detect(segments[p], segments[p+1], [o[0],o[1]], o[2]):
                return False
    return True

def fix_angle(theta):
    if theta < 0.0:
        return theta + 2 * pi
    if theta > 2 * pi:
        return theta - 2 * pi
    return theta

# Chooser that uses digits of Pi to make choices. Works for any base up to 10.
class PiChooser:
    def __init__(self, filename):
        # Load pi digits from file
        with open(filename, "r") as f:
            self.pi_digits = f.read().strip()
        self.index = 0  # keep track of which digit we’re on

    def choose(self, neighbors):
        if not neighbors:
            return None

        # Get next digit from pi (cycle back if at end)
        digit = int(self.pi_digits[self.index])
        self.index = (self.index + 1) % len(self.pi_digits)

        # Use digit to pick a neighbor
        choice = neighbors[digit % len(neighbors)]
        return choice