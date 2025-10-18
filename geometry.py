# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5

"""
from math import sqrt
import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy

#@TODO(V)
def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    r = alien.get_width()
    if alien.is_circle():
        cx, cy = alien.get_centroid()
        for(startx, starty, endx, endy) in walls:
            wall = ((startx,starty),(endx,endy))
            if point_segment_distance((cx,cy), wall) <= r:
                    return True
        return False
    else:
        head, tail = alien.get_head_and_tail()
        for(startx, starty, endx, endy) in walls:
            wall = ((startx,starty),(endx,endy))
            if segment_distance((head,tail), wall) <= r:
                    return True
        return False
    return False


#@TODO(V)
def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    w, h = window[0], window[1]
    r = alien.get_width()
    if alien.is_circle():
        x,y = alien.get_centroid()
        return (r < x < w - r and r < y < h - r)
    else:
        head, tail = alien.get_head_and_tail()
        for (x,y) in [head, tail]:
            if x <= r or x >= w - r or y <= r or y >= h - r:
                return False
        return True


#@TODO(V)
def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    A,B,C,D=polygon
    AB = (B[0] - A[0], B[1] - A[1])
    AD = (D[0] - A[0], D[1] - A[1])
    AP = (point[0] - A[0], point[1] - A[1])

    if AB[0]*AD[1] - AB[1]*AD[0] == 0:
        return False

    s = (AP[0]*AD[1] - AP[1]*AD[0]) / (AB[0]*AD[1] - AB[1]*AD[0])
    t = (AB[0]*AP[1] - AB[1]*AP[0]) / (AB[0]*AD[1] - AB[1]*AD[0])

    return 0 <= s <= 1 and 0 <= t <= 1


#@TODO(V)
def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    r = alien.get_width()

    if alien.is_circle():
        O = alien.get_centroid()
        for (x1, y1, x2, y2) in walls:
            wall = ((x1, y1), (x2, y2))
            if segment_distance((O, waypoint), wall) <= r:
                return True
        return False

    orig_pos = alien.get_centroid()
    H0, T0 = alien.get_head_and_tail()
    alien.set_alien_pos(waypoint)
    H1, T1 = alien.get_head_and_tail()
    alien.set_alien_pos(orig_pos)

    P = [H0, H1, T1, T0]
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        edges.append((P[i], P[j]))

    for (x1, y1, x2, y2) in walls:
        wall = ((x1, y1), (x2, y2))

        if is_point_in_polygon(wall[0], P) or is_point_in_polygon(wall[1], P):
            return True

        for e in edges:
            if do_segments_intersect(e, wall):
                return True

        for e in edges:
            d = segment_distance(e, wall)
            if d <= r:
                return True

    return False


#@TODO(V)
def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    x,y=p
    (x1,y1),(x2,y2)=s
    t1=(x2-x1, y2-y1)
    t2=(x-x1,y-y1)
    if(t1[0]*t1[0]+t1[1]*t1[1] == 0):
        return sqrt((x - x1)**2 + (y - y1)**2)
    t=(t1[0]*t2[0]+t1[1]*t2[1])/(t1[0]*t1[0]+t1[1]*t1[1])
    if t<0:
        return sqrt((x-x1)**2+(y-y1)**2)
    if t>1:
        return sqrt((x-x2)**2+(y-y2)**2)
    C = (x1+t*(x2-x1),y1+t*(y2-y1))
    return sqrt((x-C[0])**2+(y-C[1])**2)
    


#@TODO(V)
def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    A=s1[0]
    B=s1[1]
    C=s2[0]
    D=s2[1]
    Ax, Ay = A[0], A[1]
    Bx, By = B[0], B[1]
    Cx, Cy = C[0], C[1]
    Dx, Dy = D[0], D[1]
    
    if (max(Ax,Bx) < min(Cx,Dx) or max(Cx, Dx) < min(Ax, Bx) 
        or max(Ay, By) < min(Cy, Dy) or max(Cy, Dy) < min(Ay, By)):
        return False
    l1 = line_equation(A,B)
    l2 = line_equation(C,D)
    k1,k2,b1,b2 = l1[1],l2[1],l1[2],l2[2]
    if(l1[0] != 0 and l2[0] != 0):
        if k1 == k2:
            if b1 == b2:
                overlap_x = max(min(Ax, Bx), min(Cx, Dx)) <= min(max(Ax, Bx), max(Cx, Dx))
                overlap_y = max(min(Ay, By), min(Cy, Dy)) <= min(max(Ay, By), max(Cy, Dy))
                return overlap_x and overlap_y
            return False
        P = ((b2-b1)/(k1-k2),k1*(b2-b1)/(k1-k2)+b1)
    elif(l1[0] == 0 and l2[0] != 0):
        P=(-b1, k2*-b1+b2)
    elif(l1[0] != 0 and l2[0] == 0):
        P=(-b2, k1*-b2+b1)
    elif(l1[0]+l2[0]==0):
        if -b1 != -b2:
            return False
        return max(min(Ay, By), min(Cy, Dy)) <= min(max(Ay, By), max(Cy, Dy))
    x,y=P[0],P[1]
    return (x >= min(Ax, Bx) and x <= max(Ax, Bx) and
        y >= min(Ay, By) and y <= max(Ay, By) and
        x >= min(Cx, Dx) and x <= max(Cx, Dx) and
        y >= min(Cy, Dy) and y <= max(Cy, Dy))

def line_equation(A, B):
    Ax, Ay = A[0], A[1]
    Bx, By = B[0], B[1]
    if Ax==Bx:
        return (0,1,-Ax)
    else:
        k=(By-Ay)/(Bx-Ax)
        b=Ay-k*Ax
        return (1,k,b)



#@TODO(V)
def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1,s2):
        return 0
    A,B,C,D = s1[0],s1[1],s2[0],s2[1]
    return min(point_segment_distance(A, s2), point_segment_distance(B, s2), point_segment_distance(C, s1), point_segment_distance(D, s1))


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
