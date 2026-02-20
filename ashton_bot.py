import random
from game_world.racetrack import RaceTrack
from heapq import heappush, heappop
import math
from copy import deepcopy
import numpy as np

Point = tuple[int, int]
Stateid = tuple[Point, bytes]

moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def state_id(pos: Point, track: RaceTrack) -> Stateid:
    # I identify each separate state uniquely
    p = (int(pos[0]), int(pos[1]))
    return (p, track.active.astype(np.int8).tobytes())


def distance(a: Point, b: Point) -> int:
    # to get dist
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def apply_button_if_present(pos: Point, track: RaceTrack) -> None:
    # toggles track if bot is on button
    p = (int(pos[0]), int(pos[1]))
    if track.buttons[p]:
        track.toggle(int(track.button_colors[p]))


def astar_state_space(start: Point, track: RaceTrack) -> list[Point] | None:
    # I am astaring through the open space to look for the best path I can find
    new_track = deepcopy(track)
    apply_button_if_present(start, new_track)

    start_pos = (int(start[0]), int(start[1]))
    goal = (int(new_track.target[0]), int(new_track.target[1]))

    start_k = state_id(start_pos, new_track)

    frontier = []
    tie = 0

    g_best = {start_k: 0}
    parent = {start_k: None}

    heappush(frontier, (distance(start_pos, goal), tie, 0, start_pos, new_track))

    while frontier:
        _, _, g, pos, t = heappop(frontier)
        k = state_id(pos, t)

        if g != g_best.get(k, 10**16):
            continue

        if pos == goal:
            path: list[Point] = []
            current: Stateid | None = k
            while current is not None:
                path.append(current[0])
                current = parent[current]
            path.reverse()
            return path

        safe = set()
        for cell in t.find_traversable_cells():
            r, c = cell
            safe.add((int(r),int(c)))
        #print(safe)

        for dr, dc in moves:
            next = (pos[0] + dr, pos[1] + dc)
            if next not in safe:
                continue

            t2 = deepcopy(t)
            apply_button_if_present(next, t2)
            k2 = state_id(next, t2)
            g2 = g + 1

            if g2 < g_best.get(k2, 10**18):
                g_best[k2] = g2
                parent[k2] = k
                tie += 1
                f2 = g2 + distance(next, goal)
                heappush(frontier, (f2, tie, g2, next, t2))

    return None

route = None
i = 0
planned = False


def ashton_move(loc: Point, track: RaceTrack) -> Point:
    # funtion to submit moves based on calculated path
    global route, i, planned

    if not planned:
        route = astar_state_space(loc, track)
        i = 0
        planned = True

    if not route or len(route) < 2:
        planned = False
        route = None
        return (0, 0)

    loc = (int(loc[0]), int(loc[1]))

    while i < len(route) and route[i] == loc:
        i += 1

    if i >= len(route):
        planned = False
        route = None
        return (0, 0)

    next = route[i]
    return (int(next[0]) - loc[0], int(next[1]) - loc[1])
