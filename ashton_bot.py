import random
from game_world.racetrack import RaceTrack
from heapq import heappush, heappop
import math
from copy import deepcopy
import numpy as np



Point = tuple[int, int]

def astar(start_cell: Point, end_cell: Point, track: RaceTrack) -> list[Point] | None:
    safe = {(int(r), int(c)) for (r, c) in track.find_traversable_cells()}

    br, bc = np.where(track.buttons.astype(bool))
    buttons = set(zip(br.astype(int), bc.astype(int)))

    forbidden = buttons - {start_cell, end_cell}
    safe -= forbidden
    #set up frontier as a list of tuples that include the priority value, a counter for tie-breaking, and the NavMeshCell
    frontier: list[tuple[float, int, Point]] = []

    #essentially, the counter is just what is going to be compared when priorities are equal
    counter = 1
    heappush(frontier, (0.0, counter, start_cell))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_cell] = None
    cost_so_far[start_cell] = 0
    moves = [(1,0),(0,1),(-1,0),(0,-1)]

    #loop through frontier as long as it isn't empty
    while frontier:
        #discard priority and count values of this cell in the frontier because we only need the cell itself
        _priority, _count, current = heappop(frontier)

        #if we've reached the end_cell, we know that this is the shortest path because we found it first
        #go through came_from and add the path of cells to the list, eventually reversing it because we start from end_cell
        if current == end_cell:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        for dr, dc in moves:
            neighbor = (current[0] + dr, current[1] + dc)

            if neighbor not in safe:
                continue

            #find the cost to get from start to the neighbor
            new_cost = cost_so_far[current] + 1

            #check if the neighbor either hasn't been visited or if it is cheaper to go there
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]: 
                cost_so_far[neighbor] = new_cost

                #priority is the new cost added to the heuristic distance
                priority = new_cost + (math.dist(neighbor,end_cell)) # f = g + h
                counter+=1

                #add this neighbor to the frontier heapk
                heappush(frontier,(priority, counter, neighbor))

                #add the current cell to the came_from dict 
                came_from[neighbor] = current
    return None
def plan_route(start: Point, track: RaceTrack) -> list[Point] | None:
    def state_key(pos: Point, t: RaceTrack) -> tuple[Point, bytes]:
        pos = (int(pos[0]), int(pos[1]))
        return (pos, t.active.astype(np.int8).tobytes())

    memo: dict[tuple[Point, bytes], tuple[int, list[Point]] | None] = {}
    visiting: set[tuple[Point, bytes]] = set()

    def helper(pos: Point, t: RaceTrack) -> tuple[int, list[Point]] | None:
        key = state_key(pos, t)


        if key in memo:
            return memo[key]


        if key in visiting:
            return None
        visiting.add(key)

        best: tuple[int, list[Point]] | None = None

        path_to_target = astar(pos, t.target, t)
        if path_to_target is not None:
            best = (len(path_to_target) - 1, [t.target])

        candidates: list[tuple[int, Point]] = []
        for b in t.find_buttons():
            b = (int(b[0]), int(b[1]))
            path_to_b = astar(pos, b, t)
            if path_to_b is None:
                continue
            candidates.append((len(path_to_b) - 1, b))

        candidates.sort(key=lambda x: x[0])   

        for dist_to_b, b in candidates:
            t2 = deepcopy(t)
            color = int(t2.button_colors[b[0], b[1]])
            t2.toggle(color)

            sub = helper(b, t2)
            if sub is None:
                continue

            sub_cost, sub_route = sub
            total_cost = dist_to_b + sub_cost
            route = [b] + sub_route

            if best is None or total_cost < best[0]:
                best = (total_cost, route)

        visiting.remove(key)
        memo[key] = best
        return best

    result = helper(start, track)
    return None if result is None else result[1]

route: list[Point] | None = None 
i: int = 0
planned: bool = False

def ashton_move(loc: Point, track: RaceTrack) -> Point:
    global route, i, planned


    t_plan = deepcopy(track)
    if t_plan.buttons[loc]:
        t_plan.toggle(int(t_plan.button_colors[loc]))


    if not planned:
        print("planing route")
        route = plan_route(loc, t_plan)  
        i = 0
        planned = True

    if not route:
        path = astar(loc, t_plan.target, t_plan)
        if path and len(path) >= 2:
            nxt = path[1]
            return (nxt[0] - loc[0], nxt[1] - loc[1])
        return (0, 0)

  
    while i < len(route) and loc == route[i]:
        i += 1

    goal = route[i] if i < len(route) else t_plan.target

    path = astar(loc, goal, t_plan)
    if path and len(path) >= 2:
        nxt = path[1]
        return (nxt[0] - loc[0], nxt[1] - loc[1])

    planned = False
    route = None
    return (0, 0)
