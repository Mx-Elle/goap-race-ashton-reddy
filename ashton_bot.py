import random
from game_world.racetrack import RaceTrack
from heapq import heappush, heappop


Point = tuple[int, int]


def astar(start_cell: tuple, end_cell: tuple) -> list[tuple]| None:
    
    #set up frontier as a list of tuples that include the priority value, a counter for tie-breaking, and the NavMeshCell
    frontier: list[tuple[float, int, tuple]] = []

    #essentially, the counter is just what is going to be compared when priorities are equal
    counter = 1
    heappush(frontier, (0.0, counter, start_cell))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_cell] = None
    cost_so_far[start_cell] = 0

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

        for neighbor in current.neighbors:

            #find the cost to get from start to the neighbor
            new_cost = cost_so_far[current] + current.distance(neighbor)

            #check if the neighbor either hasn't been visited or if it is cheaper to go there
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]: 
                cost_so_far[neighbor] = new_cost

                #priority is the new cost added to the heuristic distance
                priority = new_cost + neighbor.distance(end_cell) # g = f + h
                counter+=1

                #add this neighbor to the frontier heapk
                heappush(frontier,(priority, counter, neighbor))

                #add the current cell to the came_from dict 
                came_from[neighbor] = current
    return None          

def ashton_move(loc: Point, track: RaceTrack) -> Point:
    safe = track.find_traversable_cells()
    options = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = {opt: (loc[0] + opt[0], loc[1] + opt[1]) for opt in options}
    safe_options = [opt for opt in neighbors if neighbors[opt] in safe]
    return random.choice(safe_options)