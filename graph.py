from heapq import heappush, heappop 
import numpy
from numpy import inf

class Graph:

    def __init__(self, directed = False,  weighted = False):

        self.directed = directed
        self.weighted = weighted
        self.graph = {}
        self.vertex_num = 0

    def add_vertex(self, vertex):
        if not vertex in self.graph.keys():
            self.graph[vertex] = {}
            self.vertex_num += 1
        
    def add_edge(self, origin, destiny, weight=0):
        
        if not origin in self.graph.keys():
            self.add_vertex(origin)
            
        if not destiny in self.graph.keys():
            self.add_vertex(destiny)

        self.graph[origin][destiny] = weight
        if not self.directed:
            self.graph[destiny][origin] = weight
            
            
    def get_cost_matrix(self):
        
        graph = self.graph
        Cost = {}
        for i in graph.keys():
            Cost[i] = {}
            for j in graph.keys():
                if j in graph[i].keys():
                    Cost[i][j] = graph[i][j]
                else:
                    Cost[i][j] = 0 if i==j else inf
        return Cost
                   
        
    
    def SSSP(self, source, get_paths = False, algorithm = 'dijkstra'):
        '''
            Single source shortest path
        '''
        return self._dijkstra(source, get_paths = get_paths)
    
    
    
    def APSP(self, algorithm = 'floyd warshall'):
        '''
            All pairs shortest path
        '''
        
        if algorithm == 'floyd warshall':
            return self._floyd_warshall()
        
        if algorithm == 'dijkstra':
            Cost = {}
            for i in self.graph.keys():
                Cost[i] = self._dijkstra(i)
            return Cost
        
        raise Exception("Algorithm mispelled or not available")
        
        
        
    def _floyd_warshall(self):
        
        '''
            Floyd Warsall algorithm to solve the all pairs shortest path problem.
            Complexity: O(V^3)

            Output:
                distances : (Cost) Dictionary that, for every node, has the 
                                   shortest distance from that node to all the 
                                   other nodes in the graph
        
        '''
        
        graph = self.graph
        Cost = self.get_cost_matrix()
                        
        for i in graph.keys():
            for j in graph.keys():
                for k in graph.keys():
                    Cost[j][k] = min(Cost[j][k], Cost[j][i] + Cost[i][k])
                    
        return Cost   
        
        
        
    def _dijkstra(self, source, get_paths = False):
        '''
            Dijkstra algorithm (lazy deletion and using a heap to get the minimum)
            Complexity: O((V+E) log V)

            Input:
                source    : (Obj)  The source vertex

            Output:
                distances : (dict) Dictionary that, for every key, has the 
                                   shortest distance from the source to 
                                   that key value

                paths     : (dict) Dictionary that, for every key, has the an 
                                   array with the vertexes visited in the shortest 
                                   path from the source to that key

        '''
        graph = self.graph
        # The heap to obtain the minimum in each iteration
        heap = []

        distances = { source: 0 }
        paths = { source: [source] }

        # Updating heap
        for u in graph[source]:
            heappush(heap, (graph[source][u], u, source))

        while heap:
            # Vertex at the shortest distance in current step
            distance, s, last_vertex = heappop(heap)

            # Lazy deletion
            if s in distances.keys():
                continue

            # Obtained shortest distance to current vertex, updating values
            distances[s] = distance
            
            if get_paths:
                paths[s] = [v for v in paths[last_vertex]]
                paths[s].append(s)

            # Updating heap
            for u in graph[s]:
                if not u in distances.keys():
                    heappush(heap, (graph[s][u] + distances[s], u, s))

        if get_paths:
            return distances, paths
        else:
            return distances