from unionfind import UnionFind
from heapq import heappush, heappop 
import numpy
from numpy import inf
import queue

class Tree:
    
    def __init__(self, directed = False,  weighted = False):

        self.directed = directed
        self.weighted = weighted
        self.tree = {}
        self.vertex_num = 0
        self.vertex = []
        self.components = UnionFind()
        
    def add_edge(self, origin, destiny, weight=0):
        
        def add_vertex(self, vertex):
            if not vertex in self.tree.keys():
                self.components.add(vertex)
                self.tree[vertex] = {}
                self.vertex_num += 1
                self.vertex.append(vertex)
        
        if not origin in self.tree.keys():
            add_vertex(self,origin)
            
        if not destiny in self.tree.keys():
            add_vertex(self,destiny)
            
        if self.components.connected(origin, destiny):
            raise Exception("Cannot add edge, would create a cicle")

        self.tree[origin][destiny] = weight
        if not self.directed:
            self.tree[destiny][origin] = weight
        
        self.components.union(origin,destiny)
        
    
class Graph:

    def __init__(self, directed = False,  weighted = False):

        self.directed = directed
        self.weighted = weighted
        self.graph = {}
        self.vertex_num = 0
        self.edge_num = 0
        self.cost_matrix = {}
        self.edge_list = []
        

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
        
        self.edge_list.append((weight, origin, destiny))
        self.edge_num += 1
            
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
        
        self.cost_matrix = Cost
                   
    
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
        
    
    def MST(self, algorithm = 'kruskal', root = '') :
        '''
            Minimum spanning tree
        '''
        
        if algorithm == 'kruskal':
            return self._kruskal()
        
        if algorithm == 'prim':
            if not root:
                root = list(self.graph.keys())[0]
            return self._prim(root)
        
        raise Exception("Algorithm mispelled or not available")
    
    
    def has_negative_cycle(self):
        
        Cost = self._floyd_warshall()
        
        for vertex in Cost.keys():
            if Cost[vertex][vertex] < 0:
                return True
        
        return False
        
        
    def _dfs(self, root, dis):
        
        for v in self.graph[root]:
            if not v in dis:
                dis[v] = dis[root] + 1
                dis = self._dfs(v, dis)
        
        return dis
    
    def _bfs(self, root):
        
        dis = {root:0}
        
        q = queue.Queue()
        q.put(root)
        
        while not q.empty():
            u = q.get()
            for v in self.graph[u]:
                if not v in dis:
                    dis[v] = dis[u]+1
                    q.put(v)
        
        return dis
        
        
    def _kruskal(self):
        
        graph = self.graph
        edge_list = sorted(self.edge_list)
        
        uf = UnionFind(list(graph.keys()))
        
        tree = Tree(weighted=True)
        minimum_cost = 0
        
        for cost, u, v in edge_list:
            
            if uf.n_comps == 1:
                break
                
            if not uf.connected(u,v):
                uf.union(u,v)
                minimum_cost += cost
                tree.add_edge(u,v,cost)
                
        return minimum_cost, tree
        
        
    def _prim(self, root):
        
        graph = self.graph
        heap = []
        
        tree = Tree(weighted=True)
        minimum_cost = 0
        
        for v in graph[root]:
            heappush(heap, (graph[root][v], root, v ))

        while self.vertex_num != tree.vertex_num :
            
            cost, u, v = heappop(heap)
            
            if v in tree.vertex:
                continue

            minimum_cost += cost
            tree.add_edge(u,v,cost)

            for u in graph[v]:
                if not u in tree.vertex:
                    heappush(heap, (graph[v][u], v, u))
        
        return minimum_cost, tree
        
    def _floyd_warshall(self):
        
        '''
            Floyd Warsall algorithm to solve the all pairs shortest path problem.
            Complexity: O(V^3)

            Output:
                distances : (Cost) Dictionary that, for every node, has the 
                                   shortest distance from that node to all the 
                                   other nodes in the graph
        
        '''
        
        self.get_cost_matrix()
        
        graph = self.graph
        Cost = self.cost_matrix
                        
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