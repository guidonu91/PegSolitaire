from utils import *
import math, random, sys, time, bisect, string
import copy, sys

#______________________________________________________________________________

class Problem(object):
    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial; self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        abstract

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        abstract

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Override this
        method if checking against a single self.goal is not enough."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        abstract
#______________________________________________________________________________

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        update(self, state=state, parent=parent, action=action,
               path_cost=path_cost, depth=0)
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        "Fig. 3.10"
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state, action, next))

    def solution(self):
        "Return the sequence of actions to go from the root to this node."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

#______________________________________________________________________________
# Uninformed Search algorithms

def tree_search(problem, frontier):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Don't worry about repeated paths to a state. [Fig. 3.7]"""
    frontier.append(Node(problem.initial))
    #Agrego tres contadores de modo a responder el ejercicio 8
    #Para probar descomentar las instrucciones comentadas y comentar los returns
    #maximo_branching_factor = 0
    #total_de_nodos_del_arbol = 1
    #total_de_soluciones = 0
    #c_espacial_BFTS = 0
    while frontier:
        node = frontier.pop()
        #c_espacial_BFTS =  c_espacial_BFTS + 1
        if problem.goal_test(node.state):
            #total_de_soluciones = total_de_soluciones + 1
            #c_espacial_BFTS =  c_espacial_BFTS + len(frontier)
            #print c_espacial_BFTS
            return node
        nodos_generados = node.expand(problem)
        frontier.extend(nodos_generados)
        #total_de_nodos_del_arbol += len(nodos_generados)
        #if (len(nodos_generados)> maximo_branching_factor):
            #maximo_branching_factor = len(nodos_generados)
    return None
    #print 'maximo_branching_factor =', maximo_branching_factor
    #print 'total_de_nodos_del_arbol =', total_de_nodos_del_arbol
    #print 'total_de_soluciones =', total_de_soluciones
    
def graph_search(problem, frontier):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    If two paths reach a state, only use the first one. [Fig. 3.7]"""
    frontier.append(Node(problem.initial))
#   explored = set()
    explored = []
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
#       explored.add(node.state)
        if explored.count(node.state)==0:
            explored.append(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored
                        and child not in frontier)
    return None

def breadth_first_tree_search(problem):
    "Search the shallowest nodes in the search tree first."
    return tree_search(problem, FIFOQueue())

def depth_first_tree_search(problem):
    "Search the deepest nodes in the search tree first."
    return tree_search(problem, Stack())

def depth_first_graph_search(problem):
    "Search the deepest nodes in the search tree first."
    return graph_search(problem, Stack())

def breadth_first_search(problem):
    "[Fig. 3.11]"
    c_espacial_BFGS = 1
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = FIFOQueue()
    frontier.append(node)
#   explored = set()
    explored = []
    while frontier:
        c_espacial_BFGS = c_espacial_BFGS + 1
        node = frontier.pop()
#       explored.add(node.state)
        if explored.count(node.state)==0:
            explored.append(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                c_espacial_BFGS = c_espacial_BFGS + 1
                if problem.goal_test(child.state):
                    c_espacial_BFGS = len(explored) + len(frontier)
                    print c_espacial_BFGS
                    return child
                frontier.append(child)
    return None

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
#   explored = set()
    explored = []
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
#       explored.add(node.state)
        if explored.count(node.state)==0:
            explored.append(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None  

def uniform_cost_search(problem):
    "[Fig. 3.14]"
    return best_first_graph_search(problem, lambda node: node.path_cost)

def depth_limited_search(problem, limit=100): #limite mas grande
    "[Fig. 3.17]"
    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return if_(cutoff_occurred, 'cutoff', None)

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)

def recursive_depth_first_search(problem): 
    "[Fig. 3.17]"
    def recursive_dls(node, problem):
        if problem.goal_test(node.state):
            return node
        else:
            expandidos = node.expand(problem)
            expandidos.reverse()
            for child in expandidos:
                result = recursive_dls(child, problem)
                if result is not None:
                    return result
    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem)

def iterative_deepening_search(problem):
    "[Fig. 3.18]"
    for depth in xrange(sys.maxint):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result

#______________________________________________________________________________
# Informed (Heuristic) Search

greedy_best_first_graph_search = best_first_graph_search
    # Greedy best-first search is accomplished by specifying f(n) = h(n).

def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))

def recursive_greedy_best_first_graph_search(problem, f):  
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    #explored = set()
    explored = []
    def recursive_greedy(problem, f, frontier, explored):
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
#       explored.add(node.state)
        if explored.count(node.state)==0:
            explored.append(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
        return recursive_greedy(problem, f, frontier, explored)
    return recursive_greedy(problem, f, frontier, explored)

#______________________________________________________________________________
# Code to compare searchers on various problems.

class InstrumentedProblem(Problem):
    """Delegates to a problem, and keeps statistics."""

    def __init__(self, problem):
        self.problem = problem
        self.succs = self.goal_tests = self.states = 0
        self.found = None

    def actions(self, state):
        self.succs += 1
        return self.problem.actions(state)

    def result(self, state, action):
        self.states += 1
        return self.problem.result(state, action)

    def goal_test(self, state):
        self.goal_tests += 1
        result = self.problem.goal_test(state)
        if result:
            self.found = state
        return result

    def path_cost(self, c, state1, action, state2):
        return self.problem.path_cost(c, state1, action, state2)

    def value(self, state):
        return self.problem.value(state)

    def __getattr__(self, attr):
        return getattr(self.problem, attr)

    def __repr__(self):
        return '<%4d/%4d/%d>' % (self.succs, self.goal_tests, self.states)

def compare_searchers_uninformed(problems, header,
                      searchers=[breadth_first_tree_search,
                                 breadth_first_search,
                                 uniform_cost_search,
                                 depth_first_tree_search,
                                 depth_first_graph_search,
                                 recursive_depth_first_search,
                                 iterative_deepening_search]):
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        searcher(p)
        return p
    table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)

def compare_searchers_informed(problems, header, heuristic,
                               searchers=[greedy_best_first_graph_search,
                                          recursive_greedy_best_first_graph_search,
                                          astar_search]):
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        searcher(p,heuristic)
        return p
    table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)

#______________________________________________________________________________
#Definicion del problema Zoz

class ZozProblem(Problem):
    """Extension de la clase problem para el juego Zoz"""
    def __init__(self, N, x, y):
        """Inicializo el problema Zoz, asigno atributos para el lado 'N' del tablero, creo
        el estado inicial, y un estado ficticio para la busqueda bidireccional"""

        #Creo el atributo para el lado 'N' del tablero
        self.N = N

        #Creo el estado inicial a partir de los valores obtenidos de N, x,y
        self.initial = [[1 for j in range(i+1)] for i in range(N)]
        self.initial[x][y] = 0

        #Creo un estado ficticio utilizado para la busqueda bidireccional
        self.inicial_ficticio = [[0 for j in range(i+1)] for i in range(N)] 
        
    def actions(self, state):
        """Defino las acciones que permitiran crear nuevos nodos"""
        acciones = []
        if self.cantidad_fichas(state) <= self.cantidad_celdas()/2:     #Encuentro fichas y pregunto por espacios en blanco a su alrededor
                                                                        #(Esto tambien se hace en caso de igual numero de fichas y espacios en blanco)
            for i in range(self.N):                                       
                for j in range(i+1):
                    if state[i][j] == 1:
                        if i-2 >=0 and j-2 >= 0 and state[i-2][j-2] == 0 and state[i-1][j-1] == 1: #izquierda arriba
                            acciones.append([[i,j],[i-1,j-1],[i-2,j-2]])
                        if j-2 >= 0 and state[i][j-2] == 0 and state[i][j-1] == 1: #izquierda medio
                            acciones.append([[i,j],[i,j-1],[i,j-2]])
                        if i+2 < self.N and state[i+2][j] == 0 and state[i+1][j] == 1: #izquierda abajo    
                            acciones.append([[i,j],[i+1,j],[i+2,j]])
                        if i+2 < self.N and state[i+2][j+2] == 0 and state[i+1][j+1] == 1: #derecha abajo
                            acciones.append([[i,j],[i+1,j+1],[i+2,j+2]])
                        if j+2 <= i and state[i][j+2] == 0 and state[i][j+1] == 1: #derecha medio
                            acciones.append([[i,j],[i,j+1],[i,j+2]])
                        if j <= i-2 and state[i-2][j] == 0 and state[i-1][j] == 1: #derecha arriba
                            acciones.append([[i,j],[i-1,j],[i-2,j]])
        else:                                                           #Encuentro espacios en blanco y pregunto por fichas a su alrededor                           
            for i in range(self.N):                 
                for j in range(i+1):
                    if state[i][j] == 0:
                        if i-2 >=0 and j-2 >= 0 and state[i-2][j-2] == 1 and state[i-1][j-1] == 1: #izquierda arriba
                            acciones.append([[i-2,j-2],[i-1,j-1],[i,j]])
                        if j-2 >= 0 and state[i][j-2] == 1 and state[i][j-1] == 1: #izquierda medio
                            acciones.append([[i,j-2],[i,j-1],[i,j]])
                        if i+2 < self.N and state[i+2][j] == 1 and state[i+1][j] == 1: #izquierda abajo    
                            acciones.append([[i+2,j],[i+1,j],[i,j]])
                        if i+2 < self.N and state[i+2][j+2] == 1 and state[i+1][j+1] == 1: #derecha abajo
                            acciones.append([[i+2,j+2],[i+1,j+1],[i,j]])
                        if j+2 <= i and state[i][j+2] == 1 and state[i][j+1] == 1: #derecha medio
                            acciones.append([[i,j+2],[i,j+1],[i,j]])
                        if j <= i-2 and state[i-2][j] == 1 and state[i-1][j] == 1: #derecha arriba
                            acciones.append([[i-2,j],[i-1,j],[i,j]])
        return acciones                  
    
    def cantidad_fichas(self, state):
        """Funcion que retorna la cantidad de fichas que contiene el tablero"""
        cont = 0
        for i in range(self.N):
            for j in range(i+1):
                if state[i][j] == 1:
                    cont = cont + 1
        return cont

    def cantidad_celdas(self):
        """Funcion que retorna la cantidad de celdas del tablero"""
        return self.N*(self.N+1)/2

    def result(self, state, movimiento):
        """A partir de las acciones obtenidas genero el nuevo estado"""
        new = copy.deepcopy(state)
        new[movimiento[0][0]][movimiento[0][1]] = 0
        new[movimiento[1][0]][movimiento[1][1]] = 0
        new[movimiento[2][0]][movimiento[2][1]] = 1
        return new

    def goal_test(self, state):
        """El estado meta se halla al encontrar una configuracion de tablero con una sola ficha colocada"""
        if self.cantidad_fichas(state) == 1:
            return True
        else:
            return False

    def predecesors(self, state):
        """Calcula los predecesores de un nodo, devuelve una lista con los estados predecesores al estado del nodo actual"""

        #Lista con los estados predecesores
        pred = []

        #Si el estado analizado es el estado inicial ficticio (celdas todas vacias), cargo en pred los estados meta de zoz 
        if state == self.inicial_ficticio:
            for i in range(self.N):                                       
                for j in range(i+1):
                    new = copy.deepcopy(state)
                    new[i][j] = 1
                    pred.append(new)

        #Si el estado analizado no corresponde al estado inicial ficticio, calculo los estados que preceden al estado actual
        else:

            #Lista que guarda acciones, donde una accion es una lista que contiene las posiciones de una celda con una ficha
            #colocada y las posiciones de dos celdas vacias (la primera celda vacia adyacente a la celda con ficha, y la
            #segunda celda vacia adyacente a la primera en la misma linea que la celda con ficha)
            acciones = []

            #Caculo cada accion y cargo en acciones
            for i in range(self.N):                                       
                for j in range(i+1):
                    if state[i][j] == 1:
                        #Celdas vacias izquierda arriba
                        if i-2 >=0 and j-2 >= 0 and state[i-2][j-2] == 0 and state[i-1][j-1] == 0: 
                            acciones.append([[i,j],[i-1,j-1],[i-2,j-2]])

                        #Celdas vacias izquierda medio
                        if j-2 >= 0 and state[i][j-2] == 0 and state[i][j-1] == 0: 
                            acciones.append([[i,j],[i,j-1],[i,j-2]])

                        #Celdas vacias izquierda abajo
                        if i+2 < self.N and state[i+2][j] == 0 and state[i+1][j] == 0:     
                            acciones.append([[i,j],[i+1,j],[i+2,j]])

                        #Celdas vacias derecha abajo
                        if i+2 < self.N and state[i+2][j+2] == 0 and state[i+1][j+1] == 0: 
                            acciones.append([[i,j],[i+1,j+1],[i+2,j+2]])

                        #Celdas vacias derecha medio
                        if j+2 <= i and state[i][j+2] == 0 and state[i][j+1] == 0: 
                            acciones.append([[i,j],[i,j+1],[i,j+2]])

                        #Celdas vacias derecha arriba
                        if j <= i-2 and state[i-2][j] == 0 and state[i-1][j] == 0: 
                            acciones.append([[i,j],[i-1,j],[i-2,j]])
                            
            #A partir de las acciones calculadas, genero los nuevos estados y los cargo en pred              
            for accion in acciones: 
                new = copy.deepcopy(state)
                new[accion[0][0]][accion[0][1]] = 0
                new[accion[1][0]][accion[1][1]] = 1
                new[accion[2][0]][accion[2][1]] = 1
                pred.append(new)
                
        return pred

#______________________________________________________________________________
#Definicion de la heuristica utilizada las estrategias de busqueda informada

def heuristic(node):
    """Suma 1 por cada ficha colocada en los bordes y por cada celda vacia en el interior"""
    h = 0
    n = len(node.state)
    for i in range(n):
        for j in range(i+1):
            if i==(n-1):
                if node.state[i][j]==1:
                    h = h+1
            else:
                if j==0 or j==i:
                    if node.state[i][j]==1:
                        h = h+1
                else:
                    if node.state[i][j]==0:
                        h = h+1
    return h 

#______________________________________________________________________________
#Funciones auxiliares para imprimir estados y acciones

def imprimir(a, n):
    l = []
    while a!=None:	
            l.append(a.state)
            l.append(a.action)
            a = a.parent
    l.pop()
    l.reverse()
    for i in range(len(l)):
            if i%2 == 0: 
                    if i==0:
                        print 'Estado Inicial:',
                    elif i==(len(l)-1):
                        print 'Estado Meta:',
                    else:
                        print 'Estado:',
                    print l[i]
                    for j in range(n):
                        sys.stdout.write(' '*((n-1)-j))
                        for k in range(j+1):
                            print (l[i][j][k]),
                        print ''
            else:
                    print 'Accion:', l[i]
                    #print 'Ficha a mover en la posicion:', l[i][0]
                    #print 'Ficha comida en la posicion:', l[i][1]
                    #print 'Ficha movida a la posicion:', l[i][2]

                    l[i-1][l[i][0][0]][l[i][0][1]] = 'I'
                    l[i-1][l[i][2][0]][l[i][2][1]] = 'F'
                                        
                    for j in range(n):
                        sys.stdout.write(' '*((n-1)-j))
                        for k in range(j+1):
                            print (l[i-1][j][k]),
                        print ''
            print ''

def imprimir_estado(estado, n):
    print 'Estado:',
    print estado
    for j in range(n):
        sys.stdout.write(' '*((n-1)-j))
        for k in range(j+1):
            print (estado[j][k]),
        print ''
    print ''
#______________________________________________________________________________
#Main del programa

while (1):
    while (1):
        print """Trabajo Practico de Informatica 2 - ZOZ
a) Ejercicio 2
b) Ejercicio 5
c) Ejercicio 6
d) Compare_searchers
e) Salir\n"""
        opcion = raw_input("Elija la opcion deseada:\n")
        if (opcion>='a' and opcion<='e'): break
        print 'Valor no valido\n'

    if opcion == 'e': break

    if opcion == 'a':
        while(1):
            while (1):
                print """ESTRATEGIAS DE BUSQUEDA
a) Breadth First Tree Search (BFTS)
b) Breadth First Graph Search (BFGS)
c) Uniform Cost Graph Search (UCGS)
d) Depth First Tree Search (DFTS)
e) Depth First Graph Search (DFGS)
f) Recursive Depth First Search (RDFS)
g) Iterative Deepening Search (IDS)
h) Greedy Best First Graph Search (GBFGS)
i) Recursive Greedy Best First Search (RGBFS)
j) A* (A*S)
k) Volver al menu principal\n"""
                opcion = raw_input("Elija la opcion deseada:\n")
                if (opcion>='a' and opcion<='k'): break
                print 'Valor no valido\n'
            
            if opcion == 'k': break;

            while (1):
                n = input("Ingrese N:\n")
                f = input("Ingrese la fila de la celda vacia\n")
                c = input("Ingrese la columna de la celda vacia\n")
                if (n>=5 and n<=10 and f>=0 and f<n and c>=0 and f>=c): break
                print 'Valores no validos\n'
                
            if opcion == 'a':
                print '\nBreadth First Tree Search (BFTS)'
                a = breadth_first_tree_search(ZozProblem(n,f,c))
                imprimir(a, n)
            elif opcion == 'b':
                print '\nBreadth First Graph Search (BFGS)'
                a = breadth_first_search(ZozProblem(n,f,c))
                imprimir(a, n)
            elif opcion == 'c':
                print '\nUniform Cost Graph Search (UCGS)'
                a = uniform_cost_search(ZozProblem(n,f,c))
                imprimir(a, n)
            elif opcion == 'd':
                print '\nDepth First Tree Search (DFTS)'
                a = depth_first_tree_search(ZozProblem(n,f,c))
                imprimir(a, n)
            elif opcion == 'e':
                print '\nDepth First Graph Search (DFGS)'
                a = depth_first_graph_search(ZozProblem(n,f,c))
                imprimir(a, n)
            elif opcion == 'f':
                print '\nRecursive Depth First Search (RDFS)'
                a = recursive_depth_first_search(ZozProblem(n,f,c))
                imprimir(a, n)
            elif opcion == 'g':
                print '\nIterative Deepening Search (IDS)'
                a = iterative_deepening_search(ZozProblem(n,f,c))
                imprimir(a, n)
            elif opcion == 'h':
                print '\nGreedy Best First Graph Search (GBFGS)'
                a = best_first_graph_search(ZozProblem(n,f,c),heuristic)
                imprimir(a, n)
            elif opcion == 'i':
                print '\nRecursive Greedy Best First Search (RGBFS)'
                a = recursive_best_first_search(ZozProblem(n,f,c),heuristic)
                imprimir(a, n)
            elif opcion == 'j':
                print '\nA* (A*S)'
                a = astar_search(ZozProblem(n,f,c),heuristic)
                imprimir(a, n)
                
            raw_input('Presione ENTER para continuar:\n')

    if opcion == 'b':
        print 'Se realizara tres evaluaciones de la funcion predecesora sobre los estados siguientes:'

        estados_a_probar = []
        estados_a_probar.append([[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0, 0]])
        estados_a_probar.append([[0], [1, 1], [0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0, 0]])
        estados_a_probar.append([[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]])

        for i in range(3):
            imprimir_estado(estados_a_probar[i], 5)

        raw_input('Presione ENTER para continuar:\n')

        for i in range(3):
            print 'Evaluacion',i+1
            imprimir_estado(estados_a_probar[i], 5)
            predecesores = ZozProblem(5,2,1).predecesors(estados_a_probar[i])
            print 'Estados predecesores'
            for estado in predecesores:
                imprimir_estado(estado,5)
            raw_input('Presione ENTER para continuar:\n')
    
    if opcion == 'c':
        print 'Estrategia utilizada: Depth First Graph Search (DFGS)'
        for i in range(5):                                       
            for j in range(i+1):
                goal = depth_first_graph_search(ZozProblem(5,i,j))
                print 'Celda vacia inicial:',i,j
                print 'Estado meta:',goal.state
                
        raw_input('\nPresione ENTER para continuar:\n')

    if opcion == 'd':
        print 'Resultados de la funcion compare_searchers para Zoz con N=5 y celda vacia en 2,1'
        print 'Busquedas no informadas' 
        compare_searchers_uninformed([ZozProblem(5,2,1)],['Searcher','ZozProblem'])
        print 'Busquedas informadas' 
        compare_searchers_informed([ZozProblem(5,2,1)],['Searcher','ZozProblem'], heuristic)
        raw_input('\nPresione ENTER para continuar:\n')


    
