#---------------------------------------------------------------------------------------------------------------------
'''
Assignment 1
Student Name: Hai Nam Ngo
Student ID: 103488515
'''
#---------------------------------------------------------------------------------------------------------------------
import sys  #this is used mainly for reading text file and handle data in it.
import os   #this is used for handling error related to path file.
from queue import PriorityQueue   #this is used for A* search only
import pygame   #this is used for visualization for the GUI
import time

class MyMap:
    def __init__(self, grid_size, walls, initial_state, goal_states):
        #create the map after reading text file from the main class (down below) with all of the necessary attributes
        self.grid_size = grid_size   #related to max height and max width (size of the map)
        self.walls = walls  #related to data of the wall (pure data that is read from the text file)
        self.initial_state = initial_state     #the start point of the agent
        self.goal_states = goal_states        #list of the goal states 
        
        #these two attributes are created so other classes can access them
        self.wall_list = self.make_wall_list(walls)           
        self.path_list = self.make_path_list(grid_size, walls)
   
    #This function is created to create a wall list, which is convert from the pure data
    #We can identify which cell in the map is wall
    def make_wall_list(self, walls):
        wall_list = []
        for wall in walls:
            x, y, width, height = wall  #x and y is the coordinate, width and height is the size of wall
            
            for i in range(x, x + width):
                for j in range(y, y + height):
                    wall_list.append((i, j))                             
        return wall_list   
    
    #this function is created to identify the available path for the agent (If it is in the map and it is not in the wall list, then the cell is a valid path)    
    def make_path_list(self, grid_size, walls):
        wall_list = self.make_wall_list(walls)
        path_list = []
        
        height, width = grid_size

        for i in range(width):
            for j in range(height):
                if (i, j) not in wall_list:
                    path_list.append((i, j))
        return path_list

#---------------------------------------------------------------------------------------------------------------------
#This is the parent class of all other search methods, it has reuseable functions that can be shared among all search methods
class Search:
    def __init__(self, my_map, initial_state, goal_states):
        self.my_map = my_map
        self.initial_state = initial_state
        self.goal_states = goal_states

    #this function is created to check if the goal state is the same as the current state or not
    #if it is the same, that means the goal state is found
    def check_goal_state(self, current_state):
        return current_state in self.goal_states   #the output will be TRUE or FALSE

    #this function is created to find the surrounding cell of the current states in four directions (UP, DOWN, LEFT, RIGHT)
    def find_neighbor(self, current_state,directions):
        neighbor_list = []
        for direction in directions:
            x, y = current_state[0] + direction[0], current_state[1] + direction[1]
            neighbor = (x, y)
            neighbor_list.append(neighbor)
        return neighbor_list

    #this is the translator, it can read the current state and new state to identify the direction that the agent has moved
    def getDirection(self, current_state, new_state):
        x, y = current_state
        nx, ny = new_state
        
        if (nx, ny) == (x, y - 1):
            return 'up'
        elif (nx, ny) == (x, y + 1):
            return 'down'
        elif (nx, ny) == (x - 1, y):
            return 'left'
        elif (nx, ny) == (x + 1, y):
            return 'right'
    
    #this function will used the output path of the search methods to create a list of the direction that the agent has moved    
    def convert_to_directions(self,Path):
        direction_list = []  #blank list for the direction
        
        #each move is actually a dictionary with a child cell and a parent cell
        for current_state, new_state in Path.items():
            direction = self.getDirection(current_state, new_state)
            direction_list.append(direction)
        return direction_list  

    #This function generates a tuple of output containing necessary information for the main class to print out
    def generate_output(self, explored, initial_path):
        '''
        Example: initial_path can be a dictionary with a child cell and a parent cell
        initial_path = ['B': 'A', 'C': 'B']
        However we cannot identify the direction with this path, but if we reverse it, it will become ['A': 'B', 'B': 'C']
        So we know that A -> B -> C. And this function will do that for me. That is the logic of this function
        '''
        #Initialize an empty dictionary to store the forward path
        fwdPath = {}
        #Initialize an empty list to store the direction list
        direction_list = []  

        #Identify which goal is reached
        for goal in self.goal_states:  
            #Check if the goal state is reached during the search
            if goal in explored: 
                cell = goal
                path = []

                #Reconstruct the path from the goal state to the initial state
                while cell != self.initial_state:
                    path.append(cell)
                    cell = initial_path[cell]
                
                #Add the initial state to the path and reverse it to get the forward path
                path.append(self.initial_state)
                path.reverse()

                #Construct the forward path dictionary
                for i in range(len(path) - 1):
                    fwdPath[path[i]] = path[i + 1]

                #Convert the forward path to a list of directions
                direction_list = self.convert_to_directions(fwdPath)
                
                #Return the goal state, number of explored states, direction list, and the forward path
                return goal, len(explored), direction_list, fwdPath

    #h(n) = a heuristic function that estimate the cost of the cheapest path from start to goal   
    def manhattan(self, cell1, cell2):
        x1, y1 = cell1
        x2, y2 = cell2    
        return abs(x1-x2) + abs( y1-y2) 
    
    #Add a method to retrieve explored cells
    def get_explored(self):
        return self.explored

#---------------------------------------------------------------------------------------------------------------------
#Uninformed Search Strategies: DFS and BFS
#---------------------------------------------------------------------------------------------------------------------
#References: https://www.youtube.com/watch?v=sTRK9mQgYuc&t=0s (Depth First Search (DFS) in Python by Learning Orbis)
class DFS(Search):
    def __init__(self, my_map, initial_state, goal_states):
        super().__init__(my_map, initial_state, goal_states)

    def search(self):
        #add the initial state to both explored and frontier list
        explored = [self.initial_state]
        frontier = [self.initial_state]
        
        #create a dictionary for reversing path 
        dfsPath = {}   

        while frontier:
            current_state = frontier.pop()
            
            if self.check_goal_state(current_state):  #Check if current state is a goal state
                self.explored = explored                   
                return self.generate_output(explored,dfsPath)

            #Prioritize UP before LEFT before DOWN before RIGHT
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  #RIGHT, DOWN, LEFT, UP  (LIFO queue)
            neighbors = self.find_neighbor(current_state,directions)  #find neighbor of the current cell
            for neighbor in neighbors:
                if neighbor in self.my_map.path_list and neighbor not in explored:   #identify if it is a valid neighbor or not
                    #if yes, add it to the explored and frontier list
                    explored.append(neighbor)
                    frontier.append(neighbor)
                    
                    #add the first data to the dictionary 'neighbor' coordinate is the key and current state coordinate is the data
                    dfsPath[neighbor] = current_state
        
        #if no goal is found, return the length of the explored list            
        self.explored = explored                 
        return len(explored)

#---------------------------------------------------------------------------------------------------------------------
#References: https://www.youtube.com/watch?v=D14YK-0MtcQ&t=0s (Breadth First Search (BFS) in Python by Learning Orbis)
#Nearly the same with the DFS
class BFS(Search):
    def __init__(self, my_map, initial_state, goal_states):
        super().__init__(my_map, initial_state, goal_states)

    def search(self):
        explored = [self.initial_state]
        frontier = [self.initial_state]
        bfsPath = {}   

        while frontier:
            current_state = frontier.pop(0)  #difference point from the DFS, it pops the oldest value in the frontier list, not the newest value
            
            if self.check_goal_state(current_state):  #Check if current state is a goal state
                self.explored = explored                   
                return self.generate_output(explored,bfsPath)

            #Prioritize UP before LEFT before DOWN before RIGHT
            directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  #UP, LEFT, DOWN, RIGHT  (FIFO queue)
            neighbors = self.find_neighbor(current_state,directions)
            for neighbor in neighbors:
                if neighbor in self.my_map.path_list and neighbor not in explored:
                    explored.append(neighbor)
                    frontier.append(neighbor)
                    #add the first data to the dictionary 'neighbor' coordinate is the key and current state coordinate is the data
                    bfsPath[neighbor] = current_state
                
        self.explored = explored                   
        return len(explored)

#---------------------------------------------------------------------------------------------------------------------
#Informed Search Strategies: GBFS and AS
#---------------------------------------------------------------------------------------------------------------------
#References: https://www.youtube.com/watch?v=W9zSr9jnoqY (A-Star A* Search in Python by Learning Orbis)
class AS(Search):
    def __init__(self, my_map, initial_state, goal_states):
        super().__init__(my_map, initial_state, goal_states)

    def search(self):
        path_cost = 1
        start = self.initial_state
        explored = [start]  #Initialize the set of explored states with the initial state
        aPath = {}  #Initialize the dictionary to store the path

        #Iterate over each goal state
        for goal in self.goal_states:
            #f(n) = g(n) + h(n)
            #g(n) = cost of the path from start to goal
            #h(n) = heuristic function estimating the cost of the cheapest path from start to goal (I use Manhattan distance)
            
            #Initialize g_score and f_score dictionaries with infinite values for all cells
            g_score = {cell: float('inf') for cell in self.my_map.path_list}
            f_score = {cell: float('inf') for cell in self.my_map.path_list}
            
            #Set g_score of the start cell to 0 and calculate f_score using Manhattan distance heuristic
            g_score[start] = 0
            f_score[start] = self.manhattan(start, goal)
            
            #Create a priority queue and add the start cell with its f_score as priority
            open = PriorityQueue()
            open.put ((f_score[start], self.manhattan(start, goal), start))  #the format is: open.put ((f(n), h(n), cell))
        
            #Start A* search algorithm
            while open:
                #Get the cell with the lowest f_score from the priority queue
                current_state = open.get()[2]
                
                #Check if the current state is a goal state
                if self.check_goal_state(current_state):
                    self.explored = explored
                    return self.generate_output(explored,aPath)

                #Define the possible movement directions: UP, LEFT, DOWN, RIGHT
                directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
                
                #Find neighbors of the current state
                neighbors = self.find_neighbor(current_state, directions)
                valid_neighbor = False

                for neighbor in neighbors:
                    #Check if the neighbor is a valid path cell and not explored
                    if neighbor in self.my_map.path_list and neighbor not in explored:
                        explored.append(neighbor)
                        valid_neighbor = True
                        
                        #Calculate the temporary g_score and f_score for the neighbor
                        temp_g_score = g_score[current_state] + path_cost
                        temp_f_score = temp_g_score + self.manhattan(neighbor, goal)
                       
                        #Update g_score and f_score if the new scores are better
                        if temp_f_score < f_score[neighbor]:
                            g_score[neighbor] = temp_g_score
                            f_score[neighbor] = temp_f_score             
                            #Add the neighbor to the priority queue with its f_score as priority
                            open.put((f_score[neighbor], self.manhattan(neighbor, goal), neighbor))
                            #Update the path dictionary with the current state as the parent of the neighbor
                            aPath[neighbor] = current_state
                    
                if not valid_neighbor:
                    if open.empty():
                        break              
            self.explored = explored
            return len(explored)                            
#---------------------------------------------------------------------------------------------------------------------
class GBFS(Search):
    def __init__(self, my_map, initial_state, goal_states):
        super().__init__(my_map, initial_state, goal_states)

    def search(self):
        unvisited = {n: float('inf') for n in self.my_map.path_list}
        unvisited[self.initial_state] = 0
        explored = [self.initial_state]
        gbfsPath = {}

        while unvisited:
            current_state = min(unvisited, key=unvisited.get)

            #Check if the current state is a goal state
            if self.check_goal_state(current_state):
                self.explored = explored
                return self.generate_output(explored, gbfsPath)

            #Define the possible movement directions: UP, LEFT, DOWN, RIGHT
            directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

            #Find neighbors of the current state
            neighbors = self.find_neighbor(current_state, directions)
            for neighbor in neighbors:
                #Check if the neighbor is a valid path cell and not explored
                if neighbor in self.my_map.path_list and neighbor not in explored:
                    #Add the neighbor to the set of explored states         
                    tempDist = unvisited[current_state] + 1
                    if tempDist < unvisited[neighbor]:
                        explored.append(neighbor)  
                        unvisited[neighbor] = tempDist
                        gbfsPath[neighbor] = current_state     
                    else:
                        self.explored = explored
                        return len(explored)                          
            #Remove the current state from unvisited since it has been explored
            unvisited.pop(current_state)
#---------------------------------------------------------------------------------------------------------------------
#Custom Uninformed Search: Depth-limited Search
#Defines the Depth-Limited Search (DLS) algorithm.
#Explores nodes in DFS manner with a depth limit.
#Stops the search when a goal state is found, the depth limit is reached, or the frontier is empty.
#---------------------------------------------------------------------------------------------------------------------
class DLS(Search):
    def __init__(self, my_map, initial_state, goal_states):
        super().__init__(my_map, initial_state, goal_states)

    def search(self):
        explored = [self.initial_state]  #List to store explored cells
        frontier = [(self.initial_state,0)]  #Stack to maintain frontier with depth
        dlsPath = {}  #Dictionary to store path
        
        depth_limit = 1000  #Custom depth limit
        
        while frontier:
            current_state, depth = frontier.pop()  #Retrieve current state and depth from the stack
            
            if depth > depth_limit:  #Check if depth limit is reached
                continue
            
            if self.check_goal_state(current_state):  #Check if current state is a goal state
                self.explored = explored  #Store explored cells
                return self.generate_output(explored, dlsPath)  #Generate output
                
            #Prioritize UP before LEFT before DOWN before RIGHT
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  #Define movement directions
            neighbors = self.find_neighbor(current_state, directions)  #Find neighboring cells
            
            for neighbor in neighbors:
                if neighbor in self.my_map.path_list and neighbor not in explored:
                    explored.append(neighbor)  #Add neighbor to explored list
                    frontier.append((neighbor, depth + 1))  #Add neighbor to frontier with increased depth
                    dlsPath[neighbor] = current_state  #Update path dictionary
                    
        self.explored = explored  #Store explored cells
        return len(explored)  #Return number of explored cells

#---------------------------------------------------------------------------------------------------------------------
#Custom informed Search: Iterative Deepening A Star Search
#---------------------------------------------------------------------------------------------------------------------
class IDA(Search):
    def __init__(self, my_map, initial_state, goal_states):
        super().__init__(my_map, initial_state, goal_states)

    def search(self):
        path_cost = 1
        start = self.initial_state
        explored = [start] #Initialize the set of explored states with the initial state
        idaPath = {}  #Initialize the dictionary to store the path

        #Iterate over each goal state
        for goal in self.goal_states:
            #Set the initial threshold for this goal state
            threshold = self.manhattan(start, goal)
            while True:
                #Perform depth-bound iterative deepening search
                result, new_threshold = self.search_recursively(start, goal, path_cost, threshold, explored, idaPath)
                if result == 'FOUND':
                    #If a solution is found, return the generated output
                    self.explored = explored
                    return self.generate_output(explored, idaPath)
                if result == 'NOT_FOUND':
                    #If no solution found within the current threshold, break out of the loop
                    break
                threshold = new_threshold  #Update the threshold based on the returned new threshold

        #If no solution found for any goal state, return the total number of explored states
        self.explored = explored
        return len(explored)

    def search_recursively(self, current_state, goal, path_cost, threshold, explored, idaPath):
        #Calculate the heuristic function value (h(n))
        f_score = self.manhattan(current_state, goal)
        #If the heuristic value exceeds the current threshold, return "NOT_FOUND" and the heuristic value
        if f_score > threshold:
            return 'NOT_FOUND', f_score

        #Check if the current state is the goal state
        if self.check_goal_state(current_state):
            return 'FOUND', f_score

        #Define the possible movement directions: UP, LEFT, DOWN, RIGHT
        directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

        #Find neighbors of the current state
        neighbors = self.find_neighbor(current_state, directions)
        valid_neighbor = False

        for neighbor in neighbors:
            #Check if the neighbor is a valid path cell and not explored
            if neighbor in self.my_map.path_list and neighbor not in explored:
                explored.append(neighbor)
                valid_neighbor = True

                #Recursive call with the neighbor
                result, new_threshold = self.search_recursively(neighbor, goal, path_cost, threshold, explored, idaPath)
                if result == 'FOUND':
                    #If a solution is found, update the path and return "FOUND" along with the new threshold
                    idaPath[neighbor] = current_state
                    return 'FOUND', new_threshold
            

        if not valid_neighbor:
            #If no valid neighbors found, return "NOT_FOUND" with an infinite threshold
            return 'NOT_FOUND', float('inf')

        #All neighbors explored, no goal found at this depth, return "NOT_FOUND" with an infinite threshold
        return 'NOT_FOUND', float('inf')

#---------------------------------------------------------------------------------------------------------------------   
class Visualization:
    def __init__(self, method, my_map, path):
        #Initialize visualization parameters
        self.my_map = my_map
        self.method = method
        self.path = path
        self.cell_size = 50  #Size of each cell in pixels
        self.screen_width = my_map.grid_size[1] * self.cell_size  #Height of the screen in pixels
        self.screen_height = my_map.grid_size[0] * self.cell_size   #Width of the screen in pixels
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height+100))  #Initialize the screen
        self.drawn_flag = False  #Flag to track if the visualization has been drawn
        pygame.display.set_caption("Search Visualization")  #Set window title

    def draw_map(self):
        #Draw the map based on data from the text file
        for y in range(self.my_map.grid_size[0]):
            for x in range(self.my_map.grid_size[1]):
                cell_x = x * self.cell_size
                cell_y = y * self.cell_size

                #Draw walls
                if (x, y) in self.my_map.wall_list:
                    pygame.draw.rect(self.screen, (128, 128, 128), (cell_x + 2, cell_y + 2, self.cell_size - 2, self.cell_size - 2))
                #Draw paths
                elif (x, y) in self.my_map.path_list:
                    pygame.draw.rect(self.screen, (255, 255, 255), (cell_x + 2, cell_y + 2, self.cell_size - 2, self.cell_size - 2))
                #Draw initial state
                if (x, y) == self.my_map.initial_state:
                    pygame.draw.rect(self.screen, (255, 0, 0), (cell_x + 2, cell_y + 2, self.cell_size - 2, self.cell_size - 2))
                #Draw goal states
                if (x, y) in self.my_map.goal_states:
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_x + 2, cell_y + 2, self.cell_size - 2, self.cell_size - 2))
       
    def draw_process(self):
        #Draw the exploration process
        if not self.drawn_flag:
            explored = self.method.get_explored()  #Retrieve explored cells
            for cell in explored[1:]:  #the reason why I did not paint the last value in the explored list because it is the goal state, I want to keep it green.
                if cell not in self.my_map.goal_states:
                    cell_x = cell[0] * self.cell_size
                    cell_y = cell[1] * self.cell_size
                    #Draw explored cells
                    pygame.draw.rect(self.screen, (173, 216, 230), (cell_x + 2, cell_y + 2, self.cell_size - 2, self.cell_size - 2))
                    pygame.time.delay(50)  #Delay to visualize exploration process
                    pygame.display.flip()  #Update display
            
            if self.path is None:
                self.drawn_flag = True
            else:
                #Draw the final path if available
                for cell in self.path:
                    cell_x = cell[0] * self.cell_size
                    cell_y = cell[1] * self.cell_size
                    pygame.draw.rect(self.screen, (255, 255, 0), (cell_x + 2, cell_y + 2, self.cell_size - 2, self.cell_size - 2))
                    pygame.time.delay(50)  #Delay to visualize path drawing process
                    pygame.display.flip()  #Update display
                self.drawn_flag = True
        
    def draw_button(self):
        font = pygame.font.SysFont('sans', 20)
        text_BFS = font.render('BFS', True, (0,0,0))
        text_DFS = font.render('DFS', True, (0,0,0))
        text_AS = font.render('AS', True, (0,0,0))
        text_GBFS = font.render('GBFS', True, (0,0,0))
        text_DLS = font.render('DLS', True, (0,0,0))
        text_IDA = font.render('IDA', True, (0,0,0))

        #DFS button
        pygame.draw.rect(self.screen, (255, 255, 255), (10, self.screen_height + 10, 60, 50))
        self.screen.blit(text_DFS, (20, self.screen_height + 20))
        
        #BFS button
        pygame.draw.rect(self.screen, (255, 255, 255), (80, self.screen_height + 10, 60, 50))
        self.screen.blit(text_BFS, (90, self.screen_height + 20))
        
        #AS button
        pygame.draw.rect(self.screen, (255, 255, 255), (150, self.screen_height + 10, 60, 50))
        self.screen.blit(text_AS, (160, self.screen_height + 20))

        #GBFS button
        pygame.draw.rect(self.screen, (255, 255, 255), (220, self.screen_height + 10, 60, 50))
        self.screen.blit(text_GBFS, (230, self.screen_height + 20))
        
        #DLS button
        pygame.draw.rect(self.screen, (255, 255, 255), (290, self.screen_height + 10, 60, 50))
        self.screen.blit(text_DLS, (300, self.screen_height + 20))

        #IDA button
        pygame.draw.rect(self.screen, (255, 255, 255), (360, self.screen_height + 10, 60, 50))
        self.screen.blit(text_IDA, (370, self.screen_height + 20))
                  
    def run(self):
        pygame.init()  #Initialize pygame
        running = True
        self.screen.fill((0, 0, 0))  #Fill the screen with black
        
        while running:
            method = None  #Initialize method variable
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False  #Set running flag to False to exit the loop
                #Check if the mouse clicks on the buttons
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if self.screen_height + 10 <= mouse_y <= self.screen_height + 60:
                        #declare the start time when before the program starts to search
                        start_time = time.time()
                        if 10 <= mouse_x <= 70:  #DFS button
                            sys.argv[2] = 'DFS'
                            method = DFS(self.my_map, self.my_map.initial_state, self.my_map.goal_states) 
                            
                        elif 80 <= mouse_x <= 140:  #BFS button
                            sys.argv[2] = 'BFS'
                            method = BFS(self.my_map, self.my_map.initial_state, self.my_map.goal_states) 
                            
                        elif 150 <= mouse_x <= 210:  #AS button
                            sys.argv[2] = 'AS'
                            method = AS(self.my_map, self.my_map.initial_state, self.my_map.goal_states) 
                            
                        elif 220 <= mouse_x <= 280:  #GBFS button
                            sys.argv[2] = 'GBFS'
                            method = GBFS(self.my_map, self.my_map.initial_state, self.my_map.goal_states) 
                            
                        elif 290 <= mouse_x <= 350:  #DLS button
                            sys.argv[2] = 'DLS'
                            method = DLS(self.my_map, self.my_map.initial_state, self.my_map.goal_states) 
                            
                        elif 360 <= mouse_x <= 420:  #IDA button
                            sys.argv[2] = 'IDA'
                            method = IDA(self.my_map, self.my_map.initial_state, self.my_map.goal_states) 
                        
            #THIS IS FOR THE GUI, starting from the second run (first run is received data from the command line)
            #Work the same as the main class
            if method is not None:
                result = method.search()
                
                print('\n',sys.argv[1], sys.argv[2])  #Print the filename and algorithm
                #Display the search result
                if type(result) == tuple:
                    #If a path to the goal is found, print the number of nodes expanded and the path
                    print('<Node {}> {}\n{}'.format(result[0], result[1], result[2]))
                    #Create a visualization object with the search result
                    visualization = Visualization(method, self.my_map, result[3])
                if type(result) == int:
                    #If no goal is reachable, print the message
                    print('No goal is reachable; {}'.format(result))
                    visualization = Visualization(method, self.my_map, None)
                    
                #print out the execution time of the algorithm
                print("Execution time: %s seconds" %(time.time() - start_time))
                visualization.run()
             
            self.draw_map()  #Draw the map
            self.draw_button()  #Draw the button           
            self.draw_process()  #Draw the exploration process and final path
                
        pygame.quit()  #Quit pygame when done 
        sys.exit() #terminate the program

#---------------------------------------------------------------------------------------------------------------------        
class Main:
    #Check if the correct number of command line arguments is provided
    if len(sys.argv) != 3:
        print('-----------------------------------------------------------------------------------------------------------------------\n')
        print('Error: You need to use the following format "python search.py [map_file.txt] [Search algorithm]". Please check the command.\n')
        print('-----------------------------------------------------------------------------------------------------------------------')
        sys.exit()  #Exit the program if the format is incorrect
    else:
        #Extract the filename and algorithm from command line arguments
        filename = sys.argv[1]
        
        #Check if the file exists
        if not os.path.exists(filename):
            print('-----------------------------------------------------------------------------------------------------------------------\n')
            print('Error: The map file does not exist. Please check the file path.\n')
            print('-----------------------------------------------------------------------------------------------------------------------')
            sys.exit()  #Exit the program if the file does not exist
    
        algorithm = sys.argv[2]  #Get the search algorithm
        
        print(sys.argv[1], sys.argv[2])  #Print the filename and algorithm
        
        try:
            #Read data from the file
            with open(filename) as file:
                data = file.readlines() 
                grid_size = eval(data[0])  #Extract grid size from the file
                initial_state = eval(data[1])  #Extract initial state from the file
                #Extract goal states from the file and convert them to a list of tuples
                goal_states = [eval(coordinates) for coordinates in data[2].split('|')]  
                #Extract wall coordinates from the file and convert them to a list of tuples
                walls = [eval(wall) for wall in data[3:]]  
        except:
            #Handle file reading errors
            print('-----------------------------------------------------------------------------------------------------------------------\n')
            print('Error: Cannot read the file. Please check the file content.\n')
            print('-----------------------------------------------------------------------------------------------------------------------')
            sys.exit()  #Exit the program if there's an error reading the file
            
        #Create a map object with the extracted data
        my_map = MyMap(grid_size, walls, initial_state, goal_states)
        
        #Choose the appropriate search algorithm based on the input
        if algorithm == 'BFS' or algorithm == 'bfs':
            method = BFS(my_map, initial_state, goal_states) 
            
        elif algorithm == 'DFS' or algorithm == 'dfs':
            method = DFS(my_map, initial_state, goal_states)   
             
        elif algorithm == 'AS' or algorithm == 'as':
            method = AS(my_map, initial_state, goal_states)     
              
        elif algorithm == 'GBFS' or algorithm == 'gbfs':
            method = GBFS(my_map, initial_state, goal_states)
            
        elif algorithm == 'DLS' or algorithm == 'dls':
            method = DLS(my_map, initial_state, goal_states)
            
        elif algorithm == 'IDA' or algorithm == 'ida':
            method = IDA(my_map, initial_state, goal_states)
        else:
            #Handle incorrect algorithm input
            print('-----------------------------------------------------------------------------------------------------------------------\n')
            print('Error: Wrong search algorithm input. Please check the command. \n')
            print('-----------------------------------------------------------------------------------------------------------------------')
            sys.exit()  #Exit the program if the algorithm input is incorrect
            
        #declare the start time when before the program starts to search
        start_time = time.time()
        #Perform the search using the selected algorithm
        result = method.search()
        
        #Display the search result
        if type(result) == tuple:
            #If a path to the goal is found, print the number of nodes expanded and the path
            print('<Node {}> {}\n{}'.format(result[0], result[1], result[2]))
            #Create a visualization object with the search result
            visualization = Visualization(method, my_map, result[3])
        if type(result) == int:
            #If no goal is reachable, print the message
            print('No goal is reachable; {}'.format(result))
            #Create a visualization object without a path (None)
            visualization = Visualization(method, my_map, None)
        
        #print out the execution time of the algorithm
        print("Execution time: %s seconds" %(time.time() - start_time))
        #Run the visualization
        visualization.run()