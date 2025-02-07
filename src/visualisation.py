import pygame
import numpy as np

class MazeVisualizer:
    def __init__(self, maze, cell_size=80):
        pygame.init()
        self.maze = maze
        self.cell_size = cell_size
        self.width = maze.grid.shape[1] * cell_size
        self.height = maze.grid.shape[0] * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Cognitive Evolution Maze")
        
        # Colors
        self.colors = {
            'wall': (38, 38, 38),
            'path': (242, 242, 242),
            'agent': (255, 0, 100),
            'goal': (0, 200, 100),
            'start': (100, 100, 255)
        }
        
    def draw_maze(self):
        for i in range(self.maze.grid.shape[0]):
            for j in range(self.maze.grid.shape[1]):
                rect = pygame.Rect(j*self.cell_size, i*self.cell_size,
                                 self.cell_size, self.cell_size)
                if self.maze.grid[i,j] == 1:
                    pygame.draw.rect(self.screen, self.colors['wall'], rect)
                else:
                    pygame.draw.rect(self.screen, self.colors['path'], rect)
                    
                # Draw start/goal
                if (i,j) == self.maze.start:
                    pygame.draw.circle(self.screen, self.colors['start'], 
                                     rect.center, self.cell_size//6)
                elif (i,j) == self.maze.goal:
                    pygame.draw.circle(self.screen, self.colors['goal'],
                                     rect.center, self.cell_size//6)
    
    def draw_agent(self, position):
        i,j = position
        x = j * self.cell_size + self.cell_size//2
        y = i * self.cell_size + self.cell_size//2
        pygame.draw.circle(self.screen, self.colors['agent'], (x,y), self.cell_size//8)
    
    def update(self, agent_position):
        self.screen.fill((255, 255, 255))
        self.draw_maze()
        self.draw_agent(agent_position)
        pygame.display.flip()
        
    def quit(self):
        pygame.quit()