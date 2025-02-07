import numpy as np

class Maze:
    def __init__(self):
        self.grid = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ])
        self.start = (1, 1)
        self.goal = (3, 3)
        
    def get_observation(self, position):
        return {
            'current_pos': np.array(position, dtype=np.float32),
            'goal_pos': np.array(self.goal, dtype=np.float32),
            'walls': self._get_nearby_walls(position).astype(np.float32)
        }
    
    def _get_nearby_walls(self, pos):
        return np.array([self.grid[pos[0]+di, pos[1]+dj] 
                       for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]])