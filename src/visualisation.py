import pygame
import numpy as np
import random
import os
import time

class MazeVisualizer:
    def __init__(self, maze, cell_size=None, window_size=600):
        pygame.init()
        self.maze = maze
        
        # Calculate cell size based on maze dimensions to fit the window
        if cell_size is None:
            # Make sure cells are small enough to fit on most screens
            self.cell_size = min(window_size // maze.size, 30)
        else:
            self.cell_size = cell_size
            
        # Calculate total dimensions based on cell size
        self.width = min(maze.grid.shape[1] * self.cell_size, 800)
        self.height = min(maze.grid.shape[0] * self.cell_size, 600)
        
        # UI Configuration
        self.ui_height = 100
        self.total_height = self.height + self.ui_height
        
        # Force a reasonable window size
        self.screen = pygame.display.set_mode((self.width, self.total_height), pygame.RESIZABLE)
        pygame.display.set_caption("Cognitive Evolution Maze")
        
        # Colors
        self.colors = {
            'wall': (40, 40, 40),  # Dark gray for walls
            'path': (240, 240, 240),  # Light gray for paths
            'start': (0, 200, 0),  # Green for start
            'goal': (200, 0, 0),  # Red for goal
            'ui_bg': (45, 45, 45),  # Dark background for UI
            'button': (80, 80, 80),  # Gray for buttons
            'slider': (100, 100, 100),  # Light gray for sliders
            'text': (255, 255, 255)  # White text
        }
        
        # Create a color palette for different agents
        self.agent_colors = []
        for i in range(50):  # Support up to 50 different agent colors
            hue = (i * 25) % 360  # Spread colors evenly
            # Convert HSV to RGB (simplified)
            if hue < 60:
                r, g, b = 255, int(hue * 4.25), 0
            elif hue < 120:
                r, g, b = int((120 - hue) * 4.25), 255, 0
            elif hue < 180:
                r, g, b = 0, 255, int((hue - 120) * 4.25)
            elif hue < 240:
                r, g, b = 0, int((240 - hue) * 4.25), 255
            elif hue < 300:
                r, g, b = int((hue - 240) * 4.25), 0, 255
            else:
                r, g, b = 255, 0, int((360 - hue) * 4.25)
            self.agent_colors.append((r, g, b))
        
        self.font = pygame.font.SysFont(None, 24)
    
        # UI State - start with a slower speed
        self.simulation_speed = 0.2  # Much slower default speed
        self.max_speed = 3.0  # Maximum speed multiplier
        self.current_generation = 0
        self.viewing_generation = 0  # The generation currently being viewed
        self.history_mode = False    # Whether we're looking at history or live
        self.paused = False
        self.population_size = 30  # Increased default population
        
        # Buttons - generate them in draw_ui instead
        self.buttons = []
        
        # Track best agent fitness
        self.best_fitness = 0.0
        self.all_agents = []
        
        # Timer display
        self.max_steps = maze.size * maze.size * 2
        self.current_step = 0
        self.time_percentage = 100  # Percentage of time remaining
        
        # Neural network visualization
        self.selected_agent_idx = None
        self.nn_vis_active = False  # Whether the neural network visualization is active
        self.nn_vis_buttons = []    # Buttons for the neural network visualization
        
        # Create directory for saving visualizations and agents
        self.save_dir = "saved_agents"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def draw_maze(self):
        # Draw the maze grid
        for i in range(self.maze.grid.shape[0]):
            for j in range(self.maze.grid.shape[1]):
                rect = pygame.Rect(j*self.cell_size, i*self.cell_size,
                               self.cell_size, self.cell_size)
                if self.maze.grid[i,j] == 1:
                    pygame.draw.rect(self.screen, self.colors['wall'], rect)
                else:
                    pygame.draw.rect(self.screen, self.colors['path'], rect)
                    
                # Draw grid lines for clarity if cells are big enough
                if self.cell_size > 5:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
                    
        # Draw start and goal
        start_i, start_j = self.maze.start
        goal_i, goal_j = self.maze.goal
        
        # Calculate centers
        start_center = (start_j * self.cell_size + self.cell_size//2, 
                      start_i * self.cell_size + self.cell_size//2)
        goal_center = (goal_j * self.cell_size + self.cell_size//2, 
                     goal_i * self.cell_size + self.cell_size//2)
        
        # Draw start and goal markers
        radius = max(3, self.cell_size//4)
        pygame.draw.circle(self.screen, self.colors['start'], 
                         start_center, radius)
        pygame.draw.circle(self.screen, self.colors['goal'],
                         goal_center, radius)
    
    def draw_agents(self, agent_positions):
        """Draw multiple agents with different colors"""
        for idx, position in enumerate(agent_positions):
            if position:  # Skip None positions
                i, j = position
                x = j * self.cell_size + self.cell_size//2
                y = i * self.cell_size + self.cell_size//2
                
                # Use a color from our palette, cycling if needed
                agent_color = self.agent_colors[idx % len(self.agent_colors)]
                
                # Make the agents smaller
                radius = max(2, self.cell_size//6)
                
                # Highlight selected agent if any
                if idx == self.selected_agent_idx:
                    # Draw a highlight around the selected agent
                    pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius + 2)
                
                pygame.draw.circle(self.screen, agent_color, (x, y), radius)
                
                # Add small agent index if there's enough space
                if self.cell_size > 15 and len(agent_positions) <= 50:
                    # Only show agent numbers if not too many
                    small_font = pygame.font.SysFont(None, max(10, self.cell_size // 3))
                    idx_text = small_font.render(str(idx), True, (255, 255, 255))
                    self.screen.blit(idx_text, (x - idx_text.get_width() // 2, 
                                             y - idx_text.get_height() // 2))
    
    def update(self, agent_positions, population_size):
        """Update the visualization with multiple agent positions"""
        self.screen.fill((255, 255, 255))
        
        # If neural network visualization is active, draw that instead
        if self.nn_vis_active and self.selected_agent_idx is not None and self.all_agents:
            self.draw_neural_network_visualization()
        else:
            # Normal maze visualization
            self.draw_maze()
            self.draw_agents(agent_positions)
            
        self.draw_ui(population_size)
        pygame.display.flip()
        
    def draw_ui(self, population_size):
        # Clear the buttons list
        self.buttons = []
        
        # UI Background
        ui_rect = pygame.Rect(0, self.height, self.width, self.ui_height)
        pygame.draw.rect(self.screen, self.colors['ui_bg'], ui_rect)
        
        # Scale UI elements based on window width
        button_width = min(40, self.width // 20)
        button_height = min(30, self.ui_height // 3)
        button_spacing = min(20, self.width // 40)
        
        # Generation Navigation
        gen_nav_btn1 = self.draw_button("<", button_spacing, 
                                     self.height + 20, button_width, button_height)
        gen_nav_btn2 = self.draw_button(">", 2*button_spacing + button_width, 
                                     self.height + 20, button_width, button_height)
        
        # Add generation text - highlight if viewing historical generation
        gen_text = f"Gen: {self.current_generation}"
        text_color = (255, 200, 0) if self.viewing_generation != self.current_generation else self.colors['text']
        
        # Draw text with different color if viewing history
        gen_text_surf = pygame.font.SysFont(None, 24).render(gen_text, True, text_color)
        self.screen.blit(gen_text_surf, (3*button_spacing + 2*button_width, self.height + 25))
        
        # Add "Return to Current" button if viewing history
        if self.viewing_generation != self.current_generation:
            return_btn = self.draw_button("⟳", 3*button_spacing + 2*button_width + 100, 
                                      self.height + 20, button_width, button_height)
            self.buttons.append({"text": "⟳", "rect": return_btn})
        
        # Speed Control
        self.draw_text("Speed:", button_spacing, self.height + 60)
        slider_width = min(150, self.width // 6)
        self.draw_slider(3*button_spacing, self.height + 65, 
                       slider_width, 10, self.simulation_speed/self.max_speed)
        
        # Population Control
        pop_btn1 = self.draw_button("-", self.width - 2*button_spacing - 2*button_width, 
                                 self.height + 20, button_width, button_height)
        pop_btn2 = self.draw_button("+", self.width - button_spacing - button_width, 
                                 self.height + 20, button_width, button_height)
        self.draw_text(f"Pop: {population_size}", 
                     self.width - 3*button_spacing - 3*button_width, self.height + 25)
        
        # Neural Network Visualization button
        if self.nn_vis_active:
            nn_vis_btn_text = "Back"
        else:
            nn_vis_btn_text = "NN Vis"
            
        nn_vis_btn = self.draw_button(nn_vis_btn_text, 
                                    self.width - 4*button_spacing - 4*button_width, 
                                    self.height + 60, 
                                    button_width*1.5, button_height)
        
        # Pause Button at center
        # Use "II" (pause symbol) when NOT paused and "▶" (play symbol) when paused
        pause_btn_text = "II" if not self.paused else "▶"
        pause_btn = self.draw_button(pause_btn_text, self.width//2 - button_width//2, 
                                   self.height + 20, button_width, button_height)
        
        # Best fitness display
        self.draw_text(f"Best: {self.best_fitness:.2f}", 
                     self.width//2 - 40, self.height + 60)
                     
        # Draw timer bar
        time_remaining_str = f"Time: {self.time_percentage}%"
        self.draw_text(time_remaining_str, 
                      self.width - 120, self.height + 60)
                      
        # Draw progress bar showing remaining time
        bar_width = min(150, self.width // 5)
        bar_height = 8
        bar_x = self.width - bar_width - button_spacing
        bar_y = self.height + 75
        
        # Background (empty) bar
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        pygame.Rect(bar_x, bar_y, bar_width, bar_height), 
                        border_radius=4)
        
        # Foreground (filled) bar - colored by time remaining
        if self.time_percentage > 70:
            bar_color = (0, 200, 0)  # Green
        elif self.time_percentage > 30:
            bar_color = (200, 200, 0)  # Yellow
        else:
            bar_color = (200, 0, 0)  # Red
            
        fill_width = int(bar_width * self.time_percentage / 100)
        pygame.draw.rect(self.screen, bar_color, 
                        pygame.Rect(bar_x, bar_y, fill_width, bar_height), 
                        border_radius=4)
        
        # Store buttons as proper button objects with rectangles
        self.buttons = [
            {"text": "<", "rect": gen_nav_btn1},
            {"text": ">", "rect": gen_nav_btn2},
            {"text": "-", "rect": pop_btn1},
            {"text": "+", "rect": pop_btn2},
            {"text": pause_btn_text, "rect": pause_btn},
            {"text": nn_vis_btn_text, "rect": nn_vis_btn}
        ] + ([{"text": "⟳", "rect": return_btn}] if self.viewing_generation != self.current_generation else [])
    
    def draw_neural_network_visualization(self):
        """Draw the neural network visualization screen"""
        # Clear the screen
        self.screen.fill((30, 30, 30))
        
        # Clear the neural network buttons
        self.nn_vis_buttons = []
        
        # Add message if no agent is selected
        if self.selected_agent_idx is None or not self.all_agents:
            self.draw_text("Select an agent to visualize its neural network", 
                         self.width//2 - 150, self.height//2)
            return
        
        # Get the selected agent
        if 0 <= self.selected_agent_idx < len(self.all_agents):
            agent = self.all_agents[self.selected_agent_idx]
            
            # Display agent information
            agent_info = f"Agent #{self.selected_agent_idx} - Fitness: {agent.fitness:.3f}"
            if agent.reached_goal:
                agent_info += f" - Reached goal in {agent.steps_taken} steps"
                
            self.draw_text(agent_info, 20, 20, color=(255, 255, 255), font_size=28)
            
            # Display neural network architecture info
            arch_info = f"Network: {agent.input_size} inputs → {agent.hidden_size} hidden → {agent.output_size} outputs"
            self.draw_text(arch_info, 20, 60, color=(200, 200, 200))
            
            # Add a save button
            save_btn_w, save_btn_h = 100, 40
            save_btn = self.draw_button("Save Agent", 
                                     self.width - save_btn_w - 20, 20, 
                                     save_btn_w, save_btn_h)
            self.nn_vis_buttons.append({"text": "Save Agent", "rect": save_btn})
            
            # Indicate that visualization is not shown directly
            self.draw_text("The neural network visualization will be saved as a PNG file", 
                         20, 100, color=(200, 200, 200))
            
            # Add instructions for saving and viewing
            instructions = [
                "Click 'Save Agent' to save this agent's neural network visualization",
                f"Files will be saved to the '{self.save_dir}' directory",
                "The visualization includes weights and biases for each layer"
            ]
            
            for i, text in enumerate(instructions):
                self.draw_text(text, 20, 140 + i*30, color=(180, 180, 180))
                
            # Display extra controls
            next_agent_btn = self.draw_button("Next Agent", 
                                          20, self.height - 60, 120, 40)
            prev_agent_btn = self.draw_button("Prev Agent", 
                                          150, self.height - 60, 120, 40)
                                          
            self.nn_vis_buttons.extend([
                {"text": "Next Agent", "rect": next_agent_btn},
                {"text": "Prev Agent", "rect": prev_agent_btn}
            ])

    def save_selected_agent(self):
        """Save the selected agent's visualization and network"""
        if self.selected_agent_idx is None or not self.all_agents:
            return
            
        if 0 <= self.selected_agent_idx < len(self.all_agents):
            agent = self.all_agents[self.selected_agent_idx]
            
            # Create timestamp for filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename_base = f"agent_{self.viewing_generation}_{self.selected_agent_idx}_{timestamp}"
            
            # Save visualization
            vis_path = os.path.join(self.save_dir, f"{filename_base}_network.png")
            agent.visualize_network(save_path=vis_path)
            
            # Save agent data
            json_path = os.path.join(self.save_dir, f"{filename_base}_data.json")
            agent.save_to_file(json_path)
            
            print(f"Saved agent #{self.selected_agent_idx} to {vis_path} and {json_path}")
            return vis_path, json_path
            
        return None, None

    # The rest of the methods remain the same
    def draw_button(self, text, x, y, w, h):
        btn_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.colors['button'], btn_rect, border_radius=5)
        text_surf = pygame.font.SysFont(None, 24).render(text, True, self.colors['text'])
        self.screen.blit(text_surf, (x + w//2 - text_surf.get_width()//2, 
                                   y + h//2 - text_surf.get_height()//2))
        return btn_rect

    def draw_slider(self, x, y, w, h, value):
        # Slider track
        track_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.colors['slider'], track_rect, border_radius=5)
        
        # Slider thumb
        thumb_x = x + int(value * w)
        pygame.draw.circle(self.screen, (200, 200, 200), (thumb_x, y + h//2), h)

    def draw_text(self, text, x, y, color=None, font_size=24):
        if color is None:
            color = self.colors['text']
        text_surf = pygame.font.SysFont(None, font_size).render(text, True, color)
        self.screen.blit(text_surf, (x, y))

    def check_button_click(self, text, mouse_x, mouse_y):
        # Check if a button with the given text was clicked
        for btn in self.buttons:
            if btn["text"] == text:
                if btn["rect"].collidepoint(mouse_x, mouse_y):
                    return True
        return False

    def update_slider(self, mouse_x):
        # Find slider position in UI
        slider_x = 3*min(20, self.width // 40)  # Match draw_slider position
        slider_w = min(150, self.width // 6)
        # Convert mouse position to slider value
        raw_value = (mouse_x - slider_x) / slider_w
        self.simulation_speed = max(0.1, min(self.max_speed, raw_value * self.max_speed))    
        
    def quit(self):
        pygame.quit()
        
    # Handle all UI interactions
    def handle_events(self):
        """Process all pygame events and return if the simulation should continue"""
        running = True
        new_pop_size = self.population_size
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                
                # Handle clicks in the maze area (agent selection)
                if not self.nn_vis_active and y < self.height:
                    # Convert click coordinates to maze coordinates
                    maze_j = x // self.cell_size
                    maze_i = y // self.cell_size
                    
                    # Check if there's an agent at this position
                    if self.all_agents:
                        for idx, agent in enumerate(self.all_agents):
                            if agent.position and agent.position[0] == maze_i and agent.position[1] == maze_j:
                                # Select this agent
                                self.selected_agent_idx = idx
                                print(f"Selected agent #{idx} at position {agent.position}")
                                break
                                
                # Handle NN visualization mode buttons
                if self.nn_vis_active:
                    for btn in self.nn_vis_buttons:
                        if btn["rect"].collidepoint(x, y):
                            if btn["text"] == "Save Agent":
                                self.save_selected_agent()
                            elif btn["text"] == "Next Agent" and self.all_agents:
                                self.selected_agent_idx = (self.selected_agent_idx + 1) % len(self.all_agents)
                            elif btn["text"] == "Prev Agent" and self.all_agents:
                                self.selected_agent_idx = (self.selected_agent_idx - 1) % len(self.all_agents)
                
                # Check main UI button clicks 
                for btn in self.buttons:
                    if btn["rect"].collidepoint(x, y):
                        if btn["text"] == "<":  # Previous gen
                            if self.viewing_generation > 0:
                                self.viewing_generation -= 1
                                print(f"Viewing generation {self.viewing_generation}")
                        elif btn["text"] == ">":  # Next gen
                            if self.viewing_generation < self.current_generation:
                                self.viewing_generation += 1
                                print(f"Viewing generation {self.viewing_generation}")
                        elif btn["text"] == "⟳":  # Return to current
                            self.viewing_generation = self.current_generation
                            print(f"Returned to current generation {self.current_generation}")
                        elif btn["text"] == "-":  # Decrease population
                            new_pop_size = max(10, self.population_size-10)
                        elif btn["text"] == "+":  # Increase population
                            new_pop_size = min(100, self.population_size + 10)
                        elif btn["text"] == "NN Vis":  # Toggle NN visualization
                            self.nn_vis_active = True
                            print("Neural network visualization mode activated")
                        elif btn["text"] == "Back":  # Return from NN visualization
                            self.nn_vis_active = False
                            print("Returned to maze view")
                        elif btn["text"] in ["II", "▶"]:  # Pause/resume
                            self.paused = not self.paused
                            print(f"Pause state toggled: {'Paused' if self.paused else 'Running'}")
                            # Force redraw immediately to show updated button
                            pygame.display.flip()
                            
                # Handle slider drag
                slider_y = self.height + 65
                slider_x = 3*min(20, self.width // 40)  # Match draw_slider position
                slider_width = min(150, self.width // 6)
                if (slider_x-5 <= x <= slider_x+slider_width+5) and (slider_y-10 <= y <= slider_y+10):
                    self.update_slider(x)
                    
            if event.type == pygame.VIDEORESIZE:
                # Handle window resize
                self.width, self.total_height = event.w, event.h
                self.height = self.total_height - self.ui_height
                self.screen = pygame.display.set_mode((self.width, self.total_height), pygame.RESIZABLE)

        return running, new_pop_size