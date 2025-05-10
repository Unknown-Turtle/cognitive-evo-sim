import os
import time
import sys
import math
import random
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QMessageBox,
    QDialog, QListWidget, QListWidgetItem, QDialogButtonBox,
    QInputDialog
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from datetime import datetime

from src.maze import CellType
from src.evolution import MazeEvolution
from src.utils import (
    load_genome, plot_fitness_history, visualize_network, 
    plot_network_complexity, plot_solution_diversity  # Import all utility functions
)

class EvolutionThread(QThread):
    """Thread for running the evolution without blocking the GUI."""
    update_signal = pyqtSignal(dict)
    generation_completed_signal = pyqtSignal()
    
    def __init__(self, evolution):
        """Initialize the evolution thread.

        Args:
            evolution: The MazeEvolution instance to run.
        """
        super().__init__()
        self.evolution = evolution
        self.running = False
        self.paused = False
        self.speed = 5.0  # Default to 5x speed
        self.simulate_one_generation = False
    
    def run(self):
        """Run the evolution loop."""
        self.running = True
        while self.running:
            if not self.paused:
                if self.simulate_one_generation:
                    # Run one full generation
                    stats = self.evolution.run_generation()
                    
                    # Emit update signal with stats
                    self.update_signal.emit(stats)
                    
                    # Emit generation completed signal
                    self.generation_completed_signal.emit()
                    
                    # Reset the flag
                    self.simulate_one_generation = False
                    
                    # Pause after completing one generation
                    self.paused = True
                else:
                    # Just wait for the next command
                    time.sleep(0.1)
            else:
                # Sleep when paused
                time.sleep(0.1)
    
    def simulate_generation(self):
        """Simulate one full generation."""
        self.simulate_one_generation = True
        self.paused = False
    
    def stop(self):
        """Stop the evolution."""
        self.running = False
        self.wait()

class MazeVisualization(QWidget):
    """Widget for visualizing the maze and agents."""
    
    def __init__(self, parent=None):
        """Initialize the maze visualization widget.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.evolution = None
        self.cell_size = 20
        self.show_all_agents = False
        self.agent_colors = [
            QColor(255, 0, 0),      # Red (best agent)
            QColor(0, 0, 255),      # Blue
            QColor(0, 255, 255),    # Cyan
            QColor(255, 0, 255),    # Magenta
            QColor(255, 255, 0)     # Yellow (worst agent)
        ]
        
        # Visualization settings
        self.selected_agents = []
        self.current_step = 0
        self.total_steps = 0
        self.simulation_running = False
    
    def set_evolution(self, evolution):
        """Set the evolution to visualize."""
        self.evolution = evolution
        # Update widget size
        maze_width = evolution.maze.width * self.cell_size
        maze_height = evolution.maze.height * self.cell_size
        self.setMinimumSize(maze_width, maze_height)
        self.update()
    
    def prepare_visualization(self):
        """Prepare for visualizing a generation with selected agents."""
        if not self.evolution or not self.evolution.agents:
            return
            
        # Sort agents by fitness
        sorted_agents = sorted(
            [(genome_id, agent) for genome_id, agent in self.evolution.agents],
            key=lambda x: x[1].calculate_fitness(self.evolution.steps_per_generation),
            reverse=True
        )
        
        # Select 5 agents: best, worst, and 3 in between
        num_agents = len(sorted_agents)
        if num_agents >= 5:
            indices = [0, num_agents // 4, num_agents // 2, 3 * num_agents // 4, num_agents - 1]
            self.selected_agents = [sorted_agents[i][1] for i in indices]
        else:
            # If less than 5 agents, use all available
            self.selected_agents = [agent for _, agent in sorted_agents]
        
        # Reset visualization state
        self.current_step = 0
        self.total_steps = self.evolution.steps_per_generation
        self.simulation_running = False
        
        # Reset agent positions
        for agent in self.selected_agents:
            agent.x = agent.initial_x
            agent.y = agent.initial_y
            agent.steps_taken = 0
            agent.reached_goal = False
        
        self.update()
    
    def step_visualization(self):
        """Advance the visualization by one step."""
        if not self.simulation_running or not self.selected_agents:
            return False
            
        # Move each agent
        for agent in self.selected_agents:
            if not agent.reached_goal:
                agent.move(self.evolution.maze)
        
        # Increment step counter
        self.current_step += 1
        
        # Check if we've reached the end
        if self.current_step >= self.total_steps:
            self.simulation_running = False
            return False
            
        self.update()
        return True
    
    def start_visualization(self):
        """Start the visualization."""
        self.simulation_running = True
    
    def paintEvent(self, event):
        """Paint the maze and agents."""
        if not self.evolution:
            return
        
        painter = QPainter(self)
        
        # Draw maze
        maze = self.evolution.maze
        for y in range(maze.height):
            for x in range(maze.width):
                cell_type = maze.get_cell(x, y)
                
                # Set color based on cell type
                if cell_type == CellType.WALL:
                    color = QColor(0, 0, 0)  # Black
                elif cell_type == CellType.EMPTY:
                    color = QColor(200, 200, 200)  # Light gray
                elif cell_type == CellType.SPAWN:
                    color = QColor(0, 255, 0)  # Green
                elif cell_type == CellType.GOAL:
                    color = QColor(255, 215, 0)  # Gold
                
                # Draw the cell
                painter.fillRect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size, 
                    color
                )
        
        # Draw selected agents
        if self.selected_agents and self.simulation_running:
            for i, agent in enumerate(self.selected_agents):
                color = self.agent_colors[i % len(self.agent_colors)]
                painter.setPen(color)
                painter.setBrush(color)
                painter.drawEllipse(
                    agent.x * self.cell_size + self.cell_size // 4,
                    agent.y * self.cell_size + self.cell_size // 4,
                    self.cell_size // 2,
                    self.cell_size // 2
                )
        
        #(for debugging)
        elif self.show_all_agents and self.evolution.agents:
            for i, (genome_id, agent) in enumerate(self.evolution.agents):
                color = self.agent_colors[i % len(self.agent_colors)]
                painter.setPen(color)
                painter.setBrush(color)
                painter.drawEllipse(
                    agent.x * self.cell_size + self.cell_size // 4,
                    agent.y * self.cell_size + self.cell_size // 4,
                    self.cell_size // 2,
                    self.cell_size // 2
                )
                    
        # Finish painting
        painter.end()

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize the main application window and set up the UI components."""
        super().__init__()
        self.setWindowTitle("NEAT Maze Evolution")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Create evolution
        self.evolution = MazeEvolution()
        
        # Create main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create layout
        main_layout = QHBoxLayout(main_widget)
        
        # Create maze visualization
        self.maze_viz = MazeVisualization()
        self.maze_viz.set_evolution(self.evolution)
        main_layout.addWidget(self.maze_viz, 3)  # 3/4 of the width
        
        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, 1)  # 1/4 of the width
        
        # Add control buttons
        self.sim_button = QPushButton("Simulate Next Generation")
        self.sim_button.clicked.connect(self.simulate_generation)
        control_layout.addWidget(self.sim_button)
        
        self.vis_button = QPushButton("Visualize Agents")
        self.vis_button.clicked.connect(self.start_visualization)
        self.vis_button.setEnabled(False)
        control_layout.addWidget(self.vis_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_evolution)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        # Add step counter
        self.step_layout = QHBoxLayout()
        self.step_layout.addWidget(QLabel("Steps:"))
        self.step_counter = QLabel("0/0")
        self.step_layout.addWidget(self.step_counter)
        control_layout.addLayout(self.step_layout)
        
        # Add speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "2x", "5x", "10x"])
        self.speed_combo.setCurrentIndex(3)  # Default to 5x
        self.speed_combo.currentIndexChanged.connect(self.set_speed)
        speed_layout.addWidget(self.speed_combo)
        control_layout.addLayout(speed_layout)
        
        # Add toggle for showing all agents
        self.show_all_agents_button = QPushButton("Show All Agents")
        self.show_all_agents_button.setCheckable(True)
        self.show_all_agents_button.toggled.connect(self.toggle_show_all_agents)
        control_layout.addWidget(self.show_all_agents_button)
        
        # Add stats display
        self.stats_layout = QVBoxLayout()
        self.generation_label = QLabel("Generation: 0")
        self.fitness_label = QLabel("Best Fitness: 0.0")
        self.avg_fitness_label = QLabel("Avg Fitness: 0.0")
        self.species_label = QLabel("Species: 0")
        self.completion_label = QLabel("Completion Rate: 0%")
        
        self.stats_layout.addWidget(self.generation_label)
        self.stats_layout.addWidget(self.fitness_label)
        self.stats_layout.addWidget(self.avg_fitness_label)
        self.stats_layout.addWidget(self.species_label)
        self.stats_layout.addWidget(self.completion_label)
        control_layout.addLayout(self.stats_layout)
        
        # Add Maze Selection Dropdown
        maze_layout = QHBoxLayout()
        maze_layout.addWidget(QLabel("Maze Type:"))
        self.maze_combo = QComboBox()
        self.maze_combo.addItems(["L-Shape", "U-Shape Maze", "C-Shape Maze"])
        self.maze_combo.currentIndexChanged.connect(self.change_maze)
        maze_layout.addWidget(self.maze_combo)
        control_layout.addLayout(maze_layout)
        
        # Add save/load buttons
        self.save_button = QPushButton("Save Best Genome")
        self.save_button.clicked.connect(self.save_genome)
        control_layout.addWidget(self.save_button)
        
        # Add checkpoint buttons
        checkpoint_layout = QHBoxLayout()
        
        self.save_checkpoint_button = QPushButton("Save Checkpoint")
        self.save_checkpoint_button.clicked.connect(self.save_checkpoint)
        checkpoint_layout.addWidget(self.save_checkpoint_button)
        
        self.load_checkpoint_button = QPushButton("Load Checkpoint")
        self.load_checkpoint_button.clicked.connect(self.load_checkpoint)
        checkpoint_layout.addWidget(self.load_checkpoint_button)
        
        control_layout.addLayout(checkpoint_layout)
        
        # Add batch simulation button
        self.batch_sim_button = QPushButton("Batch Simulate")
        self.batch_sim_button.clicked.connect(self.batch_simulate)
        control_layout.addWidget(self.batch_sim_button)
        
        # Add visualization options dropdown
        vis_layout = QHBoxLayout()
        vis_layout.addWidget(QLabel("Data Visualization:"))
        self.vis_combo = QComboBox()
        self.vis_combo.addItems([
            "Select Visualization...",
            "Fitness History",
            "Neural Network Visualization",
            "Network Complexity",
            "Solution Diversity"
        ])
        vis_layout.addWidget(self.vis_combo)
        
        # Add generate button
        self.generate_vis_button = QPushButton("Generate")
        self.generate_vis_button.clicked.connect(self.generate_visualization)
        vis_layout.addWidget(self.generate_vis_button)
        
        control_layout.addLayout(vis_layout)
        
        # Push controls to the top
        control_layout.addStretch()
        
        # Create evolution thread
        self.evolution_thread = EvolutionThread(self.evolution)
        self.evolution_thread.update_signal.connect(self.update_stats)
        self.evolution_thread.generation_completed_signal.connect(self.on_generation_completed)
        
        # Create visualization timer
        self.vis_timer = QTimer()
        self.vis_timer.timeout.connect(self.update_visualization)
        
        # Path tracking is disabled by default now
        # self.evolution.enable_path_tracking(True)
    
    def simulate_generation(self):
        """Simulate a full generation in the background."""
        self.sim_button.setEnabled(False)
        self.vis_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        self.statusBar().showMessage("Simulating generation...")
        
        if not self.evolution_thread.isRunning():
            self.evolution_thread.start()
        
        self.evolution_thread.simulate_generation()
    
    def on_generation_completed(self):
        """Handle generation completion."""
        self.sim_button.setEnabled(True)
        self.vis_button.setEnabled(True)
        
        # Show popup notification
        QMessageBox.information(self, "Generation Completed", 
                               f"Generation {self.evolution.generation} simulation completed!\n"
                               f"Best fitness: {self.evolution.best_fitness:.2f}\n"
                               f"Click 'Visualize Agents' to see the results.")
        
        # Prepare visualization
        self.maze_viz.prepare_visualization()
        self.step_counter.setText(f"0/{self.evolution.steps_per_generation}")
        
        self.statusBar().showMessage("Generation completed. Ready to visualize.")
    
    def start_visualization(self):
        """Start visualizing the selected agents."""
        if not self.maze_viz.selected_agents:
            self.maze_viz.prepare_visualization()
            
        self.maze_viz.start_visualization()
        self.vis_timer.start(int(100 / self.evolution_thread.speed))  # Adjust interval based on speed
        self.step_counter.setText(f"0/{self.evolution.steps_per_generation}")
        self.statusBar().showMessage("Visualizing agents...")
    
    def update_visualization(self):
        """Update the visualization."""
        if not self.maze_viz.step_visualization():
            # Visualization completed
            self.vis_timer.stop()
            self.statusBar().showMessage("Visualization completed.")
        
        # Update step counter
        self.step_counter.setText(f"{self.maze_viz.current_step}/{self.maze_viz.total_steps}")
    
    def stop_evolution(self):
        """Stop the evolution or visualization."""
        # Stop the evolution thread
        if self.evolution_thread.isRunning():
            self.evolution_thread.stop()
            # self.evolution_thread.wait() # Avoid potential deadlock if called from signal handler
        
        # Stop batch simulation if running
        if hasattr(self, 'batch_thread') and self.batch_thread.isRunning():
            self.batch_thread.stop()
            # self.batch_thread.wait()
        
        # Stop the visualization timer
        if self.vis_timer.isActive():
            self.vis_timer.stop()
            
        # Reset UI elements
        self.sim_button.setEnabled(True)
        self.batch_sim_button.setEnabled(True)
        # Enable viz only if a generation has run *after* maze change
        self.vis_button.setEnabled(False) 
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Stopped.")
        self.step_counter.setText("0/0")
    
    def set_speed(self, index):
        """Set the evolution speed."""
        speeds = [0.5, 1.0, 2.0, 5.0, 10.0]
        self.evolution_thread.speed = speeds[index]
        
        # Update timer interval if visualization is running
        if self.vis_timer.isActive():
            self.vis_timer.setInterval(int(100 / self.evolution_thread.speed))
    
    def toggle_show_all_agents(self, checked):
        """Toggle whether to show all agents or just the best ones."""
        self.maze_viz.show_all_agents = checked
        self.maze_viz.update()
    
    def update_stats(self, stats):
        """Update the stats display."""
        self.generation_label.setText(f"Generation: {stats['generation']}")
        self.fitness_label.setText(f"Best Fitness: {stats['best_fitness']:.2f}")
        self.avg_fitness_label.setText(f"Avg Fitness: {stats['avg_fitness']:.2f}")
        self.species_label.setText(f"Species: {stats['species_count']}")
        self.completion_label.setText(f"Completion Rate: {stats['completion_rate']*100:.1f}%")
        
        # Update visualization
        self.maze_viz.update()
    
    def save_genome(self):
        """Save the best genome."""
        result = self.evolution.save_best_genome()
        if result[0]:  # Check if genome was saved successfully
            saved_path, network_path = result
            
            # Show status bar message
            self.statusBar().showMessage(f"Genome saved to {saved_path}", 3000)
            
            # Show popup notification
            QMessageBox.information(self, "Save Successful", 
                                  f"Best genome successfully saved to:\n{saved_path}\n\n"
                                  f"Neural network visualization saved to:\n{network_path}")
    
    def generate_visualization(self):
        """Generate and save a visualization based on the selected option."""
        vis_type = self.vis_combo.currentText()
        if vis_type == "Select Visualization...":
            QMessageBox.warning(self, "Selection Required", "Please select a visualization type.")
            return
            
        # Common setup
        data_dir = os.path.join(os.getcwd(), 'data')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Map visualization type to function and data getter
        plot_map = {
            "Fitness History": {
                "func": plot_fitness_history,
                "data_getter": self.evolution.get_fitness_history_data,
                "data_key": 'generations', # Key to check if data exists
                "output_dir": "fitness_history",
                "base_filename": "fitness_history",
                "success_msg": "Fitness history plot saved",
                "error_msg": "Failed to generate fitness history plot."
            },
            "Neural Network Visualization": {
                "func": visualize_network,
                "data_getter": lambda: (self.evolution.best_genome, self.evolution.config, self.evolution.generation),
                "data_key": 0, # Check if best_genome (element 0) exists
                "output_dir": "neural_network_map",
                "base_filename": "network_visualization",
                "success_msg": "Neural network visualization saved",
                "error_msg": "Failed to generate neural network visualization."
            },
            "Network Complexity": {
                "func": plot_network_complexity,
                "data_getter": self.evolution.get_complexity_history_data,
                "data_key": None, # Check if the list itself is non-empty
                "output_dir": "n_network_complexity",
                "base_filename": "network_complexity",
                "success_msg": "Network complexity plot saved",
                "error_msg": "Failed to generate network complexity plot."
            },
            "Solution Diversity": {
                "func": plot_solution_diversity,
                "data_getter": self.evolution.get_diversity_history_data,
                "data_key": None, # Check if the list itself is non-empty
                "output_dir": "solution_diversity",
                "base_filename": "solution_diversity",
                "success_msg": "Solution diversity visualization saved",
                "error_msg": "Failed to generate solution diversity visualization."
            }
        }
        
        if vis_type in plot_map:
            config = plot_map[vis_type]
            self._generate_plot(config, data_dir, timestamp)
        else:
             QMessageBox.warning(self, "Error", f"Unknown visualization type selected: {vis_type}")
             
    def _generate_plot(self, config, data_dir, timestamp):
        """Helper function to generate and save a plot."""
        try:
            # Get data using the specific getter function
            plot_data = config["data_getter"]()
            
            # Check if data is valid/available
            data_exists = False
            if config["data_key"] is not None:
                if isinstance(plot_data, dict):
                    data_exists = bool(plot_data.get(config["data_key"]))
                elif isinstance(plot_data, tuple):
                     data_exists = plot_data[config["data_key"]] is not None # Check specific tuple element
                else: 
                     data_exists = bool(plot_data) # Assume list/other sequence
            else:
                data_exists = bool(plot_data) # Check if list/data itself is non-empty
                
            if not data_exists:
                 QMessageBox.warning(self, "No Data", "No data available yet for this visualization. Run some generations first.")
                 return

            # Prepare output path
            output_subdir = os.path.join(data_dir, config["output_dir"])
            if not os.path.exists(output_subdir):
                 os.makedirs(output_subdir)
            output_path = os.path.join(output_subdir, f'{config["base_filename"]}_{timestamp}.png')
            
            # Call the plotting function
            # Handle functions needing multiple arguments (like visualize_network)
            if config["base_filename"] == "network_visualization":
                 genome, neat_config, generation = plot_data
                 success = config["func"](genome, neat_config, output_path, generation)
            else:
                 success = config["func"](plot_data, output_path)
            
            # Show feedback
            if success:
                self.statusBar().showMessage(f'{config["success_msg"]} to {output_path}', 3000)
                QMessageBox.information(self, "Visualization Saved", 
                                      f'{config["success_msg"]} to:\n{output_path}')
            else:
                QMessageBox.warning(self, "Error", config["error_msg"])
                
        except Exception as e:
             # Catch unexpected errors during the process
             import traceback
             print(f"Unexpected error during plot generation for {config['base_filename']}: {e}")
             traceback.print_exc()
             QMessageBox.critical(self, "Generation Error", f"An error occurred generating the visualization: {e}")
    
    def save_checkpoint(self):
        """Save the current evolution state as a checkpoint."""
        if self.evolution.generation == 0:
            QMessageBox.warning(self, "No Data", "No evolution data to save. Run at least one generation first.")
            return
            
        # Ask for custom filename (optional)
        custom_name, ok = QInputDialog.getText(
            self, "Save Checkpoint", 
            "Enter custom name (optional):",
            text=f"generation_{self.evolution.generation}"
        )
        
        if ok:
            checkpoint_path = self.evolution.save_checkpoint(custom_name if custom_name else None)
            
            if checkpoint_path:
                self.statusBar().showMessage(f"Checkpoint saved to {checkpoint_path}", 3000)
                QMessageBox.information(self, "Checkpoint Saved", 
                                      f"Evolution state saved to:\n{checkpoint_path}")
    
    def load_checkpoint(self):
        """Load an evolution state from a checkpoint."""
        available_checkpoints = self.evolution.get_available_checkpoints()
        
        if not available_checkpoints:
            QMessageBox.warning(self, "No Checkpoints", "No checkpoint files found.")
            return
            
        # Create checkpoint selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Checkpoint")
        layout = QVBoxLayout(dialog)
        
        # Add list of checkpoints
        list_widget = QListWidget()
        for filename, generation, timestamp in available_checkpoints:
            # Format timestamp
            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            item = QListWidgetItem(f"Generation {generation} - {timestamp_str} - {filename}")
            item.setData(Qt.UserRole, os.path.join(self.evolution.checkpoint_dir, filename))
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted and list_widget.currentItem():
            checkpoint_path = list_widget.currentItem().data(Qt.UserRole)
            
            # Confirm before loading
            reply = QMessageBox.question(
                self, "Confirm Load", 
                "Loading a checkpoint will replace the current evolution state. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Stop any running evolution
                if self.evolution_thread.isRunning():
                    self.evolution_thread.stop()
                    self.evolution_thread.wait()
                
                # Load the checkpoint
                if self.evolution.load_checkpoint(checkpoint_path):
                    # Get proper stats from the loaded checkpoint data
                    
                    # Find the most recent generation stats in the fitness history
                    if self.evolution.fitness_history['generations']:
                        latest_idx = -1  # Most recent data point
                        avg_fitness = self.evolution.fitness_history['avg_fitnesses'][latest_idx]
                        worst_fitness = self.evolution.fitness_history['worst_fitnesses'][latest_idx]
                        
                        # Get species count from history if available
                        species_count = 0
                        if (self.evolution.species_history['generations'] and 
                            self.evolution.species_history['species_sizes']):
                            species_count = len(self.evolution.species_history['species_sizes'][-1])
                        
                        # Reconstruct agents from the population for visualization
                        if not self.evolution.agents:
                            # Get genomes from the loaded population
                            genomes = list(self.evolution.population.population.items())
                            
                            # Create temporary agents for the UI (will be fully reconstructed on next run)
                            self.evolution.agents = []
                            for genome_id, genome in genomes:
                                agent = self.evolution.create_agent_from_genome(genome)
                                self.evolution.agents.append((genome_id, agent))
                        
                        # Calculate completion rate from agents if possible
                        completion_rate = 0.0
                        if self.evolution.agents:
                            completion_rate = sum(1 for _, agent in self.evolution.agents if agent.reached_goal) / len(self.evolution.agents)
                        
                        # Update UI with proper stats
                        self.update_stats({
                            'generation': self.evolution.generation,
                            'best_fitness': self.evolution.best_fitness,
                            'avg_fitness': avg_fitness,
                            'species_count': species_count,
                            'completion_rate': completion_rate
                        })
                    else:
                        # Fallback if no history data is available
                        self.update_stats({
                            'generation': self.evolution.generation,
                            'best_fitness': self.evolution.best_fitness,
                            'avg_fitness': 0.0,
                            'species_count': 0,
                            'completion_rate': 0.0
                        })
                    
                    # Prepare visualization
                    self.maze_viz.prepare_visualization()
                    
                    self.statusBar().showMessage(f"Checkpoint loaded from {checkpoint_path}", 3000)
                    
                    # Enable visualization after loading
                    self.vis_button.setEnabled(True)
                    
                    # Restart evolution thread if it was running
                    if not self.evolution_thread.isRunning():
                        self.evolution_thread.start()
    
    def batch_simulate(self):
        """Run multiple generations in batch mode."""
        # Ask for number of generations
        num_generations, ok = QInputDialog.getInt(
            self, "Batch Simulation", 
            "Number of generations to simulate:",
            value=10, min=1, max=100
        )
        
        if not ok:
            return
            
        # Confirm before starting
        reply = QMessageBox.question(
            self, "Confirm Batch Simulation", 
            f"This will simulate {num_generations} generations without interruption.\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Disable buttons during batch simulation
            self.sim_button.setEnabled(False)
            self.vis_button.setEnabled(False)
            self.batch_sim_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            # Start batch simulation in the background
            self.statusBar().showMessage(f"Starting batch simulation of {num_generations} generations...")
            
            # Create a new batch simulation thread
            class BatchSimulationThread(QThread):
                """QThread to run batch simulation without blocking the GUI."""
                update_signal = pyqtSignal(dict)
                completed_signal = pyqtSignal()
                
                def __init__(self, evolution, num_generations):
                    """Initialize the batch simulation thread.

                    Args:
                        evolution: The MazeEvolution instance.
                        num_generations: Number of generations to simulate.
                    """
                    super().__init__()
                    self.evolution = evolution
                    self.num_generations = num_generations
                    self.running = True
                
                def run(self):
                    """Run the batch simulation."""
                    for i in range(self.num_generations):
                        if not self.running:
                            break
                            
                        # Run one generation
                        stats = self.evolution.run_generation()
                        
                        # Emit update signal
                        self.update_signal.emit(stats)
                        
                        # Sleep briefly to allow UI updates
                        time.sleep(0.01)
                    
                    # Emit completion signal
                    if self.running:
                        self.completed_signal.emit()
                
                def stop(self):
                    """Stop the batch simulation gracefully."""
                    self.running = False
            
            # Create and start the batch thread
            self.batch_thread = BatchSimulationThread(self.evolution, num_generations)
            self.batch_thread.update_signal.connect(self.update_stats)
            self.batch_thread.completed_signal.connect(self.on_batch_completed)
            
            if not self.evolution_thread.isRunning():
                self.evolution_thread.start()
                
            self.batch_thread.start()
    
    def on_batch_completed(self):
        """Handle batch simulation completion."""
        # Re-enable buttons
        self.sim_button.setEnabled(True)
        self.vis_button.setEnabled(True)
        self.batch_sim_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Update status
        self.statusBar().showMessage(f"Batch simulation completed. Generation: {self.evolution.generation}")
        
        # Show popup notification
        QMessageBox.information(self, "Batch Simulation Completed", 
                              f"Completed batch simulation through generation {self.evolution.generation}.\n"
                              f"Best fitness: {self.evolution.best_fitness:.2f}\n"
                              f"Click 'Visualize Agents' to see the results.")
        
        # Prepare visualization
        self.maze_viz.prepare_visualization()
    
    def closeEvent(self, event):
        """Clean up when the window is closed."""
        self.evolution_thread.stop()
        self.vis_timer.stop()
        event.accept()

    def change_maze(self):
        """Handle maze selection change."""
        selected_maze = self.maze_combo.currentText()
        
        # Stop any ongoing simulation/visualization
        self.stop_evolution() # Use stop to ensure clean state
        self.sim_button.setEnabled(True) # Re-enable sim button after stopping
        self.vis_button.setEnabled(False) # Disable viz until next gen
        self.stop_button.setEnabled(False)
        
        # Tell the evolution engine to change the maze
        self.evolution.set_maze(selected_maze)
        
        # Update the maze visualization widget
        self.maze_viz.set_evolution(self.evolution)
        self.maze_viz.update() # Force repaint
        
        # Update status bar
        self.statusBar().showMessage(f"Switched to {selected_maze} maze. Ready for next generation.")
        
        # Reset step counter display
        self.step_counter.setText("0/0") 