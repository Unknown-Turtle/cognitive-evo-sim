# cognitive-evo-sim: NEAT Maze Evolution

cognitive-evo-sim is a Python-based simulation environment where agents controlled by evolving neural networks (using NEAT - NeuroEvolution of Augmenting Topologies) learn to solve mazes. The project includes a PyQt5 GUI for visualization, real-time simulation control, data logging, and various analytical plots.

## Features

*   **NEAT-Powered Agents:** Agents use neural networks evolved with the `neat-python` library.
*   **Customizable Mazes:**
    *   L-Shape Maze
    *   U-Shape Maze
    *   C-Shape Maze (spiral)
    *   Dynamic maze switching during runtime.
*   **Interactive GUI (PyQt5):**
    *   Real-time visualization of the maze and agent movements.
    *   Controls for simulation: run next generation, visualize agent paths, stop.
    *   Adjustable simulation speed (0.5x to 10x).
    *   Display of key statistics: generation, best/average fitness, species count, completion rate.
    *   Maze type selection dropdown.
*   **Checkpointing:**
    *   Save and load full evolutionary states (population, histories, configuration).
    *   Automatic checkpointing at configurable intervals (default: every 5 generations).
    *   Manual checkpoint saving with custom names.
*   **Data Visualization & Analysis:**
    *   Generate and save plots for:
        *   Fitness History (best, average, worst fitness over generations).
        *   Neural Network Visualization (structure of the best agent's network).
        *   Network Complexity (average nodes and connections over generations).
        *   Solution Diversity (PCA plot of network weights, colored by fitness).
    *   Plots are saved with timestamps and generation numbers.
*   **Configuration Management:**
    *   NEAT parameters are managed through a `neat_config.txt` file (generated from `neat_config.template` if not present).
*   **Batch Simulation:** Run multiple generations non-interactively.
*   **Code Structure:** Organized into modules for evolution, maze generation, agent logic, utilities, and visualization.

## Project Structure

```
cognitive-evo-sim/
├── configs/
│   ├── neat_config.template        # Template for NEAT configuration
│   └── neat_config.txt             # NEAT configuration file (generated)
├── data/
│   ├── fitness_history/            # Saved fitness history plots
│   │   └── .gitkeep
│   ├── n_network_complexity/       # Saved network complexity plots
│   │   └── .gitkeep
│   ├── neural_network_map/         # Saved neural network visualizations
│   │   └── .gitkeep
│   └── solution_diversity/         # Saved solution diversity plots
│       └── .gitkeep
├── logs/                           # Saved genomes and other log files (gets created automaticaly)
│   └── .gitkeep
├── checkpoints/                    # Saved simulation checkpoints (gets created automaticaly)
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── agent.py                    # Agent logic and neural network interaction
│   ├── evolution.py                # Core NEAT evolution management
│   ├── maze.py                     # Maze generation and management
│   ├── utils.py                    # Utility functions (saving, plotting)
│   └── visualization.py            # PyQt5 GUI and visualization logic
├── .gitignore
├── main.py                         # Main script to run the application
└── README.md
```

## Dependencies

*   Python 3.x
*   `neat-python`
*   `numpy`
*   `matplotlib`
*   `PyQt5`
*   `scikit-learn` (for Solution Diversity plot)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd cognitive-evo-sim
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install neat-python numpy matplotlib PyQt5 scikit-learn
    ```
    Alternatively, if a `requirements.txt` file is provided:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Execute the `main.py` script from the root directory of the project:

```bash
python main.py
```

This will launch the PyQt5 GUI.

## Using the Application

1.  **Start Simulation:**
    *   Click "Simulate Next Generation" to run one generation of evolution.
    *   Statistics will update in the control panel.
2.  **Visualize Agents:**
    *   After a generation completes, click "Visualize Agents" to see a replay of selected agents moving through the maze.
    *   Control the visualization speed using the "Speed" dropdown.
3.  **Change Maze:**
    *   Select a different maze type from the "Maze Type" dropdown. The simulation will adapt.
4.  **Data Visualizations:**
    *   Select a plot type from the "Data Visualization" dropdown.
    *   Click "Generate" to create and save the plot to the `data/` subdirectories.
    *   Available plots:
        *   **Fitness History:** Tracks best, average, and worst fitness.
        *   **Neural Network Visualization:** Shows the network structure of the current best agent.
        *   **Network Complexity:** Tracks the average number of nodes and connections.
        *   **Solution Diversity:** Visualizes the diversity of solutions using PCA on network weights.
5.  **Checkpointing:**
    *   **Save Checkpoint:** Manually save the current simulation state. You can provide a custom name.
    *   **Load Checkpoint:** Load a previously saved simulation state.
    *   Checkpoints are automatically saved every 5 generations by default.
6.  **Save/Load Best Genome:**
    *   "Save Best Genome": Saves the neural network of the current best agent to the `logs/` directory and also generates its network visualization.
    *   "Load Genome": Loads a previously saved genome (primarily for inspection, does not restore full evolutionary state like checkpoints).
7.  **Batch Simulate:**
    *   Click "Batch Simulate" and enter the number of generations to run non-interactively.
8.  **NEAT Configuration:**
    *   The NEAT algorithm parameters are controlled by `configs/neat_config.txt`.
    *   If this file is not present, it will be created from `configs/neat_config.template`. You can modify the template or the generated file to tune the evolution (e.g., population size, mutation rates, activation functions).

## Notes

*   The first time you run the simulation, a `configs/neat_config.txt` file will be generated if it doesn't exist.
*   Generated data (plots, logs, checkpoints) are saved in their respective directories (`data/`, `logs/`, `checkpoints/`). These are ignored by Git by default (except for the `.gitkeep` files that ensure the directories are tracked).

---
