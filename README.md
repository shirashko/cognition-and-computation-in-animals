# Bee Hive Simulation and Analysis

This repository contains a Python simulation and analysis of bee hive behavior based on different traits and behaviors of individual bees. The simulation explores how the traits of exploration, boldness, and sociability affect the overall nectar quantity and quality collected by the hive.

![Alt Text](https://img.freepik.com/free-vector/cute-bee-flying-cartoon-vector-icon-illustration-animal-nature-icon-concept-isolated-premium-vector_138676-6016.jpg)


## Overview

The simulation involves the following main components:

- `Bee` class: Represents individual bees with traits such as exploration, boldness, and sociability. Bees simulate foraging behavior based on these traits and the nectar-related attributes.

- `create_hive` function: Creates a hive of bees with given traits and simulates their foraging behavior. It calculates average nectar quantity, nectar quality, number of surviving bees, and average traits.

- `simulate_hive_model` function: Simulates the hive model for multiple simulations and plots the results, showing the distribution of various attributes such as exploration, boldness, sociability, nectar quantity, nectar quality, and bee survival.

- `simulate_hives` function: Simulates multiple hives with different trait standard deviations and plots the results, exploring the impact of trait heterogeneity vs. homogeneity on hive metrics.

- `simulate_hives_with_traits` function: Simulates hives with different trait combinations and returns the results, followed by the identification of optimal trait combinations for maximizing nectar quantity and quality.

## Prerequisites

- Python 3.x
- Required libraries: `heapq`, `random`, `matplotlib`, `numpy`

## Usage

1. Clone the repository:

```
git clone https://github.com/shirashko/bee-hive-simulation.git
```

2. Navigate to the repository folder:

```
cd bee-hive-simulation
```

3. Run the simulation and analysis script:

```
python bee_simulation.py
```

This will run the simulation and display various plots showcasing the impact of different traits on hive metrics.

## Results

The simulation provides insights into how individual bee traits affect hive behavior. Key findings include the optimal trait combinations that maximize nectar quantity and quality within the hive, and the tradeoff between homogeneity and heterogeneity in the traits of bees within the hive. The simulation results are presented graphically in the output plots.

## License

This project is licensed under the [MIT License](LICENSE).
