# Print Factory Layout Optimization

This repository contains two implementations for optimizing the layout of a print factory. The first implementation uses a genetic algorithm, and the second one uses linear programming (LP) with the Pulp library.

## Genetic Algorithm Implementation

### Description
The genetic algorithm optimizes the layout of a print factory by rearranging the positions of machines based on an operational sequence and machine groups.

### Usage
- Run `genetic_algorithm_layout.py` to execute the genetic algorithm and visualize the initial and final shop floor layouts.
- The algorithm aims to minimize the total distance traveled in the operational sequence.

## LP Modeler Implementation

### Description
The LP modeler uses linear programming to optimize the layout by minimizing the number of pairwise exchanges needed to achieve the desired sequence.

### Usage
- Run `lp_modeler_layout.py` to execute the LP modeler and visualize the initial and final shop floor layouts.
- The LP modeler minimizes the number of swaps required to satisfy the operational sequence and keep machine groups together.

## Shop Floor Layouts

### Initial Layout
![Initial Layout](images/initial_layout.png)

### Final Layout
![Final Layout](images/final_layout.png)

## Distance Minimization

### Genetic Algorithm
![Distance Minimization - Genetic Algorithm](images/distance_minimization_genetic_algorithm.png)

### LP Modeler
![Distance Minimization - LP Modeler](images/distance_minimization_lp_modeler.png)

## Dependencies
- Python 3.x
- Matplotlib
- NumPy
- Pulp (for LP Modeler)

Feel free to explore and experiment with different parameters and sequences to observe the optimization results.
