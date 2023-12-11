import random
import numpy as np
import matplotlib.pyplot as plt

# Define the operational sequence and machine groups
operational_sequence = ["Paper storage section", "Cutting Machine 1", "Printing Machine CD 102",
                        "Cutting Machine 2", "Stripping Machine", "Punching Machine 1",
                        "Laminating Machine", "Folding Machine 3", "Folder Gluer 1",
                        "Finished goods section"]

machine_groups = [
    ["Printing Machine CD 102", "Printing Machine SM 102", "Printing Machine Ryobi"],
    ["Cutting Machine 1", "Cutting Machine 2"],
    ["Folding Machine 1", "Folding Machine 2", "Folding Machine 3"],
    ["Folder Gluer 1", "Folder Gluer 2"]
]

# Define machine positions and dimensions
machine_positions = {
    "Printing Machine CD 102": (0, 0),
    "Printing Machine SM 102": (0, 23),
    "Printing Machine Ryobi": (0, 46),
    "Cutting Machine 1": (13.5, 69),
    "Paper storage section": (22.5, 112.5),
    "Stripping Machine": (99, 0),
    "Punching Machine 1": (99, 27),
    "Punching Machine 2": (99, 58.5),
    "Laminating Machine": (85.5, 85.5),
    "Cutting Machine 2": (112.5, 112.5),
    "Inspection": (153, 0),
    "Window Patch": (193.5, 0),
    "Stairs": (214.2, 0),
    "Folder Gluer 1": (148.5, 31.5),
    "Folder Gluer 2": (148.5, 45),
    "Folding Machine 3": (148.5, 58.5),
    "ManualLaminator": (148.5, 67.5),
    "Folding Machine 2": (148.5, 90),
    "Folding Machine 1": (186.3, 90),
    "SaddleStitcher": (135, 112.5),
    "Offlinecoater": (223.2, 112.5),
    "Finished goods section": (223.2, 22.5)
}

# Specify the pathway width
pathway_width = 9

# Function to plot the layout
def plot_layout(layout, title):
    plt.figure(figsize=(10, 6))

    # Check if layout is a tuple
    if isinstance(layout, tuple):
        layout = layout[0]

    # Plot machines
    for machine, position in layout.items():
        plt.scatter(position[0], position[1], label=machine)

    # Plot pathways
    plt.axhline(y=pathway_width / 2, color='black', linestyle='--', linewidth=1, label='Pathway')
    plt.axhline(y=112.5 - pathway_width / 2, color='black', linestyle='--', linewidth=1)

    # Configure plot
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot initial layout
plot_layout(machine_positions, 'Initial Shop Floor Layout')

# Function to calculate distance between two points
def calculate_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

# Function to evaluate fitness of a layout
def evaluate_fitness(layout):
    total_distance = 0
    for i in range(len(operational_sequence) - 1):
        m1 = operational_sequence[i]
        m2 = operational_sequence[i + 1]
        total_distance += calculate_distance(layout[m1], layout[m2])
    for group in machine_groups:
        for i in range(len(group) - 1):
            m1 = group[i]
            m2 = group[i + 1]
            total_distance += calculate_distance(layout[m1], layout[m2])
    return total_distance

# Function to perform pairwise exchange mutation
def pairwise_exchange_mutation(layout):
    new_layout = layout.copy()
    machines = random.sample(layout.keys(), 2)
    new_layout[machines[0]], new_layout[machines[1]] = new_layout[machines[1]], new_layout[machines[0]]
    return new_layout

# Function to perform genetic algorithm with tracking fitness scores
def genetic_algorithm(population_size, generations):
    # Initialize population
    population = [machine_positions.copy() for _ in range(population_size)]
    fitness_scores_history = []  # List to store fitness scores for each generation

    for generation in range(generations):
        # Evaluate fitness of each layout in the population
        fitness_scores = [evaluate_fitness(layout) for layout in population]
        fitness_scores_history.append(min(fitness_scores))  # Store the best fitness score

        # Select top 50% layouts based on fitness
        selected_indices = np.argsort(fitness_scores)[:population_size // 2]
        selected_population = [population[i] for i in selected_indices]

        # Create offspring through pairwise exchange mutation
        offspring_population = [pairwise_exchange_mutation(layout) for layout in selected_population]

        # Combine selected population and offspring
        population = selected_population + offspring_population

        # Print the best fitness in each generation
        best_layout = population[np.argmin(fitness_scores)]
        best_fitness = min(fitness_scores)
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    # Return the best layout from the final generation and fitness scores history
    return population[np.argmin(fitness_scores)], fitness_scores_history

# Run the genetic algorithm with tracking fitness scores
final_layout, fitness_scores_history = genetic_algorithm(population_size=50, generations=100)

# Set initial machine positions
initial_machine_positions = machine_positions.copy()

# Print initial layout
print("\nInitial Machine Positions:")
for machine, position in machine_positions.items():
    print(f"{machine}: {position}")

# Evaluate initial fitness
initial_fitness = evaluate_fitness(machine_positions)
print("Initial Fitness Value:", initial_fitness)

# Print the final layout
print("\nFinal Machine Positions:")
for machine, position in final_layout.items():
    print(f"{machine}: {position}")

# Evaluate the fitness of the final layout
final_fitness = evaluate_fitness(final_layout)
print("Final Fitness Value:", final_fitness)

final_layout = genetic_algorithm(population_size=50, generations=100)

# Plot initial and final layouts side by side
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plot_layout(initial_machine_positions, 'Initial Layout')
plt.title('Initial Shop Floor Layout')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plot_layout(final_layout, 'Final Layout')
plt.title('Final Shop Floor Layout')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)

# Move the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
# Plot the fitness scores history
plt.figure()
plt.plot(fitness_scores_history, label='Fitness Score', color='green')
plt.title('Fitness Score over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness Score')
plt.grid(True)
plt.legend()
plt.show()