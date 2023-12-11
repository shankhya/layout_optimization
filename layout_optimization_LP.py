from pulp import LpVariable, lpSum, LpMinimize, LpProblem, value
import random
import matplotlib.pyplot as plt

def calculate_distance(positions, section1, section2):
    pos1 = positions[section1]
    pos2 = positions[section2]
    distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    return distance

def pairwise_exchange(positions, machine1, machine2):
    new_positions = positions.copy()
    new_positions[machine1], new_positions[machine2] = new_positions[machine2], new_positions[machine1]
    return new_positions

def optimize_layout(initial_positions, sequence, pathway_width=9, max_iterations=1000):
    current_positions = initial_positions.copy()
    current_distance = calculate_distance(current_positions, "Paper storage", "Finishedgoods")

    # Create LP Problem
    prob = LpProblem("LayoutOptimization", LpMinimize)

    # Create variables
    x = {machine: LpVariable(machine + "_x", lowBound=0, upBound=250) for machine in current_positions}
    y = {machine: LpVariable(machine + "_y", lowBound=0, upBound=117.5) for machine in current_positions}

    # Binary variables to represent pairwise exchanges
    swap = {(m1, m2): LpVariable(f"swap_{m1}_{m2}", cat="Binary") for m1 in current_positions for m2 in current_positions if m1 != m2}

    # Objective function
    prob += lpSum(swap[m1, m2] for m1 in current_positions for m2 in current_positions if m1 != m2)

    # Pathway constraints
    for machine, (px, py) in current_positions.items():
        prob += x[machine] >= px - pathway_width
        prob += x[machine] <= px + pathway_width
        prob += y[machine] >= py - pathway_width
        prob += y[machine] <= py + pathway_width

    # Constraints to enforce pairwise exchanges
    for m1 in current_positions:
        for m2 in current_positions:
            if m1 != m2:
                prob += x[m1] - x[m2] + 250 * swap[m1, m2] >= 0
                prob += y[m1] - y[m2] + 117.5 * swap[m1, m2] >= 0

    # Constraints to enforce the operational sequence
    for i in range(len(sequence) - 1):
        m1 = sequence[i]
        m2 = sequence[i + 1]
        prob += x[m1] <= x[m2]

    # Constraints to keep specific groups together
    for group in machine_groups:
        for i in range(len(group) - 1):
            m1 = group[i]
            m2 = group[i + 1]
            prob += x[m1] <= x[m2]



    # Solve the problem
    prob.solve()

    # Extract the optimized positions
    final_positions = {machine: (value(x[machine]), value(y[machine])) for machine in current_positions}

    # Calculate the optimized distance
    final_distance = calculate_distance(final_positions, "Paper storage", "Finishedgoods")

    return final_positions, final_distance

# Given initial machine positions
initial_machine_positions = {
     "CD102": (0, 0),
    "SM102": (0, 23),
    "Ryobi": (0, 46),
    "Cutting1": (13.5, 69),
    "Paper storage": (22.5, 112.5),
    "Stripping": (99, 0),
    "Punching1": (99, 27),
    "Punching2": (99, 58.5),
    "LaminatingAuto": (85.5, 85.5),
    "Cutting2": (112.5, 112.5),
    "Inspection": (153, 0),
    "Window Patch": (193.5, 0),
    "Stairs": (214.2, 0),
    "FolderGluer1": (148.5, 31.5),
    "FolderGluer2": (148.5, 45),
    "Folding3": (148.5, 58.5),
    "ManualLaminator": (148.5, 67.5),
    "Folding2": (148.5, 90),
    "Folding1": (186.3, 90),
    "SaddleStitcher": (135, 112.5),
    "Offlinecoater": (223.2, 112.5),
    "Finishedgoods": (223.2, 22.5)
}

# Specify the operational sequence
operational_sequence = ["Paper storage", "Cutting1", "CD102",
                        "Cutting2", "Stripping", "Punching1",
                        "LaminatingAuto", "Folding3", "FolderGluer1",
                        "Finishedgoods"]

# Specify machine groups
machine_groups = [
    ["CD102", "SM102", "Ryobi"],
    ["Folding1", "Folding2", "Folding3"],
    ["FolderGluer1", "FolderGluer2"]
]

# Specify the pathway width (you can change this value)
pathway_width = 9

# Initial positions and distance
print("Initial Machine Positions:")
for machine, position in initial_machine_positions.items():
    print(f"{machine}: {position}")

initial_distance = calculate_distance(initial_machine_positions, "Paper storage", "Finishedgoods")
print("Initial Distance Value to Finished Goods:", initial_distance)

# Optimize the layout
final_positions, final_distance = optimize_layout(initial_machine_positions, operational_sequence, pathway_width)

# Print the final results
print("\nFinal Machine Positions:")
for machine, position in final_positions.items():
    print(f"{machine}: {position}")

final_distance_to_finished_goods = calculate_distance(final_positions, "Paper storage", "Finishedgoods")
print("Final Distance Value to Finished Goods:", final_distance_to_finished_goods)

# Compare initial and final distance values
distance_reduction = initial_distance - final_distance_to_finished_goods
print(f"\nDistance Reduction to Finished Goods: {distance_reduction}")

def plot_layout(layout, title):
    plt.figure(figsize=(10, 6))

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
plot_layout(initial_machine_positions, 'Initial Shop Floor Layout')

# Optimize the layout
final_positions, final_distance = optimize_layout(initial_machine_positions, operational_sequence, pathway_width)

# Plot final layout
plot_layout(final_positions, 'Final Shop Floor Layout')

