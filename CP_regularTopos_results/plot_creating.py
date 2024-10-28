import matplotlib.pyplot as plt

# Data
average_resilience = {
    'One Tree': [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.960000, 0.880000, 0.650000, 0.420000, 0.480000], 
    'One Tree Checkpoint': [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.960000, 0.920000, 0.680000, 0.480000, 0.560000], 
    'SquareOne': [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.880000, 0.800000, 0.360000, 0.320000, 0.400000]
}


# One_Tree_success = [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.960000, 0.880000, 0.650000, 0.420000, 0.480000]
# One Tree Checkpoint_success = [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.960000, 0.920000, 0.680000, 0.480000, 0.560000]
# SquareOne_success = [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.880000, 0.800000, 0.360000, 0.320000, 0.400000]

# One_Tree_Checkpoint_hops = [8.8, 7.8, 9.2, 11.0, 11.0, 11.0, 9.8, 9.8, 12.2, 13.6, 13.4, 13.6]
# SquareOne_hops = [5.6, 6.0, 9.2, 9.2, 9.2, 8.4, 6.8, 7.2, 10.6, 8.0, 7.4, 8.8]
# One_Tree_hops = [5.4, 5.0, 9.0, 9.0, 9.2, 9.4, 7.8, 7.8, 12.0, 14.6, 15.0, 14.8]

average_hops = {
    'One Tree': [5.4, 5.0, 9.0, 9.0, 9.2, 9.4, 7.8, 7.8, 12.0, 14.6, 15.0, 14.8], 
    'One Tree Checkpoint': [8.8, 7.8, 9.2, 11.0, 11.0, 11.0, 9.8, 9.8, 12.2, 13.6, 13.4, 13.6],
    'SquareOne': [5.6, 6.0, 9.2, 9.2, 9.2, 8.4, 6.8, 7.2, 10.6, 8.0, 7.4, 8.8]
}

failure_rates = [i for i in range(1, len(average_resilience['One Tree']) + 1)]

# Define colors
colors = {
    'One Tree': 'green',
    'One Tree Checkpoint': 'red',
    'SquareOne': 'blue'
}

# Plotting Average Resilience
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for algorithm, values in average_resilience.items():
    plt.plot(failure_rates, values, marker='o', label=algorithm, color=colors[algorithm])
plt.xlabel('Failure Rate')
plt.ylabel('Resilience')
plt.title('Resilience of Algorithms for Random Generated Topologies (n=50, k=5)')
plt.legend()

# Plotting Average Hops
plt.subplot(1, 2, 2)
for algorithm, values in average_hops.items():
    plt.plot(failure_rates, values, marker='o', label=algorithm, color=colors[algorithm])
plt.xlabel('Failure Rate')
plt.ylabel('Hops')
plt.title('Hops of Algorithms for Random Generated Topologies (n=50, k=5)')
plt.legend()

plt.tight_layout()
plt.show()
