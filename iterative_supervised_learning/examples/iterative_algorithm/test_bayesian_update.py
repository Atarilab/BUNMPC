import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use the Tkinter backend


# Define the range and resolution of the goal space
vx_min, vx_max, vx_bins = 0.0, 1.0, 100
vy_min, vy_max, vy_bins = -0.5, 0.5, 100

# Create a 2D grid for (vx, vy)
vx_vals = np.linspace(vx_min, vx_max, vx_bins)
vy_vals = np.linspace(vy_min, vy_max, vy_bins)

# Initialize the uniform distribution over the grid
P_vxvy = np.ones((vx_bins, vy_bins)) / (vx_bins * vy_bins)

def compute_likelihood(vx_vals, vy_vals, observed_goal, error, sigma=0.1):
    """
    Compute the likelihood P(e | vx, vy) for each grid point.

    Args:
        vx_vals (array): Discretized vx values.
        vy_vals (array): Discretized vy values.
        observed_goal (tuple): Observed goal (vx, vy).
        error (float): Observed error associated with the goal.
        sigma (float): Standard deviation for the Gaussian.

    Returns:
        ndarray: Likelihood values for the entire grid.
    """
    vx_obs, vy_obs = observed_goal
    likelihood = np.zeros((len(vx_vals), len(vy_vals)))

    for i, vx in enumerate(vx_vals):
        for j, vy in enumerate(vy_vals):
            # Gaussian likelihood centered at the observed goal
            goal_diff = np.array([vx - vx_obs, vy - vy_obs])
            likelihood[i, j] = np.exp(-np.sum(goal_diff**2) / (2 * sigma**2))

    # Normalize likelihood (optional for stability)
    likelihood /= np.sum(likelihood)
    return likelihood

def update_goal_distribution(P_vxvy, likelihood):
    """
    Update the goal distribution using the likelihood.

    Args:
        P_vxvy (ndarray): Current prior distribution P(vx, vy).
        likelihood (ndarray): Likelihood P(e | vx, vy).

    Returns:
        ndarray: Updated posterior distribution P(vx, vy | e).
    """
    # Compute the unnormalized posterior
    posterior = P_vxvy * likelihood

    # Normalize the posterior to sum to 1
    posterior /= np.sum(posterior)
    return posterior

def sample_from_distribution(P_vxvy, vx_vals, vy_vals):
    """
    Sample a goal (vx, vy) from the updated distribution.

    Args:
        P_vxvy (ndarray): Updated posterior distribution.
        vx_vals (array): Discretized vx values.
        vy_vals (array): Discretized vy values.

    Returns:
        tuple: Sampled goal (vx, vy).
    """
    # Flatten the distribution and sample an index
    flat_distribution = P_vxvy.flatten()
    sampled_index = np.random.choice(len(flat_distribution), p=flat_distribution)

    # Convert the index back to grid coordinates
    i, j = np.unravel_index(sampled_index, P_vxvy.shape)
    return vx_vals[i], vy_vals[j]

def plot_2d_distribution(P_vxvy, vx_vals, vy_vals, title="P(vx, vy)"):
    """
    Visualize P(vx, vy) as a 2D heatmap.

    Args:
        P_vxvy (ndarray): 2D distribution P(vx, vy).
        vx_vals (array): Discretized vx values.
        vy_vals (array): Discretized vy values.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.imshow(P_vxvy.T, extent=(vx_min, vx_max, vy_min, vy_max), origin='lower', cmap='viridis')
    fig.colorbar(c, label="Probability Density")
    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.set_title(title)
    plt.show()

def plot_3d_distribution(P_vxvy, vx_vals, vy_vals, title="P(vx, vy) in 3D"):
    """
    Visualize P(vx, vy) as a 3D surface plot.

    Args:
        P_vxvy (ndarray): 2D distribution P(vx, vy).
        vx_vals (array): Discretized vx values.
        vy_vals (array): Discretized vy values.
        title (str): Title of the plot.
    """
    # Create a meshgrid for plotting
    vx_grid, vy_grid = np.meshgrid(vx_vals, vy_vals, indexing="ij")

    # Create the figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the 3D surface
    surf = ax.plot_surface(vx_grid, vy_grid, P_vxvy.T, cmap="viridis", edgecolor="k")

    # Add labels and title
    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.set_zlabel("P(vx, vy)")
    ax.set_title(title)

    # Add a color bar
    fig.colorbar(surf, ax=ax, label="Probability Density")

    # Save plot to a file instead of displaying it (optional for non-interactive backend)
    plt.savefig("3d_distribution.png")

    # Show the plot (only if interactive backends are supported)
    plt.show()

# Example usage
observed_errors = [0.3, 0.5, 0.1, 0.4, 0.6, 0.2, 0.5, 0.3, 0.2, 0.4]  # Example observed errors
observed_goals = [(0.5, 0.1), (0.2, -0.1), (0.7, 0.2), (0.4, 0.0), (0.6, 0.3),
                  (0.3, -0.2), (0.5, 0.2), (0.1, 0.0), (0.4, 0.1), (0.6, -0.3)]  # Example goals

for error, goal in zip(observed_errors, observed_goals):
    # Compute the likelihood for the current observation
    likelihood = compute_likelihood(vx_vals, vy_vals, goal, error, sigma=0.1)

    # Update the goal distribution
    P_vxvy = update_goal_distribution(P_vxvy, likelihood)

    # Plot the distribution
    # plot_2d_distribution(P_vxvy, vx_vals, vy_vals, title="P(vx, vy)")
    plot_3d_distribution(P_vxvy, vx_vals, vy_vals)

    # Sample a new goal from the updated distribution
    new_goal = sample_from_distribution(P_vxvy, vx_vals, vy_vals)
    print(f"Sampled new goal: {new_goal}")
