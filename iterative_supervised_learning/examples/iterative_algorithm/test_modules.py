# test error-conditioned goal distribution
# TODO: Test rollout_policy, error: too many values to unpack

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')
# Define the range and resolution of the goal space
vx_min, vx_max, vx_bins = 0.0, 1.0, 10
vy_min, vy_max, vy_bins = -0.5, 0.5, 10
w_min, w_max, w_bins = -0.2, 0.2, 10

# Create a 3D grid for (vx, vy, w)
vx_vals = np.linspace(vx_min, vx_max, vx_bins)
vy_vals = np.linspace(vy_min, vy_max, vy_bins)
w_vals = np.linspace(w_min, w_max, w_bins)

# Initialize the uniform distribution over the grid
P_vxvyw = np.ones((vx_bins, vy_bins, w_bins)) / (vx_bins * vy_bins * w_bins)

def compute_likelihood(vx_vals, vy_vals, w_vals, observed_goal, error, sigma=0.1):
    """
    Compute the likelihood P(e | vx, vy, w) for each grid point.

    Args:
        vx_vals (array): Discretized vx values.
        vy_vals (array): Discretized vy values.
        w_vals (array): Discretized w values.
        observed_goal (tuple): Observed goal (vx, vy, w).
        error (float): Observed error associated with the goal.
        sigma (float): Standard deviation for the Gaussian.

    Returns:
        ndarray: Likelihood values for the entire grid.
    """
    vx_obs, vy_obs, w_obs = observed_goal
    likelihood = np.zeros((len(vx_vals), len(vy_vals), len(w_vals)))

    for i, vx in enumerate(vx_vals):
        for j, vy in enumerate(vy_vals):
            for k, w in enumerate(w_vals):
                # Gaussian likelihood centered at the observed goal
                goal_diff = np.array([vx - vx_obs, vy - vy_obs, w - w_obs])
                likelihood[i, j, k] = np.exp(-np.sum(goal_diff**2) / (2 * sigma**2))

    # Normalize likelihood (optional for stability)
    likelihood /= np.sum(likelihood)
    return likelihood

def update_goal_distribution(P_vxvyw, likelihood):
    """
    Update the goal distribution using the likelihood.

    Args:
        P_vxvyw (ndarray): Current prior distribution P(vx, vy, w).
        likelihood (ndarray): Likelihood P(e | vx, vy, w).

    Returns:
        ndarray: Updated posterior distribution P(vx, vy, w | e).
    """
    # Compute the unnormalized posterior
    posterior = P_vxvyw * likelihood

    # Normalize the posterior to sum to 1
    posterior /= np.sum(posterior)
    return posterior

def sample_from_distribution(P_vxvyw, vx_vals, vy_vals, w_vals):
    """
    Sample a goal (vx, vy, w) from the updated distribution.

    Args:
        P_vxvyw (ndarray): Updated posterior distribution.
        vx_vals (array): Discretized vx values.
        vy_vals (array): Discretized vy values.
        w_vals (array): Discretized w values.

    Returns:
        tuple: Sampled goal (vx, vy, w).
    """
    # Flatten the distribution and sample an index
    flat_distribution = P_vxvyw.flatten()
    sampled_index = np.random.choice(len(flat_distribution), p=flat_distribution)

    # Convert the index back to grid coordinates
    i, j, k = np.unravel_index(sampled_index, P_vxvyw.shape)
    return vx_vals[i], vy_vals[j], w_vals[k]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_distribution(P_vxvyw, vx_vals, vy_vals, w_vals, slice_w_idx=None, title="P(vx, vy, w)"):
    """
    Visualize P(vx, vy, w) as a 3D scatter plot for a fixed w or entire grid.

    Args:
        P_vxvyw (ndarray): 3D distribution P(vx, vy, w).
        vx_vals (array): Discretized vx values.
        vy_vals (array): Discretized vy values.
        w_vals (array): Discretized w values.
        slice_w_idx (int, optional): Index of w slice to visualize. Defaults to None.
        title (str): Title of the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get indices of grid points
    vx_idx, vy_idx, w_idx = np.meshgrid(
        range(len(vx_vals)),
        range(len(vy_vals)),
        range(len(w_vals)),
        indexing="ij",
    )

    # Flatten the indices and corresponding probabilities
    vx_idx = vx_idx.flatten()
    vy_idx = vy_idx.flatten()
    w_idx = w_idx.flatten()
    probabilities = P_vxvyw.flatten()

    # Filter probabilities for a specific slice of w if provided
    if slice_w_idx is not None:
        mask = (w_idx == slice_w_idx)
        vx_idx = vx_idx[mask]
        vy_idx = vy_idx[mask]
        probabilities = probabilities[mask]

    # Map indices back to their corresponding values
    vx_points = vx_vals[vx_idx]
    vy_points = vy_vals[vy_idx]
    w_points = w_vals[w_idx] if slice_w_idx is None else w_vals[slice_w_idx]

    # Plot 3D scatter
    sc = ax.scatter(vx_points, vy_points, w_points, c=probabilities, cmap="viridis", s=100 * probabilities)
    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.set_zlabel("w")
    fig.colorbar(sc, label="Probability")
    ax.set_title(title)
    plt.show()

def plot_3d_surface(P_vxvyw, vx_vals, vy_vals, w_vals, slice_w_idx=None, title="P(vx, vy) for fixed w"):
    """
    Visualize P(vx, vy) as a 3D surface plot for a fixed w.

    Args:
        P_vxvyw (ndarray): 3D distribution P(vx, vy, w).
        vx_vals (array): Discretized vx values.
        vy_vals (array): Discretized vy values.
        w_vals (array): Discretized w values.
        slice_w_idx (int, optional): Index of w slice to visualize. Defaults to None.
        title (str): Title of the plot.
    """
    if slice_w_idx is None:
        raise ValueError("Please provide a specific slice_w_idx for surface visualization.")

    # Extract the 2D slice for the given w index
    P_vxvy = P_vxvyw[:, :, slice_w_idx]

    # Normalize for visualization (optional)
    P_vxvy /= np.sum(P_vxvy)

    # Create a meshgrid for vx and vy
    vx_grid, vy_grid = np.meshgrid(vx_vals, vy_vals, indexing="ij")

    # Plot the surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(vx_grid, vy_grid, P_vxvy, cmap="viridis", edgecolor="k")

    # Add labels and title
    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.set_zlabel("P(vx, vy)")
    ax.set_title(f"{title}, w = {w_vals[slice_w_idx]:.2f}")
    fig.colorbar(surf, ax=ax, label="Probability Density")
    plt.show()

def print_distribution_as_list(P_vxvyw, vx_vals, vy_vals, w_vals):
    """
    Print the P(vx, vy, w) distribution as a list of tuples with probabilities.

    Args:
        P_vxvyw (ndarray): 3D distribution P(vx, vy, w).
        vx_vals (array): Discretized vx values.
        vy_vals (array): Discretized vy values.
        w_vals (array): Discretized w values.
    """
    print("Distribution (vx, vy, w):")
    for i, vx in enumerate(vx_vals):
        for j, vy in enumerate(vy_vals):
            for k, w in enumerate(w_vals):
                probability = P_vxvyw[i, j, k]
                print(f"({vx:.2f}, {vy:.2f}, {w:.2f}): {probability:.6f}")

# Example usage
print_distribution_as_list(P_vxvyw, vx_vals, vy_vals, w_vals)


# Initialize error list (to simulate observations)
observed_errors = [0.3, 0.5, 0.1, 0.4, 0.6, 0.2, 0.5, 0.3, 0.2, 0.4] # Example observed errors
observed_goals = [(0.5, 0.1, 0.0),
                  (0.2, -0.1, -0.05),
                  (0.7, 0.2, 0.1),
                  (0.4, 0.0, -0.1),
                  (0.6, 0.3, 0.05),
                  (0.3, -0.2, 0.0),
                  (0.5, 0.2, -0.05),
                  (0.1, 0.0, 0.1),
                  (0.4, 0.1, -0.05),
                  (0.6, -0.3, 0.0)]# Example goals

for error, goal in zip(observed_errors, observed_goals):
    # Compute the likelihood for the current observation
    likelihood = compute_likelihood(vx_vals, vy_vals, w_vals, goal, error, sigma=0.1)

    # Update the goal distribution
    P_vxvyw = update_goal_distribution(P_vxvyw, likelihood)
    
    # Example: Plot P(vx, vy, w) for the first slice of w
    plot_3d_distribution(P_vxvyw, vx_vals, vy_vals, w_vals, slice_w_idx=0, title="P(vx, vy, w) for w = w_min")
    plt.savefig('plot.png')  # Save the plot to a file
    
    # Plot the 3D surface for the first slice of w
    # plot_3d_surface(P_vxvyw, vx_vals, vy_vals, w_vals, slice_w_idx=0, title="3D Surface Plot")

    # print_distribution_as_list(P_vxvyw, vx_vals, vy_vals, w_vals)
    
    # Sample a new goal from the updated distribution
    new_goal = sample_from_distribution(P_vxvyw, vx_vals, vy_vals, w_vals)
    print(f"Sampled new goal: {new_goal}")
