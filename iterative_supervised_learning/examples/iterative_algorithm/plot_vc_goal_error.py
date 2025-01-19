import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use a non-interactive backend

# Read data from the Excel file
file_path = "error_data.xlsx"  # Replace with your file path
error_data = pd.read_excel(file_path, sheet_name=None)  # Read all sheets

# Extract data for plotting
# Assuming "error_vx_his" and "error_vy_his" sheets exist
error_vx_his = pd.DataFrame(error_data['error_vx_his'])
error_vy_his = pd.DataFrame(error_data['error_vy_his'])

# Get the number of policies and goals
num_policies = error_vx_his.shape[0]
num_goals = error_vx_his.shape[1]

# Plot errors for vx
plt.figure(figsize=(10, 6))
for goal_idx in range(num_goals):  # Iterate over each goal
    plt.plot(
        range(1, num_policies + 1),  # Policy indices (1-based)
        error_vx_his.iloc[:, goal_idx],  # Errors for this goal across policies
        marker='o',
        label=f'Goal {goal_idx + 1}'  # Goal label
    )
plt.xlabel('Policy')
plt.ylabel('Error (vx)')
plt.title('Errors in vx Across Policies and Goals')
plt.xticks(range(1, num_policies + 1), labels=[f"Policy {i}" for i in range(1, num_policies + 1)])
plt.legend(title="Goals")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot errors for vy
plt.figure(figsize=(10, 6))
for goal_idx in range(num_goals):  # Iterate over each goal
    plt.plot(
        range(1, num_policies + 1),  # Policy indices (1-based)
        error_vy_his.iloc[:, goal_idx],  # Errors for this goal across policies
        marker='x',
        label=f'Goal {goal_idx + 1}'  # Goal label
    )
plt.xlabel('Policy')
plt.ylabel('Error (vy)')
plt.title('Errors in vy Across Policies and Goals')
plt.xticks(range(1, num_policies + 1), labels=[f"Policy {i}" for i in range(1, num_policies + 1)])
plt.legend(title="Goals")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot separate graphs for each goal in vx
for goal_idx in range(num_goals):  # Iterate over each goal
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, num_policies + 1),  # Policy indices (1-based)
        error_vx_his.iloc[:, goal_idx],  # Errors for this goal across policies
        marker='o',
        linestyle='-', 
        label=f'Goal {goal_idx + 1}'
    )
    plt.xlabel('Policy')
    plt.ylabel(f'Error (vx) for Goal {goal_idx + 1}')
    plt.title(f'Error in vx for Goal {goal_idx + 1} Across Policies')
    plt.xticks(range(1, num_policies + 1), labels=[f"Policy {i}" for i in range(1, num_policies + 1)])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"vx_error_goal_{goal_idx + 1}.png")  # Save plot to file
    plt.show()

# Plot separate graphs for each goal in vy
for goal_idx in range(num_goals):  # Iterate over each goal
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, num_policies + 1),  # Policy indices (1-based)
        error_vy_his.iloc[:, goal_idx],  # Errors for this goal across policies
        marker='x',
        linestyle='-', 
        label=f'Goal {goal_idx + 1}'
    )
    plt.xlabel('Policy')
    plt.ylabel(f'Error (vy) for Goal {goal_idx + 1}')
    plt.title(f'Error in vy for Goal {goal_idx + 1} Across Policies')
    plt.xticks(range(1, num_policies + 1), labels=[f"Policy {i}" for i in range(1, num_policies + 1)])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"vy_error_goal_{goal_idx + 1}.png")  # Save plot to file
    plt.show()