import torch
import sys

# Replace with the path to your model
model_path = "/home/atari_ws/iterative_supervised_learning/examples/iterative_algorithm/data/safedagger/trot/Dec_16_2024_14_12_55/network/policy_1.pth"

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Attempt to load the policy network
    policy_network = torch.load(model_path, map_location=device)
    
    policy_network.eval()
    
    # Check if the network is None
    if policy_network is None:
        raise ValueError("Failed to load the policy network. Check the model file.")
    
    # Print success message
    print("Policy network loaded successfully!")
    
    # Print additional details about the network (optional)
    print(policy_network)

except FileNotFoundError:
    print(f"Error: The model file was not found at path: {model_path}")
    sys.exit(1)

except ValueError as ve:
    print(f"ValueError: {ve}")
    sys.exit(1)

except Exception as e:
    print(f"An unexpected error occurred while loading the policy network: {e}")
    sys.exit(1)
