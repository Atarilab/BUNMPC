import torch
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class NetworkLoader:
    def __init__(self, device, network_savepath="./saved_models"):
        self.device = device  # CUDA or CPU
        self.network_savepath = network_savepath  # Directory for saving/loading networks
        self.vc_network = None
        self.policy_input_parameters = None

        os.makedirs(self.network_savepath, exist_ok=True)

    def load_saved_network(self, filename=None):
        """
        Load a saved network into self.vc_network.
        
        Args:
            filename (str, optional): Path to the saved network file.
                                      If None, interactive file dialog is used.
        """
        # Interactive file selection if no filename is provided
        if filename is None:
            Tk().withdraw()  # Hide root window
            filename = askopenfilename(initialdir=self.network_savepath, title="Select Saved Network File")

        # Validate file existence
        if not filename or not os.path.exists(filename):
            raise FileNotFoundError("No valid file selected or file does not exist!")

        # Load saved payload
        payload = torch.load(filename, map_location=self.device)

        # Validate payload keys
        required_keys = ['network', 'norm_policy_input']
        if not all(key in payload for key in required_keys):
            raise KeyError(f"The saved file is missing one of the required keys: {required_keys}")

        # Load network and move to device
        self.vc_network = payload['network'].to(self.device)
        self.vc_network.eval()

        # Load normalization parameters
        self.policy_input_parameters = payload['norm_policy_input']

        # Success messages
        print(f"Network successfully loaded from: {filename}")
        if self.policy_input_parameters is None:
            print("Policy Input will NOT be normalized.")
        else:
            print("Policy Input normalization parameters loaded.")

# Main execution
if __name__ == "__main__":
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize loader
    network_loader = NetworkLoader(device=device)

    # Load network (interactive or file-based)
    try:
        network_loader.load_saved_network()
        print("Loaded Network Structure:")
        print(network_loader.vc_network)
    except Exception as e:
        print(f"Error: {e}")
