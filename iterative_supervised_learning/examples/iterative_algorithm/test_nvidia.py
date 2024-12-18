import torch

def test_gpu():
    print("PyTorch version:", torch.__version__)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Ensure your NVIDIA GPU drivers and CUDA are properly configured.")
        return

    print("CUDA is available!")
    
    # Display GPU details
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs detected: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Set default device and test tensor operations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Perform a simple tensor operation
    try:
        x = torch.rand(3, 3).to(device)
        print("Tensor successfully moved to GPU.")
        print("Tensor content (on GPU):")
        print(x)
    except Exception as e:
        print(f"Failed to perform tensor operation on GPU: {e}")

if __name__ == "__main__":
    test_gpu()
