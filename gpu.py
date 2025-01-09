import torch


def check_device(tensor):
    if tensor.is_cuda:
        return 'GPU'
    else:
        return 'CPU'


def main():
    # Kiểm tra khả năng sử dụng CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')  # Sử dụng GPU
        print(f"CUDA is available. Device set to GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')  # Sử dụng CPU
        print("CUDA is not available. Device set to CPU.")

    # Tạo một tensor và kiểm tra
    tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"Tensor device: {check_device(tensor)}")

    # Tạo một mô hình mẫu và kiểm tra
    model = torch.nn.Linear(10, 5).to(device)
    print(f"Model device: {check_device(next(model.parameters()))}")


if __name__ == "__main__":
    main()
