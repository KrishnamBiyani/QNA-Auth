import torch

print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'Device Count: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'Current Device: {torch.cuda.current_device()}')
else:
    print('No CUDA devices available')
