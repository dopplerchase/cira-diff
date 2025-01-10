import os
import time
import pynvml
from torch.utils.tensorboard import SummaryWriter

# Initialize NVML and TensorBoard writer
pynvml.nvmlInit()
writer = SummaryWriter(log_dir='/mnt/data1/rchas1/latentedm_10_two_inputs_radames_fixeddata_TEST/logs/gpu_stats/')

def log_gpu_stats():
    device_count = pynvml.nvmlDeviceGetCount()
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        writer.add_scalar(f'GPU_{i}/Usage', gpu_utilization, time.time())
        writer.add_scalar(f'GPU_{i}/Temperature', temperature, time.time())

if __name__ == "__main__":
    while True:
        log_gpu_stats()
        time.sleep(30)

    # Clean up NVML
    pynvml.nvmlShutdown()