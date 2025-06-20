import torch, platform

print("Sistema operativo:", platform.system(), platform.release())
print("Arquitectura CPU:", platform.machine())
print("¿MPS disponible?      ", torch.backends.mps.is_available())
print("¿MPS compilado?       ", torch.backends.mps.is_built())
print("¿CUDA disponible?     ", torch.cuda.is_available())