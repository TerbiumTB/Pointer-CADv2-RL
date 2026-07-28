import torch

print("🔍 Check Flash Attention 2:")
print(f"   CUDA allowed: {torch.cuda.is_available()}")
print(f"   CIDA Arch: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")

# Check flash-attn
try:
    import flash_attn
    print(f"   ✅ Flash Attention installed: {flash_attn.__version__}")
except:
    print("   ⚠️ Flash Attention not found")
