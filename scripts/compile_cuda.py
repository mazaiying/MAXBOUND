import os
import torch
from torch.utils.cpp_extension import load

def compile_arch(arch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    src_dir = os.path.expanduser("~/maxbound/src/cuda")
    mod = load(
        name=f"maxbound_ext_{arch.replace('.', '')}",
        sources=[
            os.path.join(src_dir, "bindings.cpp"),
            os.path.join(src_dir, "maxbound_kernels.cu"),
        ],
        build_directory=out_dir,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch.replace('.','')},code=sm_{arch.replace('.','')}"],
        verbose=True,
    )
    print(f"Compiled sm_{arch.replace('.', '')} -> {out_dir}")
    return mod

if __name__ == "__main__":
    compile_arch("8.0", os.path.expanduser("~/maxbound/build_sm80"))
    compile_arch("8.9", os.path.expanduser("~/maxbound/build_sm89"))
