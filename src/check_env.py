"""Run this first:  python src/check_env.py

It reads the machine you are on, tells you which hardware tier of the project you are in, checks that
the libraries you need import at the pinned versions, and prints the lines to paste into section 6 of
your fine-tuning spec. It changes nothing on disk.
"""
import importlib
import platform
import subprocess
import sys

PINNED = {"torch": "2.7", "transformers": "4.51", "peft": "0.15", "datasets": "3.5", "evaluate": "0.4",
          "mlflow": "2.22", "spacy": "3.8"}
OK, WARN, BAD = "[ok]  ", "[warn]", "[FAIL]"


def total_ram_gb():
    try:
        if platform.system() == "Darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
        if platform.system() == "Linux":
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 2**20
        if platform.system() == "Windows":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                           [(n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "aev")]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.total / 2**30
    except Exception:
        pass
    return None


def main():
    problems = 0
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        print(f"{BAD} Python 3.10 to 3.13 is supported. Install one of those and recreate your virtual environment.")
        problems += 1
    if platform.system() == "Darwin" and platform.machine() == "x86_64":
        print(f"{BAD} Intel Mac: PyTorch no longer ships builds for this machine. Use notebooks/colab_train.ipynb "
              "for training and evaluation; everything in Task 1 and the data work in Task 2 still runs here "
              "without torch.")
        problems += 1

    for mod, want in PINNED.items():
        try:
            got = importlib.import_module(mod).__version__
            flag = OK if got.startswith(want) else WARN
            print(f"{flag} {mod} {got}" + ("" if flag == OK else f"  (the course is pinned to {want}.x; "
                  "run: pip install -r requirements.txt)"))
        except Exception as e:  # noqa: BLE001
            print(f"{BAD} {mod} does not import: {e}")
            problems += 1

    try:
        import spacy
        spacy.load("en_core_web_sm")
        print(f"{OK} spaCy model en_core_web_sm loads")
    except Exception:
        print(f"{BAD} spaCy model missing. Run: pip install -r requirements.txt")
        problems += 1

    ram = total_ram_gb()
    try:
        import torch
    except Exception:
        print("\nTorch is not installed, so the hardware tier cannot be read. Fix the lines above first.")
        sys.exit(1)

    print()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        vram = p.total_memory / 2**30
        cc = f"{p.major}.{p.minor}"
        precision = "bf16=True" if p.major >= 8 else "fp16=True"
        try:
            import bitsandbytes  # noqa: F401
            print(f"{OK} bitsandbytes imports, so 4-bit (QLoRA) loading is available")
        except Exception:
            print(f"{BAD} bitsandbytes is missing. Run: pip install -r requirements-cuda.txt")
            problems += 1
        if vram >= 15:
            tier = "NVIDIA 16 GB+: the reference plan (1 to 2B under QLoRA); a 7 to 8B model is the stretch"
        elif vram >= 5.5:
            tier = "NVIDIA 6 GB+: the reference plan, a 1 to 2B model under QLoRA, batch size 1 with accumulation"
        else:
            tier = "NVIDIA under 6 GB: use the 0.5B class under QLoRA, or the Colab notebook"
        print(f"TIER  {tier}")
        print("\nPaste into finetune_spec.md section 6:")
        print(f"  Machine: {p.name}, {vram:.1f} GB VRAM reported, compute capability {cc}; precision flag {precision}.")
        print("  Also paste the memory line from `nvidia-smi`.")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        if ram and ram >= 15:
            tier = ("Apple Silicon, 16 GB+: a 0.5B model under plain LoRA in fp32 runs comfortably; a 1.5B model "
                    "under plain LoRA is the stretch. Close other apps before training")
        else:
            tier = ("Apple Silicon, 8 GB: the 0.5B class under plain LoRA with gradient checkpointing and a "
                    "sequence length of 768 or less, or the Colab notebook")
        print(f"TIER  {tier}")
        print("      4-bit (QLoRA) loading needs an NVIDIA GPU. On a Mac the frozen base stays in full precision,")
        print("      so say 'plain LoRA, fp32, device mps' in spec section 3, and rule QLoRA out in section 7.")
        probe = ("import torch; s=torch.randn(1,151936,device='mps'); i=torch.tensor([[1,2,3]],device='mps'); "
                 "torch.gather(s,1,i).cpu()")
        if subprocess.run([sys.executable, "-c", probe], capture_output=True).returncode != 0:
            print(f"{WARN} On this macOS version, model.generate() with a repetition penalty crashes Python on the mps")
            print("       device (`total bytes of NDArray > 2**32`). Qwen models set repetition_penalty=1.1 by default, so")
            print("       EVERY generate() call is affected until you handle it. README, 'Generating text on a Mac', has")
            print("       the ten-line fix. Training is not affected.")
        else:
            print(f"{OK} generate() with a repetition penalty works on this macOS version")
        print("\nPaste into finetune_spec.md section 6:")
        print(f"  Machine: {chip}, {ram:.0f} GB unified memory shared with the OS, PyTorch device `mps`, no NVIDIA GPU")
        print("  (so there is no nvidia-smi line). Plan for training to use at most about 60% of that memory.")
        print("  Peak memory per run: sample torch.mps.driver_allocated_memory() every step and keep the maximum.")
    else:
        print("TIER  No GPU PyTorch can use. Two routes, and your spec names which:")
        print("      (a) the 0.5B class under plain LoRA on the CPU, fp32, hours per run; or")
        print("      (b) notebooks/colab_train.ipynb, the same scripts on a free hosted T4.")
        if platform.system() == "Windows":
            print("      On Windows with an NVIDIA card this usually means torch was installed from PyPI, which is")
            print("      the CPU build. See README 'Set up your machine' and reinstall torch from the CUDA index.")
        print("\nPaste into finetune_spec.md section 6:")
        print(f"  Machine: {platform.processor() or platform.machine()}, {ram:.0f} GB RAM, no GPU; PyTorch device `cpu`."
              if ram else f"  Machine: {platform.processor() or platform.machine()}, no GPU; PyTorch device `cpu`.")
        print("  Peak memory per run: resource.getrusage(...).ru_maxrss, or Task Manager on Windows.")

    print("\nMLflow: use a local store you name yourself, for example")
    print('  mlflow.set_tracking_uri("sqlite:///mlflow.db")')
    print("so the same line works on every MLflow version.")
    print("\n" + ("All checks passed." if not problems else f"{problems} problem(s) above to fix before Task 1."))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
