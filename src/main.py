from memory_utils import read_gpu_memory


def Optimem(model, mode):
    if mode == "page":
        print("[*] moving model to paging mode")
    elif mode == "chunk":
        print("[*] moving model to chunked mode")
    else:
        print("[*] mode not found")
        print("[*] DEBUG: " + str(read_gpu_memory()))


if __name__ == "__main__":
    print("[*] testing main")
    Optimem("", "page")
