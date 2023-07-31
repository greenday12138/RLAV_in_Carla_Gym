import torch


def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free

def get_gpu_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # 获取当前GPU名字
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        # 获取当前GPU总显存
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory / 1e9

        gpu_info = "当前GPU 型号是：{}，可用总显存为：{} GB".format(gpu_name, total_memory)
        return gpu_info, gpu_name
    else:
        src = "No GPU found"
        return src, src
