import torch
import time
import threading
import os
import concurrent.futures



# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)



# def load_model(model_file):
#     model = torch.jit.load(model_file)
#     print(f"{model_file} has loaded..")




# model_dir = "/home/zy/vision/ultralytics/d2_weights"
# for file in os.listdir(model_dir):
#     th







import torch
import concurrent.futures



model_dir = "/home/zy/vision/ultralytics/d2_weights"


# 检查CUDA是否可用
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your PyTorch installation and CUDA setup.")

# 假设我们有8个TorchScript模型文件
model_files = [f'model0{i}.ts' for i in [1,2,3,4,5,6,7,8]]

# 定义一个加载模型并移动到GPU的函数
def load_model(model_file):
    torch.jit.load(model_file)
    # model.to('cuda')
    # model.eval()
    print(f"{model_file} has loaded..")
    # return model


tt1 = time.time()

load_model("/home/zy/vision/ultralytics/d2_weights/model01.ts")


# for i in range(8):
#     load_model(os.path.join(model_dir, model_files[i]))
#     load_model(os.path.join(model_dir, model_files[i]))


tt2 = time.time()

print("load model use time: ",tt2 - tt1)










# threads = []
# t1 = time.time()


# for i in range(8):
#     threads.append(threading.Thread(target = load_model,args = (model_files[i])))

# for th in threads:
#     th.start()


# for th in threads:
#     th.join()


# t2 = time.time()
# print("load model use : ", t2-t1)











# # 创建一些输入数据并移动到GPU
# input_data = [torch.randn(1, 3, 224, 224).to('cuda') for _ in range(8)]

# # 用GPU进行推理
# outputs = []
# with torch.no_grad():
#     for model, data in zip(models, input_data):
#         output = model(data)
#         outputs.append(output)

# # 打印输出
# for i, output in enumerate(outputs):
#     print(f"Output from model {i}: {output}")
