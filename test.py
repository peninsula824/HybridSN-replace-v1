import scipy.io as sio

# 加载.mat数据集
data_path = 'D:\Pycode\project\HybridSN-pytorch\dataset'  # 替换为您的数据集路径
mat_data = sio.loadmat(data_path)

# 打印数据集中每个键对应的数据形状
for key in mat_data:
    if not key.startswith('__'):  # 跳过内部元数据
        data = mat_data[key]
        print(f"Key: {key}, Shape: {data.shape}")

# 或者，如果您知道特定键对应的数据，您也可以直接打印其形状
# key = 'your_key'
# data = mat_data[key]
# print(f"Shape of {key}: {data.shape}")
