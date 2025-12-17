import timeit
import io
import numpy as np

# 准备测试数据
shape = (480, 640, 3)
original_arr = np.random.randint(0, 255, size=shape, dtype=np.uint8)

# 方式 A 的数据源：带 Header 的 NPY 格式 (模拟 np.save 的结果)
bio = io.BytesIO()
np.save(bio, original_arr)
npy_bytes = bio.getvalue()

# 方式 B 的数据源：纯二进制流 (模拟 tobytes 的结果)
raw_bytes = original_arr.tobytes()

def test_load_bytes():
    # 模拟：创建一个新的 BytesIO 对象 + 解析 NPY Header + 内存拷贝
    f = io.BytesIO(npy_bytes)
    arr = np.load(f)

def test_from_buffer():
    # 模拟：直接指针映射 + Reshape (几乎无 CPU 开销)
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape)

# 执行测试 (运行 100,000 次)
t1 = timeit.timeit(test_load_bytes, number=10000)
t2 = timeit.timeit(test_from_buffer, number=10000)

print(f"np.load(BytesIO):  {t1:.4f} 秒")
print(f"np.frombuffer:     {t2:.4f} 秒")
print(f"性能倍数差异:       {t1/t2:.1f}x (倍)")