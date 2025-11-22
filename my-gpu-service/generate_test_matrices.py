import numpy as np

np.savez_compressed("matrix_a.npz", np.random.rand(512,512).astype(np.float32))
np.savez_compressed("matrix_b.npz", np.random.rand(512,512).astype(np.float32))

print("DONE: matrix_a.npz and matrix_b.npz created")
