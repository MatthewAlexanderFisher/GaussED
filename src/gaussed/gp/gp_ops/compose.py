

class Compose:
    def __init__(self, L2, L1): self.L2, self.L1 = L2, L1
    name = "compose"
    def apply_mean(self, m):    return self.L2.apply_mean(self.L1.apply_mean(m))
    def apply_kernel(self, K):  return self.L2.apply_kernel(self.L1.apply_kernel(K))
    def apply_kernel_left(self, K):  return self.L2.apply_kernel_left(self.L1.apply_kernel_left(K))
    def apply_kernel_right(self, K): return self.L2.apply_kernel_right(self.L1.apply_kernel_right(K))
