1. `4` `8`
2. 否。
3. `128` `2048`
4.
5. `BCDE`

# 鸡兔同笼

$$
\begin{aligned}
2b+s &= 16 \\
b+s  &= 12 \\
b    &= 4  \\
s    &= 8  \\
\end{aligned}
$$

# 编程语言

```c
void f(const void **p) {}
int main() {
    void *p = 0;
    void **q = &p;
    f(q);
}
```

# MISC

```python
import matplotlib.pyplot as plt
import numpy as np
from bitsandbytes import functional as bf


length = np.pi * 4
resolution = 256
xvals = np.arange(0, length, length / resolution)
wave = np.sin(xvals)

x_4bit, qstate = bf.quantize_fp4(torch.tensor(wave, dtype=torch.float32, device=device), blocksize=64)
dq = bf.dequantize_fp4(x_4bit, qstate)

plt.rcParams["figure.figsize"] = (14, 5)
plt.title('FP8 Sine Wave')
plt.plot(xvals, wave)
plt.plot(xvals, dq.cpu().numpy())
plt.show()
```

# 储存

<https://zhuanlan.zhihu.com/p/669385346>

# 量子计算

$$
\begin{aligned}
    H &= \frac{1}{\sqrt{2}} \begin{bmatrix}
        1 &  1 \\
        1 & -1 \\
    \end{bmatrix} \\
    H \ket{0} &= \frac{1}{\sqrt{2}} (\ket{0} + \ket{1}) \\
    H^2 &= E \\
    H^2 \ket{0} &= \ket{0} \\
\end{aligned}
$$
