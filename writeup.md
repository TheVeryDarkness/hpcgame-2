# Writeup

代码仓库将放在<https://github.com/TheVeryDarkness/hpcgame-2.git>，Writeup 提交截止后会公开。

|       |  A  |  B  |  C  |  D  |  E  |  F  |  G  |  H  |  I  |  J  |  K  |  L  | Total |
| :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :---: |
| ≥﹏≤  | 100 | 100 | 100 | 100 |  0  | 10  | 150 |  0  | 40  |  0  | 50  |  0  |  650  |
| Total | 100 | 100 | 100 | 100 | 150 | 200 | 200 | 260 | 200 | 100 | 50  | 200 | 1760  |

写 Writeup 的时候发现的 Copilot 冥场面：![Copilot 冥场面](./copilot.png)

~~我好菜啊~~

## 签到

在火车上用手机翻的。不过看到其中几位应该就能猜出来是什么了。

## 小北问答

仓库里会有当时用的 [Jupiter Notebook](./b.ipynb)。

一开始没注意到提交结果里会有每一道题的分数，所以中间瞎猜了两次。

### 鸡兔同笼

列方程手算即可。

$$
\begin{aligned}
2b+s &= 16 \\
b+s  &= 12 \\
b    &= 4  \\
s    &= 8  \\
\end{aligned}
$$

### 编程语言

找个编译器编译一下就知道了。而且 GPT 也能回答这个问题。

![编译警告](./b-pl-clang.png)

![GPT 回答](./b-pl-gpt.png)

从类型论的角度来说，如果这两者兼容的话，可以往 `void **` 指向的地方写入 `const void *`，然后再作为 `void *` 读出来。

### CPU Architecture

能搜到，而且 GPT 也知道。

<https://developer.arm.com/documentation/102340/0100/Introducing-SVE2>

![GPT 回答](./b-arch-gpt.png)

### MISC

<https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>

### 储存

上网搜完大部分选项都能判断。比如说 <https://zhuanlan.zhihu.com/p/669385346>

![GPT 回答](./b-zns-gpt.png)

GPT 4o 问答链接（前两条）：<https://poe.com/s/7Df2bqQEvKMbPnMcVcrp>

换别的大模型给的答案不对。

### OpenMPI

<https://www.open-mpi.org/software/ompi/v5.0/>

<https://github.com/open-mpi/ompi/commit/8a5c2ef25dc8e4528f0d3fd2ec91a6578160af95>

~~<https://docs.open-mpi.org/en/main/installing-open-mpi/required-support-libraries.html>~~

### RDMA

<https://zhuanlan.zhihu.com/p/649468433>

![RDMA](https://pic4.zhimg.com/v2-7f767472a1ae51dcc51a39b5fc8b5c8f_1440w.jpg)

### HPCKit

- HMPI <- OpenMPI
- KBLAS <- BLAS
- EXAGEAR 模拟器

### CXL

懒得想了，直接问大模型。

GPT 4o 问答链接（后两条）：<https://poe.com/s/7Df2bqQEvKMbPnMcVcrp>

![GPT 回答](./b-cxl-gpt.png)

换别的大模型给的答案不对。

### 量子计算

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

## 不简单的编译

_众所周知_，通常情况下，优化最好的编译器是 intel 的 icc/ifort，所以选择 intel 环境。

~~但是被后缀名晃晕了，以为这题给的源代码就是 Fortran 的，直接改 CMakeLists.txt 里的语言编译不了。~~

_众所周知_，对于科学计算而言，Fortran 一般比 C 优化得更好，所以选择 Fortran 语言。

然后用大模型根据原来的代码生成了个初版：

![GPT 回答](./c-gpt.png)

接下来就是改代码了。几个要点：

- 在编译选项里指定更高的优化等级和目标架构。
- CMake 里指定语言为 Fortran 而不是 FORTRAN。（CMake 大部分情况大小写不敏感，之前也没在 CMake 里用过 Fortran，脑子抽了）
- 要保证找得到符号的话，需要在 Fortran 里声明 `bind(C)`。部分情况下还需要指定符号的名字。
- C 中的二维数组是按行存储的，Fortran 中的二维数组是按列存储的，传入时的尺寸要换顺序。
- 默认情况下 Fortran 中数组是从 1 开始的，所以手动指定下标范围。（不过我看我之前提交的代码里有一个数组下标上界忘记减一了，还好是传指针）
- Fortran 默认按引用传参的，有几个非数组的参数需要指定成值传递。
- 用 `intent` 用来指定参数的读写权限。（没啥用，单纯怕自己不小心写错）

<https://gcc.gnu.org/onlinedocs/gfortran/Argument-passing-conventions.html>

<https://docs.oracle.com/cd/E19205-01/820-1204/6nct259sc/index.html>

## 最长公共子序列

上网搜了一下，发现[南京大学的 Lab](https://xliuqq.github.io/csblog/courses/os_lab/M2_plcs.html)里有这题，只需要换一下方向就能并行了。

然后分了三个阶段（假设 x 轴正方向从下到上，y 轴正方向从右到左，下面的数字表示阶段，假设 $n_1 = 2, n_2 = 4$）：

```txt
3 3 2 2 1
3 2 2 1 1
2 2 1 1 1
```

- 第一阶段为右下角的三角形，共计 $n_1 + 1$ 个斜行，由于右边界、下边界固定为 0，可以跳过前两个斜行。
- 第二阶段为平行四边形（如果 $n_1 > n_2$ 的话才有这一阶段），共计 $n_1 + n_2 - 1$ 个斜行。
- 第三阶段为左上角的三角形，共计 $n_2$ 个斜行。

共计 $n_1 + n_2 + 1$ 个斜行，循环时只需要保存前两个斜行的信息即可。

记第 $t$ 个斜行保存的信息为 $l_t$，$l_t[x] = dp[x][t - x], \max\{0, t-n_2-1\} \le x \le \min\{n_1, t\}$，即 `arr_1[0:x]` 和 `arr_2[0:t-x]` 的最长公共子序列长度。则

$$
\begin{aligned}
    l_{t+1}[x] &= \max \begin{cases}
        0                      & \text{if } x = 0 \text{ or } x = t \\
        l_t[x-1] + 1           & arr_1[x] = arr_2[t-x]              \\
        \max(l_t[x-1], l_t[x]) & \text{otherwise}
    \end{cases}
\end{aligned}
$$

最开始用 `#pragma omp parallel for schedule(static)`，发现拿不了满分，然后转念一想，负载应该不会太均衡，改成 `#pragma omp parallel for schedule(dynamic)`，就能拿到满分了。

## 着火的森林

~~写了半天一分没拿到，寄！~~

## 雷方块

~~写了半天只拿到 10 分，发现自己还是太年轻了，错把 baseline 当成宝。~~
