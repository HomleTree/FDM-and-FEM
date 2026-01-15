### Example1 ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import timeit

# 参数设置
epsilon = 0.00001 # 摄动参数
N = 600000 # 划分的网格数，通常要取较大值，以应对小epsilon
x = np.linspace(0, 1, N+1) # x划分的网格点，考虑到0也是一个点，所以N+1个点
h = 1.0 / N # 网格间距

# 右端项函数f(x)，这里取f(x)=1
def f(x):
    return np.cos(2 * np.pi * x)

# 边界条件
u0, u1 = 1, -1

# 初始化矩阵和右端项F
main_diag = np.zeros(N-1) # 主对角线元素
lower_diag = np.zeros(N-2) # 下对角线元素
upper_diag = np.zeros(N-2) # 上对角线元素
F = np.zeros(N-1) # 右端项

# 填充三对角线和右端项F
a = timeit.default_timer()
for i in range(1, N):
    i_row = i - 1
    xi = x[i]
    
    # 有限差分离散化
    a_im1 = epsilon / h**2 - 1/h
    a_i = -2 * epsilon / h**2 + 1/h
    a_ip1 = epsilon / h**2
    
    # 填充矩阵元素
    main_diag[i_row] = a_i
    if i > 1:
        lower_diag[i_row - 1] = a_im1
    if i < N-1:
        upper_diag[i_row] = a_ip1
    
    # 右端项及边界条件处理
    F[i_row] = f(xi)
    # 左边界
    if i == 1:
        F[i_row] -= a_im1 * u0
    # 右边界
    if i == N-1:
        F[i_row] -= a_ip1 * u1

# 构建并求解稀疏矩阵
A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
U_internal = spsolve(A, F)

# 统计计算时长
b = timeit.default_timer() - a
print('Time for compute %.2f s' % (b))

# 合并边界值
U = np.zeros(N+1)
U[0] = u0
U[-1] = u1
U[1:-1] = U_internal

plt.plot(x, U, 'b-', linewidth=1.5, label='numerical')
plt.xlabel('x', fontsize=12)
plt.ylabel('u', fontsize=12)
# plt.title(f'ε = {epsilon}', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# 计算残差
residual = np.zeros(N-1)
for i in range(1, N):
    xi = x[i]
    
    u_xx = (U[i+1] - 2*U[i] + U[i-1]) / h**2
    
    # u_x = (U[i+1] - U[i-1]) / (2 * h)  
    if i == 1:  # 左边界
        u_x = (U[i] - U[i-1]) / h # u_x ≈ (u_i - u_{i-1})/h
    elif i == N-1:  # 右边界
        u_x = (U[i+1] - U[i]) / h # u_x ≈ (u_{i+1} - u_i)/h
    else:  # 内部节点
        u_x = (U[i] - U[i-1]) / h
    
    residual[i-1] = epsilon * u_xx + u_x - f(xi)

print(f"修正后的最大残差：{np.max(np.abs(residual)):.2e}")

# 存储数据(x,u)
# np.savetxt('Example1.csv', np.column_stack((x, U)), delimiter=',')

### Example2 ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import timeit

# 参数设置
epsilon = 0.001  # 摄动参数
N = 100000  # 划分的网格数
x = np.linspace(0, 1, N+1)  # x划分的网格点
h = 1.0 / N  # 网格间距

# 右端项函数f(x)
def f(x):
    return 0  # 这里将右端项设置为1

# 边界条件
u0, u1 = 1, 1

# 初始化矩阵和右端项F
main_diag = np.zeros(N-1)  # 主对角线元素
lower_diag = np.zeros(N-2)  # 下对角线元素
upper_diag = np.zeros(N-2)  # 上对角线元素
F = np.zeros(N-1)  # 右端项

# 填充三对角线和右端项F
a = timeit.default_timer()
for i in range(1, N):
    i_row = i - 1
    xi = x[i]
    
    # 有限差分离散化
    b_i = epsilon / h**2 + xi**2 / h   # u_{i-1} 的系数
    a_i = -2 * epsilon / h**2 - xi**2 / h - 1  # u_i 的系数
    c_i = epsilon / h**2  # u_{i+1} 的系数
    
    # 填充矩阵元素
    main_diag[i_row] = a_i
    if i > 1:
        lower_diag[i_row - 1] = b_i
    if i < N-1:
        upper_diag[i_row] = c_i
    
    # 右端项及边界条件处理
    F[i_row] = f(xi)
    # 左边界
    if i == 1:
        F[i_row] -= b_i * u0
    # 右边界
    if i == N-1:
        F[i_row] -= c_i * u1

# 构建并求解稀疏矩阵
A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
U_internal = spsolve(A, F)

# 统计计算时长
b = timeit.default_timer() - a
print('Time for compute %.2f s' % (b))

# 合并边界值
U = np.zeros(N+1)
U[0] = u0
U[-1] = u1
U[1:-1] = U_internal

plt.plot(x, U, 'b-', linewidth=1.5, label='numerical')
plt.xlabel('x', fontsize=12)
plt.ylabel('u', fontsize=12)
plt.title(f'ε = {epsilon}, N = {N}', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# 计算残差
residual = np.zeros(N-1)
for i in range(1, N):
    xi = x[i]
    
    u_xx = (U[i+1] - 2*U[i] + U[i-1]) / h**2
    
    # 使用中心差分计算u_x
    if i == 1:  # 左边界
        u_x = (U[i+1] - U[i-1]) / (2 * h)
    elif i == N-1:  # 右边界
        u_x = (U[i] - U[i-2]) / (2 * h)
    else:  # 内部节点
        u_x = (U[i+1] - U[i-1]) / (2 * h)
    
    residual[i-1] = epsilon * u_xx - xi**2 * u_x - U[i]

print(f"修正后的最大残差：{np.max(np.abs(residual)):.2e}")

# 计算u_x并存储
u_x = np.zeros(N+1)  # 调整长度与x一致
for i in range(1, N):
    xi = x[i]
    coeff = xi**2
    
    if coeff > 0:
        u_x[i] = (U[i] - U[i-1]) / h  # 后向差分
    else:
        u_x[i] = (U[i+1] - U[i]) / h  # 前向差分

# 边界值处理
u_x[0] = (U[1] - U[0]) / h  # 左边界：前向差分
u_x[-1] = (U[-1] - U[-2]) / h  # 右边界：后向差分

# 对u_x取绝对值
u_x = abs(u_x)

plt.plot(x, u_x, 'b-', linewidth=1.5, label='numerical')
plt.xlabel('x', fontsize=12)
plt.ylabel('u_x', fontsize=12)
# plt.title(f'ε = {epsilon}', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# 存储数据(x,u)
# np.savetxt('Example1--What.csv', np.column_stack((x, U, u_x)), delimiter=',')

### Example3 ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import timeit

# 参数设置
epsilon = 1e-3
N = 100000  # 高网格数以应对小epsilon
x = np.linspace(0, 1, N+1)
h = 1.0 / N

# 新的右端项函数
def f(x):
    return -np.cos(2 * np.pi * x) #+ np.sin(2 * np.pi * x)

# 初始化矩阵和右端项
main_diag = np.zeros(N-1)
lower_diag = np.zeros(N-2)
upper_diag = np.zeros(N-2)
F = np.zeros(N-1)

# 填充三对角线和右端项F
a = timeit.default_timer()
for i in range(1, N):  # 内部节点i=1,2,...,N-1
    i_row = i - 1      # 矩阵行索引
    xi = x[i]
    coeff = (xi - 1)      # 对流项系数 (1 - x)
    
    # 迎风格式选择
    if coeff > 0:
        # 后向差分：u_x ≈ (u_i - u_{i-1})/h
        c_i = epsilon / h**2 - coeff / h
        a_i = (-2 * epsilon / h**2) + (coeff / h) 
        b_i = epsilon / h**2
    else:
        # 前向差分：u_x ≈ (u_{i+1} - u_i)/h
        c_i = epsilon / h**2
        a_i = (-2 * epsilon / h**2) - (coeff / h) - 1
        b_i = epsilon / h**2 + coeff / h
    
    # 填充矩阵元素
    main_diag[i_row] = a_i
    if i > 1:
        lower_diag[i_row - 1] = c_i
    if i < N-1:
        upper_diag[i_row] = b_i
    
    # 右端项及边界条件处理
    F[i_row] = f(xi)
    if i == 1:
        F[i_row] -= c_i * 1     # 左边界u(0)=1
    if i == N-1:
        F[i_row] -= b_i * (-1)  # 右边界u(2)=-1

# 构建并求解稀疏矩阵
A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
U_internal = spsolve(A, F)

# 统计计算时长
b = timeit.default_timer() - a
print('Time for compute %.2f s' % (b))

# 合并边界值
U = np.zeros(N+1)
U[0] = 1
U[-1] = -1
U[1:-1] = U_internal

# 绘制数值解
# plt.figure(figsize=(10, 6))
plt.plot(x, U, 'b-', linewidth=1.5, label='numerical')

plt.xlabel('x', fontsize=12)
plt.ylabel('u', fontsize=12)
# plt.title(f'ε = {epsilon}', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# 计算残差
residual = np.zeros(N-1)
for i in range(1, N):
    xi = x[i]
    coeff = (xi - 1) 
    
    u_xx = (U[i+1] - 2*U[i] + U[i-1]) / h**2
    
    if coeff > 0:
        u_x = (U[i] - U[i-1]) / h # u_x ≈ (u_i - u_{i-1})/h
    else:
        u_x = (U[i+1] - U[i]) / h # u_x ≈ (u_{i+1} - u_i)/h
        
    residual[i-1] = epsilon * u_xx + coeff * u_x - U[i] - f(xi)

print(f"修正后的最大残差：{np.max(np.abs(residual)):.2e}")

# 计算u_x并存储
u_x = np.zeros(N+1)  # 调整长度与x一致
for i in range(1, N):
    xi = x[i]
    coeff = (xi - 1) 
    
    if coeff > 0:
        u_x[i] = (U[i] - U[i-1]) / h  # 后向差分
    else:
        u_x[i] = (U[i+1] - U[i]) / h  # 前向差分

# 边界值处理
u_x[0] = (U[1] - U[0]) / h  # 左边界：前向差分
u_x[-1] = (U[-1] - U[-2]) / h  # 右边界：后向差分

# 对u_x取绝对值
u_x = abs(u_x)

plt.plot(x, u_x, 'b-', linewidth=1.5, label='numerical')
plt.xlabel('x', fontsize=12)
plt.ylabel('u_x', fontsize=12)
# plt.title(f'ε = {epsilon}', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
u_x.shape, x.shape
# 存储数据(x,u)
# np.savetxt('E2-1.csv', np.column_stack((x, U, u_x)), delimiter=',')

### Example4 ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import timeit

# 参数设置
epsilon = 1e-9
N = 20000000  # 高网格数以应对小epsilon
x = np.linspace(0, 2, N+1)
h = 2.0 / N

# 新的右端项函数
def f(x):
    return -np.cos(2 * np.pi * x) #+ np.sin(2 * np.pi * x)

# 初始化矩阵和右端项
main_diag = np.zeros(N-1)
lower_diag = np.zeros(N-2)
upper_diag = np.zeros(N-2)
F = np.zeros(N-1)

# 填充三对角线和右端项F
a = timeit.default_timer()
for i in range(1, N):  # 内部节点i=1,2,...,N-1
    i_row = i - 1      # 矩阵行索引
    xi = x[i]
    coeff = -(xi - 1)      # 对流项系数 (1 - x)
    
    # 迎风格式选择
    if coeff > 0:
        # 后向差分：u_x ≈ (u_i - u_{i-1})/h
        c_i = epsilon / h**2 - coeff / h
        a_i = (-2 * epsilon / h**2) + (coeff / h) - 1
        b_i = epsilon / h**2
    else:
        # 前向差分：u_x ≈ (u_{i+1} - u_i)/h
        c_i = epsilon / h**2
        a_i = (-2 * epsilon / h**2) - (coeff / h) - 1
        b_i = epsilon / h**2 + coeff / h
    
    # 填充矩阵元素
    main_diag[i_row] = a_i
    if i > 1:
        lower_diag[i_row - 1] = c_i
    if i < N-1:
        upper_diag[i_row] = b_i
    
    # 右端项及边界条件处理
    F[i_row] = f(xi)
    if i == 1:
        F[i_row] -= c_i * 1     # 左边界u(0)=1
    if i == N-1:
        F[i_row] -= b_i * (-1)  # 右边界u(2)=-1

# 构建并求解稀疏矩阵
A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
U_internal = spsolve(A, F)

# 统计计算时长
b = timeit.default_timer() - a
print('Time for compute %.2f s' % (b))

# 合并边界值
U = np.zeros(N+1)
U[0] = 1
U[-1] = -1
U[1:-1] = U_internal

# 绘制数值解
# plt.figure(figsize=(10, 6))
plt.plot(x, U, 'b-', linewidth=1.5, label='numerical')

# plt.xlabel('x', fontsize=12)
# plt.ylabel('u', fontsize=12)
# plt.title(f'ε = {epsilon}', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# 计算残差
residual = np.zeros(N-1)
for i in range(1, N):
    xi = x[i]
    coeff = -(xi - 1) 
    
    u_xx = (U[i+1] - 2*U[i] + U[i-1]) / h**2
    
    if coeff > 0:
        u_x = (U[i] - U[i-1]) / h # u_x ≈ (u_i - u_{i-1})/h
    else:
        u_x = (U[i+1] - U[i]) / h # u_x ≈ (u_{i+1} - u_i)/h
        
    residual[i-1] = epsilon * u_xx + coeff * u_x - U[i] - f(xi)

print(f"修正后的最大残差：{np.max(np.abs(residual)):.2e}")

# 计算u_x并存储
u_x = np.zeros(N+1)  # 调整长度与x一致
for i in range(1, N):
    xi = x[i]
    coeff = -(xi - 1)
    
    if coeff > 0:
        u_x[i] = (U[i] - U[i-1]) / h  # 后向差分
    else:
        u_x[i] = (U[i+1] - U[i]) / h  # 前向差分

# 边界值处理
u_x[0] = (U[1] - U[0]) / h  # 左边界：前向差分
u_x[-1] = (U[-1] - U[-2]) / h  # 右边界：后向差分

# 对u_x取绝对值
u_x = abs(u_x)

plt.plot(x, u_x, 'b-', linewidth=1.5, label='numerical')
# plt.xlabel('x', fontsize=12)
# plt.ylabel('u_x', fontsize=12)
# plt.title(f'ε = {epsilon}', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
u_x.shape, x.shape
# 存储数据(x,u)
# np.savetxt('E2--[-9].csv', np.column_stack((x, U, u_x)), delimiter=',')

### Example5 ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import timeit

# 参数设置
epsilon = 1e-3
N = 200000 # 高网格数以应对小epsilon
x = np.linspace(0, 1, N + 1)
h = 1.0 / N
# 左、右边界值
A = -1
B = 1

# 新的右端项函数
def f(x):
    return 0

# 选择一个满足边界条件的初始猜测值u0
u_initial = np.zeros(N + 1)
u_initial[0] = A
u_initial[-1] = B
# 由此，我们猜测u0 = 2x - 1
u_initial[1:-1] = 2 * x[1:-1] - 1
u_current = u_initial[1:-1].copy()

# 迭代求解
max_iter = 2000 # 最大迭代次数
tol = 1e-6 # 收敛容差

a = timeit.default_timer()
for iter in range(max_iter):
    
    # 计算残差F
    F = np.zeros(N - 1)
    
    for i in range(1, N):
        u_i = u_current[i - 1]
        
        # 考虑边界条件
        if i == 1:
            u_prev = A
        else:
            u_prev = u_current[i - 2]
            
        if i == N - 1:
            u_next = B
        else:
            u_next = u_current[i]
            
        # 扩散项：中心差分
        diffusion = epsilon * (u_next - 2 * u_i + u_prev) / h**2

        # 非线性项：迎风格式
        if u_i > 0:
            convection = u_i * (u_i - u_prev) / h
        else:
            convection = u_i * (u_next - u_i) / h
            
        # 残差计算
        F[i - 1] = diffusion + convection - u_i ** 2 - f(x[i])
        
    # 检查收敛性
    residual_norm = np.max(np.abs(F))
    print(f"Iteration {iter:2d}: Residual norm = {residual_norm:.4e}")
    if residual_norm < tol:
        print("Converged!")
        break
    
    # 构造雅可比矩阵
    main_diag = np.zeros(N - 1)
    lower_diag = np.zeros(N - 2)
    upper_diag = np.zeros(N - 2)
    
    for i in range(1, N):
        i_row = i - 1
        u_i = u_current[i - 1]
        
        # 扩散项贡献
        main_diag[i_row] = -2 * epsilon / h**2
        if i > 1:
            lower_diag[i_row - 1] = epsilon / h**2
        if i < N - 1:
            upper_diag[i_row] = epsilon / h**2
            
        # 对流项贡献
        if u_i > 0: # 使用后向差分
             
            if i == 1:
                u_prev = A
            else:
                u_prev = u_current[i - 2]
            
            main_diag[i_row] += (2 * u_i - u_prev) / h
            if i > 1:
                lower_diag[i_row - 1] += -u_i / h
        
        else: # 使用前向差分
            if i == N - 1:
                u_next = B
            else:
                u_next = u_current[i]
            
            main_diag[i_row] += (u_next - 2 * u_i) / h
            if i < N - 1:
                upper_diag[i_row] += u_i / h
         
        # -u项贡献
        main_diag[i_row] += -2 * u_i
    
    # 构建稀疏矩阵并求解
    J = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format="csr")
    delta_u = spsolve(J, -F)
    u_current += delta_u

# 统计计算时长
b = timeit.default_timer() - a
print('Time for compute %.2f s' % (b))

# 合并边界得到最终解
u_final = np.zeros(N + 1)
u_final[0] = A
u_final[-1] = B
u_final[1:-1] = u_current

# 绘制数值解
# plt.figure(figsize=(10, 6))
plt.plot(x, u_final, "b-", linewidth=1.5, label="Numerical")
plt.xlabel("x", fontsize=12)
plt.ylabel("u", fontsize=12)
# plt.title(f"Solution for ε = {epsilon}", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# 存储数据(x,u)
# np.savetxt("Example2-1,-1.csv", np.column_stack((x, u_final)), delimiter=",")

### Example6 ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import timeit

# 参数设置
epsilon = 1e-3
N = 200000  # 高网格数以应对小epsilon
x = np.linspace(0, 1, N + 1)
h = 1.0 / N
# 左、右边界值
A = -1
B = 1

# 新的右端项函数
def f(x):
    return 0

# 选择一个满足边界条件的初始猜测值u0
u_initial = np.zeros(N + 1)
u_initial[0] = A
u_initial[-1] = B
# 由此，我们猜测u0 = 2x - 1
u_initial[1:-1] = 2 * x[1:-1] - 1
u_current = u_initial[1:-1].copy()

# 迭代求解
max_iter = 2000  # 最大迭代次数
tol = 1e-8  # 收敛容差

a = timeit.default_timer()
for iter in range(max_iter):
    
    # 计算残差F
    F = np.zeros(N - 1)
    
    for i in range(1, N):
        u_i = u_current[i - 1]
        
        # 考虑边界条件
        if i == 1:
            u_prev = A
        else:
            u_prev = u_current[i - 2]
            
        if i == N - 1:
            u_next = B
        else:
            u_next = u_current[i]
            
        # 扩散项：中心差分
        diffusion = epsilon * (u_next - 2 * u_i + u_prev) / h**2

        # 非线性项：迎风格式
        if u_i > 0:
            convection = u_i * (u_i - u_prev) / h
        else:
            convection = u_i * (u_next - u_i) / h
            
        # 残差计算
        F[i - 1] = diffusion + convection - u_i - f(x[i])
        
    # 检查收敛性
    residual_norm = np.max(np.abs(F))
    print(f"Iteration {iter:2d}: Residual norm = {residual_norm:.4e}")
    if residual_norm < tol:
        print("Converged!")
        break
    
    # 构造雅可比矩阵
    main_diag = np.zeros(N - 1)
    lower_diag = np.zeros(N - 2)
    upper_diag = np.zeros(N - 2)
    
    for i in range(1, N):
        i_row = i - 1
        u_i = u_current[i - 1]
        
        # 扩散项贡献
        main_diag[i_row] = -2 * epsilon / h**2
        if i > 1:
            lower_diag[i_row - 1] = epsilon / h**2
        if i < N - 1:
            upper_diag[i_row] = epsilon / h**2
            
        # 对流项贡献
        if u_i > 0:  # 使用后向差分
            if i == 1:
                u_prev = A
            else:
                u_prev = u_current[i - 2]
            
            main_diag[i_row] += (2 * u_i - u_prev) / h
            if i > 1:
                lower_diag[i_row - 1] += -u_i / h
        
        else:  # 使用前向差分
            if i == N - 1:
                u_next = B
            else:
                u_next = u_current[i]
            
            main_diag[i_row] += (u_next - 2 * u_i) / h
            if i < N - 1:
                upper_diag[i_row] += u_i / h
         
        # -u项贡献
        main_diag[i_row] += -1

    # 构建稀疏矩阵并求解
    J = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format="csr")
    delta_u = spsolve(J, -F)
    u_current += delta_u

# 统计计算时长
b = timeit.default_timer() - a
print('Time for compute %.2f s' % (b))

# 合并边界得到最终解
u_final = np.zeros(N + 1)
u_final[0] = A
u_final[-1] = B
u_final[1:-1] = u_current

# 绘制数值解
plt.figure(figsize=(10, 6))
plt.plot(x, u_final, "b-", linewidth=1.5, label="Numerical")
plt.xlabel("x", fontsize=12)
plt.ylabel("u", fontsize=12)
plt.title(f"Solution for ε = {epsilon}", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# 计算梯度 u_x
u_x = np.zeros(N + 1)

# 使用中心差分法计算中间点的梯度
for i in range(1, N):
    u_x[i] = (u_final[i + 1] - u_final[i - 1]) / (2 * h)

# 对边界点使用前向和后向差分法
u_x[0] = (u_final[1] - u_final[0]) / h
u_x[-1] = (u_final[-1] - u_final[-2]) / h

# 绘制数值解的梯度图
plt.figure(figsize=(10, 6))
plt.plot(x, u_x, "r-", linewidth=1.5, label="Gradient u_x")
plt.xlabel("x", fontsize=12)
plt.ylabel("u_x", fontsize=12)
plt.title(f"Gradient of Solution for ε = {epsilon}", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# 存储数据(x,u)
# np.savetxt('E4--[-3].csv', np.column_stack((x, u_final, u_x)), delimiter=',')

### Example7 ###

epsilon = 0.001
Nx = 999
Ny = 999

# 定义右端项函数
def f(x, y):
    return 0.0

# 求解偏微分方程
x, y, u = solve_pde(epsilon, Nx, Ny, f=f)

# 可视化
X, Y = np.meshgrid(x, y, indexing='ij')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, u, cmap='jet', rstride=1, cstride=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.set_zlabel('U')
ax.tick_params(axis='z', pad=4)
ax.tick_params(axis='x', pad=-5)
ax.tick_params(axis='y', pad=2)
ax.locator_params(axis='z', nbins=5)
# plt.title('Solution of the Convection-Diffusion Equation')
fig.colorbar(surf, ax=ax)
plt.show()

# 存储数据(x,y,u)
# data = np.column_stack((X.flatten(), Y.flatten(), u.flatten()))
# np.savetxt("abab.csv", data, delimiter=",", header="x, y, u", comments="")

### Example8 ###
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import timeit
def solve_pde(epsilon, Nx, Ny, f=None):
    dx = 1.0 / (Nx + 1)
    dy = 1.0 / (Ny + 1)
    
    x = np.linspace(0, 1, Nx + 2)
    y = np.linspace(0, 1, Ny + 2)
    
    # 初始化矩阵和右侧向量
    A = lil_matrix((Nx * Ny, Nx * Ny))
    b = np.zeros(Nx * Ny)
    
    for i in range(1, Nx + 1):
        for j in range(1, Ny + 1):
            k = (i - 1) * Ny + (j - 1)
            b_k = 0.0

            x_val = x[i]
            y_val = y[j]
            
            # 自身系数
            A[k, k] = -2 * epsilon / dx**2 - 2 * epsilon / dy**2 + 1/dx + 1/dy
            
            # 右侧节点 (i+1, j)
            if i < Nx:
                A[k, k + Ny] = epsilon / dx**2
            else:
                # 右边界条件 u=2*sin(πy)
                b_k += -epsilon / dx**2 * 2 * np.sin(np.pi * y_val)
            
            # 左侧节点 (i-1, j)
            if i > 1:
                A[k, k - Ny] = epsilon / dx**2 - 1/dx 
            else:
                # 左边界条件 u=sin(πy)
                b_k += (-epsilon / dx**2 + 1/dx) * np.sin(np.pi * y_val)
            
            # 上方节点 (i, j+1)
            if j < Ny:
                A[k, k + 1] = epsilon / dy**2
            
            # 下方节点 (i, j-1)
            if j > 1:
                A[k, k - 1] = epsilon / dy**2 - 1/dy
            
            # 添加右端项 f(x, y)
            if f is not None:
                b_k += f(x_val, y_val)
            
            b[k] = b_k
    
    # 转换矩阵格式并求解
    A = A.tocsr()
    u = spsolve(A, b)
    
    # 构建包含边界的解矩阵
    u_total = np.zeros((Nx + 2, Ny + 2))
    u_total[1:-1, 1:-1] = u.reshape(Nx, Ny)
    
    # 填充边界
    u_total[:, 0] = 0
    u_total[:, -1] = 0
    for j in range(Ny + 2):
        y_val = y[j]
        u_total[0, j] = np.sin(np.pi * y_val)
        u_total[-1, j] = 2 * np.sin(np.pi * y_val)
    
    return x, y, u_total

# 参数设置
epsilon = 0.001
Nx = 999
Ny = 999

# 定义右端项函数
def f(x, y):
    return 0.0

# 求解偏微分方程
x, y, u = solve_pde(epsilon, Nx, Ny, f=f)

# 可视化
X, Y = np.meshgrid(x, y, indexing='ij')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, u, cmap='jet', rstride=1, cstride=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
# ax.set_zlabel('U')
ax.view_init(elev=30, azim=-135)
ax.tick_params(axis='z', pad=4)
ax.tick_params(axis='x', pad=-5)
ax.tick_params(axis='y', pad=2)
ax.locator_params(axis='z', nbins=5)
# plt.title('Solution of the Convection-Diffusion Equation')
fig.colorbar(surf, ax=ax)
plt.show()

### Example9 ###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import diags
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import timeit

# 参数设置
epsilon = 0.001
L = 1.0
T = 1.0
Nx = 1200
Nt = 6000
dx = L / Nx
dt = T / Nt

x = np.linspace(0, L + dx, Nx)
t = np.linspace(0, T + dt, Nt+1)

X, T = np.meshgrid(x, t)
# 初始化解数组
u = np.zeros((Nt + 1, Nx))
u[0, :] = np.cos(2 * np.pi * X[0, :]) # 初始条件

# 时间步进
for n in range(Nt):
    for i in range(1, Nx - 1):
        # 计算对流项 u_x
        u_x = (u[n, i + 1] - u[n, i - 1]) / (2 * dx)
        # 计算扩散项 u_xx
        u_xx = (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / (dx**2)
        # 更新下一个时间步的值
        u[n + 1, i] = u[n, i] + dt * (epsilon * u_xx + u_x + u[n, i])
    
    # 更新边界条件
    u[n + 1, 0] = 0  # 左边界
    u[n + 1, -1] = 1  # 右边界

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
surf = ax.pcolormesh(T, X, u, cmap='jet', edgecolor='none')
norm = mcolors.Normalize(vmin=u.min(), vmax=u.max())
fig.colorbar(surf, norm=norm)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_title('Exact u(x, y)')
plt.show()

# 存储数据(x,t,u)
# data = np.column_stack((X.flatten(), T.flatten(), u.flatten()))
# np.savetxt("bcbc.csv", data, delimiter=",", header="x, t, u", comments="")

### Example10 ###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 参数设置
nu = 0.0001   # 扩散系数
L = 2.0            # 空间范围 [-1, 1]
dx = 1 / 6400         # 空间步长
Nx = int(L / dx)            # 空间网格数
x = np.linspace(-1, 1, Nx )  # 空间网格点

T = 1.0            # 总时间
dt = 0.0001         # 时间步长
Nt = int(T / dt)  # 时间步数

t = np.linspace(0, T, Nt )  # 时间网格点
X, T = np.meshgrid(x, t)  # 生成网格

# 初始条件
u = -np.sin(np.pi * x)
u[0] = u[-1] = 0.0  # 应用边界条件

# 存储解以绘制3D结果
u_all = np.zeros((Nt , Nx ))  # 初始化存储数组
u_all[0, :] = u.copy()  # 保存初始条件

# 时间迭代
for n in range(Nt-1):
    u_prev = u.copy()
    # 计算内部点的导数（使用中心差分）
    u_x = (u_prev[2:] - u_prev[:-2]) / (2 * dx)          # 对流项一阶导数
    u_xx = (u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]) / dx**2  # 扩散项二阶导数
    # 更新内部点
    u[1:-1] = u_prev[1:-1] + dt * (-u_prev[1:-1] * u_x + nu * u_xx)
    # 强制边界条件
    u[0], u[-1] = 0.0, 0.0
    u_all[n+1, :] = u.copy()

# 绘制3D结果
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
surf = ax.pcolormesh(T, X, u_all, cmap='jet', edgecolor='none')
norm = mcolors.Normalize(vmin=u_all.min(), vmax=u_all.max())
fig.colorbar(surf, norm=norm)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_title('Exact u(x, y)')
plt.show()

# 存储数据(x,t,u)
# data = np.column_stack((X.flatten(), T.flatten(), u_all.flatten()))
# np.savetxt("burgers-2.csv", data, delimiter=",", header="x, t, u", comments="")