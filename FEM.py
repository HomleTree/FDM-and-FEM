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
from fenics import *
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Create mesh
nx = ny = 1200
mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)

# Define function space
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def boundary_bottom_top(x, on_boundary):
    return on_boundary and (near(x[1], 0) or near(x[1], 1))

def boundary_left(x, on_boundary):
    return on_boundary and near(x[0], 0)

def boundary_right(x, on_boundary):
    return on_boundary and near(x[0], 1)

bc_bottom_top = DirichletBC(V, Constant(0), boundary_bottom_top)
bc_left = DirichletBC(V, Expression('sin(pi*x[1])', degree=2), boundary_left)
bc_right = DirichletBC(V, Expression('2*sin(pi*x[1])', degree=2), boundary_right)

bcs = [bc_bottom_top, bc_left, bc_right]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

a = 0.001 * dot(grad(u), grad(v)) * dx + u.dx(0) * v * dx
L = f * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# 提取 FEniCS 函数的值到 NumPy 数组中
u_values = u.compute_vertex_values(mesh)
u_array = np.reshape(u_values, (ny+1, nx+1))  # 重新整形为二维数组

# 创建网格
X, Y = np.meshgrid(np.linspace(0, 1, nx+1), np.linspace(0, 1, ny+1))

# 绘制三维曲面
fig = plt.figure(figsize=(24, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, u_array, cmap='jet', edgecolor='none')
norm = mcolors.Normalize(vmin=u_array.min(), vmax=u_array.max())
cbar = fig.colorbar(surf, norm=norm)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y)')
ax.set_title('3D Surface plot of u(x, y)')
ax.view_init(elev=15, azim=-65)  

# 显示图形
plt.show()

# data = np.column_stack((mesh.coordinates(), u_values))
# np.savetxt("4-0.001-1.csv", data, delimiter=",", header="x, y, u", comments="")

### Example3 ###
import numpy as np
from fenics import *
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Create mesh
nx = ny = 1200
mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)

# Define function space
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def boundary_bottom_top(x, on_boundary):
    return on_boundary and (near(x[0], 0) or near(x[0], 1))

def boundary_left(x, on_boundary):
    return on_boundary and near(x[1], 0)

def boundary_right(x, on_boundary):
    return on_boundary and near(x[1], 1)

bc_bottom_top = DirichletBC(V, Constant(0), boundary_bottom_top)
bc_left = DirichletBC(V, Expression('2*sin(pi*x[0])', degree=2), boundary_left)
bc_right = DirichletBC(V, Expression('sin(pi*x[0])', degree=2), boundary_right)

bcs = [bc_bottom_top, bc_left, bc_right]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

a = -0.001 * dot(grad(u), grad(v)) * dx + u.dx(1) * v * dx
L = f * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# 提取 FEniCS 函数的值到 NumPy 数组中
u_values = u.compute_vertex_values(mesh)
u_array = np.reshape(u_values, (ny+1, nx+1))  # 重新整形为二维数组

# 创建网格
X, Y = np.meshgrid(np.linspace(0, 1, nx+1), np.linspace(0, 1, ny+1))

# 绘制三维曲面
fig = plt.figure(figsize=(24, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, u_array, cmap='jet', edgecolor='none')
norm = mcolors.Normalize(vmin=u_array.min(), vmax=u_array.max())
cbar = fig.colorbar(surf, norm=norm)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y)')
ax.set_title('3D Surface plot of u(x, y)')
ax.view_init(elev=30, azim=-70)  

# 显示图形
plt.show()

# data = np.column_stack((mesh.coordinates(), u_values))
# np.savetxt("5—0.001—1.csv", data, delimiter=",", header="x, y, u", comments="")

### Example4 ###
import numpy as np
from fenics import *
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Create mesh
nx = ny = 1200
mesh = RectangleMesh(Point(0, 0), Point(1, 1), nx, ny)

# Define function space
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def boundary_bottom_left(x, on_boundary):
    return on_boundary and (near(x[0], 0) or near(x[0], 1))

def boundary_top(x, on_boundary):
    return on_boundary and near(x[1], 0)

def boundary_down(x, on_boundary):
    return on_boundary and near(x[1], 1)

bc_bottom_left = DirichletBC(V, Constant(0), boundary_bottom_left)
bc_top = DirichletBC(V, Expression('2*sin(pi*x[0])', degree=2), boundary_top)
bc_down = DirichletBC(V, Expression('1*sin(pi*x[0])', degree=2), boundary_down)

bcs = [bc_top, bc_down, bc_bottom_left]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

a = -0.001 * dot(grad(u), grad(v)) * dx + u.dx(0) * v * dx + u.dx(1) * v * dx
L = f * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# 提取 FEniCS 函数的值到 NumPy 数组中
u_values = u.compute_vertex_values(mesh)
u_array = np.reshape(u_values, (ny+1, nx+1))  # 重新整形为二维数组

# 创建网格
X, Y = np.meshgrid(np.linspace(0, 1, nx+1), np.linspace(0, 1, ny+1))

# 绘制三维曲面
fig = plt.figure(figsize=(24, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, u_array, cmap='jet', edgecolor='none')
norm = mcolors.Normalize(vmin=u_array.min(), vmax=u_array.max())
cbar = fig.colorbar(surf, norm=norm)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y)')
ax.set_title('3D Surface plot of u(x, y)')
ax.view_init(elev=15, azim=-105)  


# 显示图形
plt.show()

# data = np.column_stack((mesh.coordinates(), u_values))
# np.savetxt("6—0.001—1.csv", data, delimiter=",", header="x, y, u", comments="")

### Example5 ###
import numpy as np
from fenics import *
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Create mesh
nx = 10000
mesh = IntervalMesh(nx, -1, 1)

# Define function space
V = FunctionSpace(mesh, 'P', 1)

def boundary_left(x, on_boundary):
    return on_boundary and near(x[0], -1)

def boundary_right(x, on_boundary):
    return on_boundary and near(x[0], 1)

# Define boundary conditions
bc_left = DirichletBC(V, Constant(1), boundary_left)
bc_right = DirichletBC(V, Constant(-2), boundary_right)
bc = [bc_left, bc_right]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)

x = SpatialCoordinate(mesh)
a = 1e-3 * dot(grad(u), grad(v)) * dx + x[0] * u.dx(0) * v * dx
L = sin(2*pi*x[0]) * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution if needed
plot(u)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution u(x)')
plt.show()