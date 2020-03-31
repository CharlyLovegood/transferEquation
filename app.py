import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt


from mpi4py import MPI

#################################################
# Явная центральная трехточечная схема
# ∂u(t,x)/∂t + a∙∂u(t,x)/∂x = f(t,x), 0≤t≤T, 0≤x≤X
# u(0,x) = φ(x), 0≤x≤X
# u(t,0) = ψ(t), 0≤t≤T
#################################################

if __name__ == "__main__":
    a = 0.25; T = 2; X = 1

    φ = lambda x: (x*x*x)/(12*a*a)
    ψ = lambda t: (a*t*t*t)/12
    f = lambda t, x: t*x
    
    h = 0.1
    τ = 0.1
    # τ = h/a # Условие Куранта выполняется
    # τ = 1.2 * h/a # Условие Куранта не выполняется
    t = np.arange(0, T + τ, τ)
    x = np.arange(0, X + h, h)

    y = [0] * len(t)
    for i in range(len(t)):
        y[i] = [0] * len(x)

    for j in range(0, len(x)):
        y[0][j] = φ(x[j]) # определяем граничные значения для х: u(0,x) = φ(x)
    
    for i in range(0, len(t)):
        y[i][0] = ψ(t[i]) # определяем граничные значения для t: u(t,0) = ψ(t)

    for i in range(0, len(t) - 1):
        for j in range(1, len(x) - 1):
            y[i+1][j] = 0.5*(y[i][j+1] + y[i][j-1]) - 0.5*(τ/h)*(y[i][j+1] - y[i][j-1]) + τ*f(t[i], x[j])
        y[i+1][len(x) - 1] = (1-τ)*y[i][len(x) - 1] + τ*(y[i][len(x) - 2]) + τ*(t[i])*(x[len(x) - 1])

    fig = plt.figure()
    print(y)
    ax = fig.add_subplot(projection='3d')
    x, t = np.meshgrid(x, t)
    u = np.array(y)
    ax.plot_surface(x, t, u)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.show()

    # python3 app.py