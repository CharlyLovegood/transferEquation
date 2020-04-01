import numpy as np
import math
import pickle
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpi4py import MPI


class time_tracker():
    start_time, end_time = time.time(), time.time()

    def finish(self):
        self.end_time = time.time()
        ex_time = self.end_time - self.start_time
        print("Execution time = {sec} seconds".format(sec=ex_time))


########
# Явная центральная трехточечная схема
# ∂u(t,x)/∂t + a∙∂u(t,x)/∂x = f(t,x), 0≤t≤T, 0≤x≤X
# u(0,x) = φ(x), 0≤x≤X
# u(t,0) = ψ(t), 0≤t≤T
########
if __name__ == "__main__":
    a = 0.25; T = 4; X = 2

    φ = lambda x: (x*x*x)/(12*a*a)
    ψ = lambda t: (a*t*t*t)/12
    f = lambda t, x: t*x
    
    h = 0.001
    τ = 0.001
    # τ ≤ h # Условие Куранта выполняется
    # τ > h # Условие Куранта не выполняется
    t = np.arange(0, T + τ, τ)
    x = np.arange(0, X + h, h)

    comm = MPI.COMM_WORLD
    P = comm.Get_size()
    rank = comm.Get_rank()
    zone = math.floor(len(x)/(P))

    execution = time_tracker()

    y = [0] * len(t)
    for i in range(len(t)):
        y[i] = [0] * len(x)

    for j in range(0, len(x)):
        y[0][j] = φ(x[j]) # определяем граничные значения для х: u(0,x) = φ(x)
    
    for i in range(0, len(t)):
        y[i][0] = ψ(t[i]) # определяем граничные значения для t: u(t,0) = ψ(t)

    # Если число процессов = 1
    if (P == 1):
        for i in range(0, len(t) - 1):
            for j in range(1, len(x) - 1):
                y[i+1][j] = 0.5*(y[i][j+1] + y[i][j-1]) - 0.5*(τ/h)*(y[i][j+1] - y[i][j-1]) + τ*f(t[i], x[j])
            y[i+1][len(x) - 1] = (1-(τ/h))*y[i][len(x) - 1] + (τ/h)*(y[i][len(x) - 2]) + τ*(t[i])*(x[len(x) - 1])
        
        execution.finish()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x, t = np.meshgrid(x, t)
        u = np.array(y)
        ax.plot_surface(x, t, u)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.show()


    # Если число процессов != 1
    if (rank == 0): # в первой зоне
        start = 1
        end = zone - 1
        right = y[0][zone]
        for i in range(0, len(t) - 1):
            for j in range(start, end):
                y[i+1][j] = 0.5*(y[i][j+1] + y[i][j-1]) - 0.5*(τ/h)*(y[i][j+1] - y[i][j-1]) + τ*f(t[i], x[j])
            y[i+1][end] = 0.5*(right + y[i][end-1]) - 0.5*(τ/h)*(right - y[i][end-1]) + τ*f(t[i], x[end])
            comm.isend(y[i+1][end], dest=1, tag=0)
            right = comm.recv(None, source=1, tag=0)

        comm.barrier() # синхронизация с остальными процессами
        b = pickle.dumps(y)
        comm.send(b, dest=P-1, tag=0)

    if (rank == P-1): # в последней зоне
        start = zone*rank + 1
        end = len(x) - 1
        left = y[0][zone*rank-1]
        for i in range(0, len(t) - 1):
            y[i+1][zone*rank] = 0.5*(y[i][start] + left) - 0.5*(τ/h)*(y[i][start] - left) + τ*f(t[i], x[zone*rank])
            for j in range(start, end):
                y[i+1][j] = 0.5*(y[i][j+1] + y[i][j-1]) - 0.5*(τ/h)*(y[i][j+1] - y[i][j-1]) + τ*f(t[i], x[j])
            y[i+1][len(x) - 1] = (1-τ/h)*y[i][len(x) - 1] + (τ/h)*(y[i][len(x) - 2]) + τ*(t[i])*(x[len(x) - 1])
            comm.isend(y[i+1][zone*rank], dest=P-2, tag=0)
            left = comm.recv(None, source=P-2, tag=0)

        comm.barrier() # синхронизация с остальными процессами

    else:
        for process in range(1, P-1):
            if (process == rank):
                ########
                # даем каждому процессу работать над своей зоной
                ########
                start = zone*rank + 1
                end = zone*(rank+1) - 1
                left = y[0][zone*rank-1]
                right = y[0][zone]
                for i in range(0, len(t) - 1):
                    y[i+1][start-1] = 0.5*(y[i][start] + left) - 0.5*(τ/h)*(y[i][start] - left) + τ*f(t[i], x[start])
                    for j in range(start, end):
                        y[i+1][j] = 0.5*(y[i][j+1] + y[i][j-1]) - 0.5*(τ/h)*(y[i][j+1] - y[i][j-1]) + τ*f(t[i], x[j])
                    y[i+1][end] = 0.5*(right + y[i][end-1]) - 0.5*(τ/h)*(right - y[i][end-1]) + τ*f(t[i], x[end])
                    
                    comm.isend(y[i+1][zone*rank], dest=rank-1, tag=0)
                    comm.isend(y[i+1][end], dest=rank+1, tag=0)
                    left = comm.recv(None, source=rank-1, tag=0)
                    right = comm.recv(None, source=rank+1, tag=0) 
                    
                comm.barrier() # синхронизация с остальными процессами
                b = pickle.dumps(y)
                comm.send(b, dest=P-1, tag=0)         


    ########
    # работа последнего процесса
    # собираем данные из остальных процессов 
    ########
    if (rank == P-1):
        for process in range(0, P-1):
            b = comm.recv(None, source=process, tag=0)
            rec = pickle.loads(b)

            start = process*zone
            end = (process+1)*zone

            for i in range(len(t) - 1):
                for j in range(start, end):
                    y[i+1][j] = rec[i+1][j]

        execution.finish()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x, t = np.meshgrid(x, t)
        u = np.array(y)
        ax.plot_surface(x, t, u)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.show()