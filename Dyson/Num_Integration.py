#!/usr/bin/env python3

import numpy as np


def lorenzSTD(u):
    sigma=10
    rho=28
    beta=8/3
    x,y,z = u
    dudt = np.array([sigma * (y-x), rho * x - y - x*z, x*y - beta*z])
    return dudt

def lorenzRho42(u):
    sigma=10
    rho=42
    beta=8/3
    x,y,z = u
    dudt = np.array([sigma * (y-x), rho * x - y - x*z, x*y - beta*z])
    return dudt

def STDlorenzTimeSeries(num,u0,dt,version):
    if version == "E":
        return EulerMaruyama(lorenzSTD,u0,dt,num)
    elif version == "R":
        return RK4(lorenzSTD,u0,dt,num)
    else:
         raise Exception('Set the final value to E for Euler, or R for RK4')

def Rho42lorenzTimeSeries(num,u0,dt,version):
    if version == "E":
        return EulerMaruyama(lorenzRho42,u0,dt,num)
    elif version == "R":
        return RK4(lorenzRho42,u0,dt,num)
    else:
         raise Exception('Set the final value to E for Euler, or R for RK4')

def EulerMaruyama(func, u0, dt,Nsteps):
    u = np.empty((Nsteps + 1,3))
    u[0] = u0

    # System data
    for i in range(Nsteps):
        u[i+1] = u[i] + func(u[i])*dt
    return u

"""
Defines the RK4 method of integration (4th Order) (DON'T TOUCH)
"""
def RK4(func, u0, dt, Nsteps):
    d = len(u0)
    u = np.empty((Nsteps + 1,d))
    u[0] = u0
    for i in range(Nsteps):
        u[i+1] = RK4OneStep(func, u[i], dt)
    return u

"""
Does 1 time step of RK4 (DON'T TOUCH)
"""
def RK4OneStep(func,u,dt):
    l = len(u)
    k1xyz = np.array([dt*(func(u))[i] for i in range(l)])
    k2xyz = np.array([dt*(func(u + k1xyz/2))[i] for i in range(l)])
    k3xyz = np.array([dt*(func(u + k2xyz/2))[i] for i in range(l)])
    k4xyz = np.array([dt*(func(u + k3xyz))[i] for i in range(l)])
    K = np.matrix([k1xyz,k2xyz,k3xyz,k4xyz])

    w = np.matrix([1/6,2/6,2/6,1/6])
    u = np.matrix(u)
    u1 = u + w * K

    return u1.getA()[0]
