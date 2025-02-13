# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:08:25 2025

@author: gustavo
"""
import numpy as np
from scipy.special import j1  
from scipy.integrate import quad_vec

import matplotlib.pyplot as plt

def sphere(q, R, scale = 1, delta_rho = 1, background = 0.001):
    V = (4/3)*np.pi*R**3
    F = 3*(np.sin(q*R)-q*R*np.cos(q*R))/np.power(q*R, 3)
    return (scale/V)*(delta_rho**2)*(V**2)*(F*F) + background

def core_shell_sphere(q, R1, R2, scale = 10, delta_rho = 1, background = 0.001):
    V1 = (4/3)*np.pi*R1**3
    V2 = (4/3)*np.pi*R2**3
    F1 = 3*(np.sin(q*R1)-q*R1*np.cos(q*R1))/np.power(q*R1, 3)
    F2 = 3*(np.sin(q*R2)-q*R2*np.cos(q*R2))/np.power(q*R2, 3)
    F  = (V1*F1-V2*F2)/(V1-V2)
    return  (scale/V1)*(delta_rho**2)*(V1**2)*(F*F) + background

def cilynder(q, R, L, scale = 10, delta_rho = 1, background = 0.001):
    V = np.pi*L*R**2
    W = (scale/V)*(delta_rho**2)*(V**2)
    
    def integrand(alpha, q):
        A1 = q*L*np.cos(alpha)
        A2 = q*R*np.sin(alpha)
        T1 = 2*np.sin(A1/2)/(A1/2)
        T2 = j1(A2)/A2
        return np.sin(alpha)*(T1*T2)**2
    
    P, _ = quad_vec(lambda alpha: integrand(alpha, q), 0, np.pi/2)
    return W*P + background

def ellipsoid(q, Re,Rp, scale = 1, delta_rho = 1, background = 0.001):
    V = (4/3)*np.pi*Re*Rp**3
    W = (scale/V)
    
    def integrand(alpha, q):
        R = np.sqrt((np.sin(alpha)*Re)**2+(np.cos(alpha)*Rp)**2)
        T1 = (np.sin(q*R)-q*R*np.cos(q*R))/(q*R)**3
        T2 = delta_rho*V
        return np.sin(alpha)*(3*T1*T2)**2
    
    P, _ = quad_vec(lambda alpha: integrand(alpha, q), 0, np.pi/2)
    
    return W*P + background

def parallelepiped(q, a, b, c, scale = 1, delta_rho = 1, background = 0.001):
    V = a*b*c
    W = (scale/V)*(2/np.pi)*(delta_rho*V)**2

    def integrand(alpha, beta, q, a, b, c):
        A1 = q*a*np.sin(alpha)*np.cos(beta)
        A2 = q*b*np.sin(alpha)*np.cos(beta)
        A3 = q*c*np.cos(alpha)
        
        T1 = np.sin(A1)/A1
        T2 = np.sin(A2)/A2
        T3 = np.sin(A3)/A3
        
        return T1*T2*T3*np.sin(alpha)
    
    def outer_integral(alpha):
        return quad_vec(lambda beta: integrand(alpha, beta, q, a, b, c), 0, np.pi/2)[0]
    
    P = quad_vec(outer_integral, 0, np.pi/2)[0]
    
    return W*P+background
        
if __name__ == "__main__":
    
    q = np.linspace(0.001, 1, 1000)
    R = 20
    R1 = 50
    R2 = 20
    L = 400
    Re = 400
    Rp = 20
    a = 35
    b = 75
    c = 400
    #plt.plot(q, sphere(q, R))
    #plt.plot(q, core_shell_sphere(q, R1, R2))
    #plt.plot(q, cilynder(q, R, L))
    #plt.plot(q, ellipsoid(q, Re, Rp))
    plt.plot(q, parallelepiped(q, a, b, c))
    plt.xlabel('q')
    plt.ylabel('I(q)')
    plt.title('Intesity of Parallelepiped')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()