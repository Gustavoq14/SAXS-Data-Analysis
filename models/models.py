# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:08:25 2025

@author: gustavo
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j1  
from scipy.special import sici
from scipy.integrate import quad_vec
from numba import njit

@njit
def form_factor_sphere(q, R):
    qR = q*R
    F = 3*(np.sin(qR) - qR*np.cos(qR))/np.power(qR, 3)
    return F*F

@njit
def sphere(q, R, scale = 1, delta_rho = 1, background = 0.001):
    V = (4/3)*np.pi*R**3
    W = (scale/V)*(delta_rho*V)**2
    P = form_factor_sphere(q, R)
    return W*P + background

@njit
def form_factor_core_shell_sphere(q, R1, R2):
    V1 = (4/3)*np.pi*R1**3
    V2 = (4/3)*np.pi*R2**3
    qR1 = q*R1
    qR2 = q*R2
    F1 = 3*(np.sin(qR1)-qR1*np.cos(qR1))/np.power(qR1, 3)
    F2 = 3*(np.sin(qR2)-qR2*np.cos(qR2))/np.power(qR2, 3)
    F  = (V1*F1-V2*F2)/(V1-V2)
    return F*F

@njit
def core_shell_sphere(q, R1, R2, scale = 1, delta_rho = 1, background = 0.001):
    V1 = (4/3)*np.pi*R1**3
    W = (scale/V1)*(delta_rho*V1)**2
    P = form_factor_core_shell_sphere(q, R1, R2)
    return  W*P + background

def form_factor_cylinder(q, R, L, alpha = None):

    def integrand(alpha, q, R, L):
        A1 = q*L*np.cos(alpha)/2
        A2 = q*R*np.sin(alpha)
        T1 = 2*np.sin(A1)/(A1+1e-10)
        T2 = j1(A2)/(A2+ 1e-10)
        return np.sin(alpha)*(T1*T2)**2
    
    if alpha is None:
        P, _ = quad_vec(lambda alpha: integrand(alpha, q, R, L), 0, np.pi/2, epsabs=1e-6, epsrel=1e-6)
    else:
        P = integrand(alpha, q, R, L)
    return P

def cylinder(q, R, L, scale = 1, delta_rho = 1, background = 0.001):
    V = np.pi*L*R**2
    W = (scale/V)*(delta_rho*V)**2
    P = form_factor_cylinder(q, R, L)
    return W*P + background

def form_factor_ellipsoid(q, Re, Rp, alpha = None):
    
    def integrand(alpha, q, Re, Rp):
        R = np.sqrt((np.sin(alpha)*Re)**2+(np.cos(alpha)*Rp)**2)
        qR = q*R
        F = 3*(np.sin(qR)-qR*np.cos(qR))/np.power(qR, 3)
        return np.sin(alpha)*(F)**2
    
    if alpha is None:
        P, _ = quad_vec(lambda alpha: integrand(alpha, q, Re, Rp), 0, np.pi/2, epsabs=1e-6, epsrel=1e-6)
    else:
        P = integrand(alpha, q, Re, Rp)
    return P
    
def ellipsoid(q, Re, Rp, scale = 1, delta_rho = 1, background = 0.001):
    V = (4/3)*np.pi*Re*Rp**2
    W = (scale/V)*(delta_rho*V)**2
    P = form_factor_ellipsoid(q, Re, Rp)
    return W*P + background

def form_factor_parallelepiped(q, a, b, c, alpha = None, beta = None):
    
    def integrand(alpha, beta, q, a, b, c):
        A1 = q*a*np.sin(alpha)*np.cos(beta)
        A2 = q*b*np.sin(alpha)*np.sin(beta)
        A3 = q*c*np.cos(alpha)
    
        T1 = np.sinc(A1 / np.pi)
        T2 = np.sinc(A2 / np.pi) 
        T3 = np.sinc(A3 / np.pi) 

        return np.power(T1*T2*T3, 2)*np.sin(alpha)
    
    if alpha is None and beta is None:
        def outer_integral(alpha, q, a, b, c):
            return quad_vec(lambda beta: integrand(alpha, beta, q, a, b, c), 0, np.pi / 2, epsabs=1e-6, epsrel=1e-6)[0]
     
        P, error = quad_vec(lambda alpha: outer_integral(alpha, q, a, b, c), 0, np.pi / 2, epsabs=1e-6, epsrel=1e-6) 
    else:
        P = integrand(alpha, beta, q, a, b, c)
    return P
    
def parallelepiped(q, a, b, c, scale = 1, delta_rho = 1, background = 0.001):
    V = a*b*c
    W = (scale/V)*(2/np.pi)*(delta_rho*V)**2    
    P = form_factor_parallelepiped(q, a, b, c)
    return W*P+background

def form_factor_rod(q, L):
    si, _ = sici(q*L)
    qL = q*L
    P = 2*si/(qL) - 4*np.power(np.sin(qL/2),2)/np.power(qL, 2)
    return P

def rod(q, L, scale = 1, delta_rho = 1, background = 0.001):
    V = L
    W = (scale/V)*(delta_rho*V)**2
    P = form_factor_rod(q, L)
    return W*P + background
    
if __name__ == "__main__":
    
    q = np.linspace(0.001, 1, 1000)
    
    R = 50
    R1 = 60
    R2 = 10
    Rc = 20
    L = 400
    Re = 120
    Rp = 40
    a = 35
    b = 75
    c = 400
    Lr = 100   

    fig, axes = plt.subplots(nrows= 2, ncols= 3, constrained_layout=True)
    
    fig.suptitle('Scattering intensity for different form factors', fontweight ="bold")
    
    axes[0, 0].plot(q, sphere(q, R))
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlim([0.001, 1])
    axes[0, 0].set_ylabel('I(q)')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_title('Sphere')
    
    axes[0, 1].plot(q, core_shell_sphere(q, R1, R2))
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlim([0.001, 1])
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    axes[0, 1].set_title('Core shell sphere')
    
    axes[0, 2].plot(q, cylinder(q, Rc, L))
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_xlim([0.001, 1])
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    axes[0, 2].set_title('Cylinder')
    
    axes[1, 0].plot(q, ellipsoid(q, Re, Rp))
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_ylabel('I(q)')
    axes[1, 0].set_xlabel('q(\u00C5\u207B\u00B9)')
    axes[1, 0].set_xlim([0.001, 1])
    axes[1, 0].set_title('Ellipsoid')
    
    axes[1, 1].plot(q, parallelepiped(q, a, b, c))
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlim([0.001, 1])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlabel('q(\u00C5\u207B\u00B9)')
    axes[1, 1].set_title('Parallelepiped')
    
    axes[1, 2].plot(q, rod(q, L))
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_xscale('log')
    axes[1, 2].set_xlim([0.001, 1])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title('Rod')
    axes[1, 2].set_xlabel('q(\u00C5\u207B\u00B9)')

    plt.show()
