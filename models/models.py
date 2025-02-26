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

def form_factor_sphere(q, R):
    F = 3*(np.sin(q*R)-q*R*np.cos(q*R))/np.power(q*R, 3)
    return F*F
    
def sphere(q, R, scale = 1, delta_rho = 1, background = 0.001):
    V = (4/3)*np.pi*R**3
    P = form_factor_sphere(q, R)
    return (scale/V)*(delta_rho**2)*(V**2)*P + background

def form_factor_core_shell_sphere(q, R1, R2):
    V1 = (4/3)*np.pi*R1**3
    V2 = (4/3)*np.pi*R2**3
    F1 = 3*(np.sin(q*R1)-q*R1*np.cos(q*R1))/np.power(q*R1, 3)
    F2 = 3*(np.sin(q*R2)-q*R2*np.cos(q*R2))/np.power(q*R2, 3)
    F  = (V1*F1-V2*F2)/(V1-V2)
    return F*F
    
def core_shell_sphere(q, R1, R2, scale = 1, delta_rho = 1, background = 0.001):
    V1 = (4/3)*np.pi*R1**3
    P = form_factor_core_shell_sphere(q, R1, R2)
    return  (scale/V1)*(delta_rho**2)*(V1**2)*P + background

def form_factor_cilinder(q, R, L):

    def integrand(alpha, q):
        A1 = q*L*np.cos(alpha)
        A2 = q*R*np.sin(alpha)
        T1 = 2*np.sin(A1/2)/(A1/2)
        T2 = j1(A2)/A2
        return np.sin(alpha)*(T1*T2)**2
    
    P, _ = quad_vec(lambda alpha: integrand(alpha, q), 0, np.pi/2)
    return P
    
def cilinder(q, R, L, scale = 1, delta_rho = 1, background = 0.001):
    V = np.pi*L*R**2
    W = (scale/V)*(delta_rho**2)*(V**2)
    P = form_factor_cilinder(q, R, L)
    return W*P + background

def form_factor_ellipsoid(q, Re, Rp, delta_rho):
    V = (4/3)*np.pi*Re*Rp**3
    
    def integrand(alpha, q):
        R = np.sqrt((np.sin(alpha)*Re)**2+(np.cos(alpha)*Rp)**2)
        T1 = (np.sin(q*R)-q*R*np.cos(q*R))/(q*R)**3
        T2 = delta_rho*V
        return np.sin(alpha)*(3*T1*T2)**2
    
    P, _ = quad_vec(lambda alpha: integrand(alpha, q), 0, np.pi/2)
    
    return P
    
def ellipsoid(q, Re, Rp, scale = 1, delta_rho = 1, background = 0.001):
    V = (4/3)*np.pi*Re*Rp**3
    W = (scale/V)
    P = form_factor_ellipsoid(q, Re, Rp, delta_rho)
    return W*P + background

def form_factor_parallelepiped(q, a, b, c):
    
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
    
    return quad_vec(outer_integral, 0, np.pi/2)[0]
    
def parallelepiped(q, a, b, c, scale = 1, delta_rho = 1, background = 0.001):
    V = a*b*c
    W = (scale/V)*(2/np.pi)*(delta_rho*V)**2    
    P = form_factor_parallelepiped(q, a, b, c)
    return W*P+background

def form_factor_rod(q, L):
    si, _ = sici(q*L)
    return 2*si/(q*L) - 4*np.power(np.sin(q*L/2),2)/np.power(q*L, 2)

def rod(q, L, scale = 1, delta_rho = 1, background = 0.001):
    V = L
    P = form_factor_rod(q, L)
    return P*(scale/V)*(delta_rho*V)**2 + background
    
if __name__ == "__main__":
    
    q = np.linspace(0.001, 2, 1000)
    
    R = 20
    R1 = 50
    R2 = 20
    L = 100
    Re = 120
    Rp = 40
    a = 50
    b = 75
    c = 150
    Lr = 100
    
    fig, axes = plt.subplots(nrows= 2, ncols= 3, constrained_layout=True)
    
    fig.suptitle('Scattering intensity for different form factors', fontweight ="bold")
    
    axes[0, 0].plot(q, sphere(q, R))
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlim([0.001, 2])
    axes[0, 0].set_ylabel('I(q)')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_title('Sphere')
    
    axes[0, 1].plot(q, core_shell_sphere(q, R1, R2))
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlim([0.001, 2])
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    axes[0, 1].set_title('Core shell sphere')
    
    axes[0, 2].plot(q, cilinder(q, R, L))
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_xlim([0.001, 2])
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    axes[0, 2].set_title('Cilinder')
    
    axes[1, 0].plot(q, ellipsoid(q, Re, Rp))
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_ylabel('I(q)')
    axes[1, 0].set_xlabel('q(\u00C5\u207B\u00B9)')
    axes[1, 0].set_xlim([0.001, 2])
    axes[1, 0].set_title('Ellipsoid')
    
    axes[1, 1].plot(q, parallelepiped(q, a, b, c))
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlim([0.001, 2])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlabel('q(\u00C5\u207B\u00B9)')
    axes[1, 1].set_title('Parallelepiped')
    
    axes[1, 2].plot(q, rod(q, L))
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_xscale('log')
    axes[1, 2].set_xlim([0.001, 2])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title('Rod')
    axes[1, 2].set_xlabel('q(\u00C5\u207B\u00B9)')

    plt.show()