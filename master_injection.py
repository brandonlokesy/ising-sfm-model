import numpy as np
from scipy.integrate import solve_ivp
import itertools
import multiprocessing
import os

def format_scientific(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def dwavelength(df, wavelength0):
    c=3e8
    f0= c/wavelength0
    return (c/f0) - (c/(f0-df))

def dfrequency(dwavelength, wavelength0):
    c=3e8
    f0=c/wavelength0
    return f0 - ((1/f0) - (dwavelength/c))**(-1)

def solve(paramList):
    dF, injP, run = paramList
    def dSdt(t, var, params):
        Ex, Ey, phix, phiy, N, m = var
        
        kappa, yp, ya, ys, y, alpha, Kinj, bsp, eta, dw, Einjx, Einjy, xix, xiy = params
        
        wx = alpha*ya - yp
        wy = yp- alpha*ya

        deltaphi = 2*wy*t + phiy - phix
        deltax = wy*t - phix
        deltay = wx*t - phiy
        
        f= np.array([
            kappa * ( (N-1)*Ex - m*Ey * (np.sin(deltaphi) + alpha*np.cos(deltaphi))) \
                - ya*Ex + Kinj*Einjx(t)*np.cos(deltax) + np.sqrt(bsp*(N+m))*xix(t),
            kappa * ( (N-1)*Ey + m*Ex * (alpha*np.cos(deltaphi) - np.sin(deltaphi))) \
                + ya*Ey + Kinj*Einjy(t)*np.cos(deltay) + np.sqrt(bsp*(N-m))*xix(t),
            kappa * ( alpha*(N-1) + m * (Ey/Ex) * (np.cos(deltaphi) \
                - alpha*np.sin(deltaphi))) - dw(t) - alpha*ya \
                    + Kinj * (Einjx(t)/Ex) * np.sin(deltax),
            kappa * ( alpha*(N-1) - m * (Ex/Ey) * (alpha*np.sin(deltaphi) \
                + np.cos(deltaphi))) - dw(t) + alpha*ya \
                    + Kinj * (Einjy(t)/Ey) * np.sin(deltay),
            -y*( N*(1+Ex**2 + Ey**2) - eta - 2*m* Ey*Ex * np.sin(deltaphi)),
            -ys*m - y*(m*(Ex**2 + Ey**2)) + 2*y*N*Ey*Ex*np.sin(deltaphi)
        ])
        return f
    
    # kappa, yp, ya, ys, y, alpha, Kinj, bsp, eta, dw, Einjx, Einjy, xix, xiy
    kappa = 125e9
    yp = 192e9
    ya = 0.02e9
    ys = 1000e9
    y = 0.67e9
    
    alpha = 3
    Kinj = 35.5e9
    bsp = 1e-5
    eta = 3.4

    # Solving Parameters
    abserr = 1.0e-9
    relerr = 1.0e-7
    del_t = 1e-12
    t = np.arange(0,40e-9,del_t)

    # Master Laser Parameters
    injectionTime = 20e-9
    injectedPower = injP
    dwValue = dF * 2 * np.pi

    def Einjx_func(t):
        if t>injectionTime:
            return injectedPower
        return 0
    def Einjy_func(t):
        if t>injectionTime:
            return injectedPower
        return 0
    def noise(t):
        return np.random.normal(0,1)
    def inj_w(t):
        if t>injectionTime:
            return dwValue
        return 0
    
    dw = inj_w
    Einjx = Einjx_func
    Einjy = Einjy_func
    xix = noise
    xiy = noise

    # # Initial conditions
    Ex, Ey = 0.001, 0.001
    phix, phiy = 0,0
    N = 0
    m = 0

    params = np.array([kappa, yp, ya, ys, y, alpha, Kinj, \
                       bsp, eta, dw, Einjx, Einjy, xix, xiy])
    var = np.array([Ex, Ey, phix, phiy, N, m])
    EValueStr = '{0:.2f}'.format(injectedPower)
    freqDetuningStr = dF/1e9
    filename = f'diagonal_{EValueStr}_{freqDetuningStr}GHz_run{int(run)}'
    wsol = solve_ivp(dSdt, t_span=(0, np.max(t)), y0=var, method = "RK45", \
        args = (params,), atol = abserr, rtol = relerr, t_eval = t)
    data = np.vstack((wsol.t.T, wsol.y))
    np.savetxt(f'./Data/{EValueStr}/{filename}.dat', data.T)
    return True

if __name__ == '__main__':
    injectedPowers = np.arange(0.1, 3.0, 0.1)
    detuningStep = np.abs(dfrequency(0.01e-9, 1550e-9))
    dwavelength_range=[0.5e-9, -0.5e-9]
    frequencyRange =list(map(lambda x: dfrequency(x, 1550e-9), dwavelength_range))
    frequencyDetunings = np.arange(frequencyRange[0], frequencyRange[1], detuningStep)
    repeats = 10
    repeat_list = range(repeats)
    paramList = list(itertools.product(frequencyDetunings, injectedPowers, repeat_list))
    pool = multiprocessing.Pool(os.cpu_count())
    pool.map(solve, paramList)