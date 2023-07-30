import numpy as np
import sdeint, multiprocessing, os

""" IMPORTANT
Change the "bits" variable to reflect system bit
Change "A" variable for non-HWP interaction
Change "AHWP" variable for HWP interaction

Change whether mutual VCSEL interaction occurs with the "mutual" boolean
Change whether a master laser sent to all VCSELs is used with the "master" boolean
"""

def dwavelength(df, wavelength0): 
    c=3e8
    f0= c/wavelength0
    return (c/f0) - (c/(f0-df))

def dfrequency(dwavelength, wavelength0):
    c=3e8
    f0=c/wavelength0
    return f0 - ((1/f0) - (dwavelength/c))**(-1)

def solve_parallel(run):
    def solve(filename, t):
        print("### Start ###")
        wsol = sdeint.itoint(dSdt, dGdt, y0=var.flatten(), tspan = t)
        data = np.vstack((t, wsol.T))
        np.savetxt(f'./Data/{filename}.dat', data.T)
        print("### END ###")
        return data
    def dSdt(var, t):
        Ex = var[:bits] #Split "var" in to each category
        Ey = var[bits:2*bits]
        phix = var[2*bits:3*bits]
        phiy = var[3*bits:4*bits]
        N = var[4*bits:5*bits]
        m = var[5*bits:6*bits]

        ExNew = np.zeros(bits)
        EyNew = np.zeros(bits)
        phixNew = np.zeros(bits)
        phiyNew = np.zeros(bits)
        NNew = np.zeros(bits)
        mNew = np.zeros(bits)

        kappa, yp, ya, ys, y, alpha, Kinj, bsp, eta, \
            dw, Einjx, Einjy, delta, xi, AMatrix, AMatrixHWP, deltaCouplingMatrix = params
        wx = alpha*ya - yp
        wy = yp- alpha*ya

        sigmajix_matrix = np.zeros(shape = (bits,bits))
        sigmajiy_matrix = np.zeros(shape = (bits,bits))
        sigmajixtoy_matrix = np.zeros(shape = (bits,bits))
        sigmajiytox_matrix = np.zeros(shape = (bits,bits))
        for j in range(bits):
            for i in range(bits):
                sigmajix_matrix[j][i] = wx + wy + phix[j] - phix[i] \
                    + deltaCouplingMatrix[j][i]
                sigmajiy_matrix[j][i] = wx + wy + phiy[j] - phiy[i]
                sigmajiytox_matrix[j][i] = wy - wx + phiy[j] - phix[i]
                sigmajixtoy_matrix[j][i] = wx - wy + phix[j] - phiy[i]
            sigmajix_matrix[j][j] = 0 #Diagonals are zero
            sigmajiy_matrix[j][j] = 0
            sigmajixtoy_matrix[j][j] = 0
            sigmajiytox_matrix[j][j] = 0

        Ex_mutual_injection_lock, Ey_mutual_injection_lock,\
            phix_mutual_injection_lock, phiy_mutual_injection_lock= 0,0,0,0

        '''E-fields are in the form (Kinj x I ) x A.cos(sigma) x Ex
        . is the Hadmard product
        x is matrix multiplication'''
        Ex_mutual_injection_lock = Kinj * np.matmul(np.multiply(AMatrix(t), np.cos(sigmajix_matrix).T), Ex) + Kinj * np.matmul(np.multiply(AMatrixHWP(t), np.cos(sigmajiytox_matrix).T), Ey)
        Ey_mutual_injection_lock = Kinj * np.matmul(np.multiply(AMatrix(t), np.cos(sigmajiy_matrix).T), Ey) + Kinj * np.matmul(np.multiply(AMatrixHWP(t), np.cos(sigmajixtoy_matrix).T), Ex)
        phix_mutual_injection_lock = Kinj * np.matmul(np.multiply(AMatrix(t), np.sin(sigmajix_matrix).T), Ex) + Kinj * np.matmul(np.multiply(AMatrixHWP(t), np.sin(sigmajiytox_matrix).T), Ey)
        phiy_mutual_injection_lock = Kinj * np.matmul(np.multiply(AMatrix(t), np.sin(sigmajiy_matrix).T), Ey) + Kinj * np.matmul(np.multiply(AMatrixHWP(t), np.sin(sigmajixtoy_matrix).T), Ex)

        for n in range(bits): #For each bit/Ising state, perform the computation
            deltax_n = wy*t - phix[n] + delta #bit-specific phase information
            deltay_n = wx*t - phiy[n]
            deltaphi_n = 2*wy*t + phiy[n] - phix[n]
            ExNew[n] = kappa * ( (N[n]-1)*Ex[n] - m[n]*Ey[n] * (np.sin(deltaphi_n) + alpha*np.cos(deltaphi_n))) - ya*Ex[n] + Kinj*Einjx(t)*np.cos(deltax_n) 
            EyNew[n] = kappa * ( (N[n]-1)*Ey[n] + m[n]*Ex[n] * (alpha * np.cos(deltaphi_n) - np.sin(deltaphi_n))) + ya*Ey[n] + Kinj*Einjy(t)*np.cos(deltay_n)
            phixNew[n] = kappa * ( alpha*(N[n]-1) + m[n] * (Ey[n]/Ex[n]) * (np.cos(deltaphi_n) - alpha*np.sin(deltaphi_n))) - dw(t) - alpha*ya + Kinj * (Einjx(t)/Ex[n]) * np.sin(deltax_n)
            phiyNew[n] = kappa * ( alpha*(N[n]-1) - m[n] * (Ex[n]/Ey[n]) * (alpha* np.sin(deltaphi_n) + np.cos(deltaphi_n))) - dw(t) + alpha*ya + Kinj * (Einjy(t)/Ey[n]) * np.sin(deltay_n)
            NNew[n] = -y*( N[n]*(1+ np.power(Ex[n],2) + np.power(Ey[n],2)) - eta - 2*m[n]* Ey[n]* Ex[n] * np.sin(deltaphi_n))
            mNew[n] = -ys*m[n] - y*(m[n]*(np.power(Ex[n],2) + np.power(Ey[n],2))) + 2*y*N[n]*Ey[n]*Ex[n]*np.sin(deltaphi_n)
            
        ExNew = ExNew + Ex_mutual_injection_lock # ! Adding mutual VCSEL coupling terms
        EyNew = EyNew + Ey_mutual_injection_lock
        phixNew = phixNew + (phix_mutual_injection_lock * (1/Ex))
        phiyNew = phiyNew + (phiy_mutual_injection_lock * (1/Ey))
        f = np.concatenate((ExNew, EyNew, phixNew, phiyNew, NNew, mNew))
        return f
    def dGdt(var, t):
        kappa, yp, ya, ys, y, alpha, Kinj, bsp, eta, dw, Einjx, \
            Einjy, delta, xi, AMatrix, AMatrixHWP, deltaCouplingMatrix = params
        Ex = var[:bits] #Split "var" in to each category
        Ey = var[bits:2*bits]
        phix = var[2*bits:3*bits]
        phiy = var[3*bits:4*bits]
        N = var[4*bits:5*bits]
        m = var[5*bits:6*bits]

        noise_term_x = np.sqrt(bsp*(N+m))
        noise_term_y = np.sqrt(bsp*(N-m))
        
        combined_terms = np.hstack((noise_term_x.flatten(), noise_term_y.flatten()))
        
        g = np.zeros(shape=(bits*6,bits*6))
        for i in range(len(combined_terms)):
            g[i][i] = combined_terms[i]
        return g

    master = True
    mutual = True
    noise = True
    bits = 3
    injectionTime = 10e-9
    couplingTime = 15e-9

    kappa = 125e9
    yp = 192e9
    ya = 0.02e9
    ys = 1000e9
    y = 0.67e9
    alpha = 3
    Kinj = 35.5e9
    bsp = 1e-5
    eta = 3.4

    # * Solving Parameters
    del_t = 1e-12
    t = np.arange(0,40e-9,del_t)

    # * Initial Values
    EConstant = 0.001
    Ex = EConstant * np.ones(bits)
    Ey = EConstant * np.ones(bits)
    phix = np.zeros(bits)
    phiy = np.zeros(bits)
    N = np.zeros(bits)
    m = np.zeros(bits)

    def inj_w(t):
        if t>injectionTime and master:
            return 1.956450e+11
        return 0
    
    dw = inj_w #frequency detuning
    delta = 0 #phase shift of injected signal

    def Einjx_func(t):
        if t>injectionTime:
            return 0.2
        return 0
    def Einjy_func(t):
        if t>injectionTime:
            return 0.2
        return 0
    Einjx = Einjx_func
    Einjy = Einjy_func
    if not master:
        Einjx = lambda x: 0
        Einjy = lambda x: 0
        dw = lambda x : 0
    def gaussian_noise(t):
        return np.random.normal(0,1)
    xi = gaussian_noise
    if not noise:
        xi = lambda x:0
    # ! Coupling Parameters
    deltaCouplingMatrix = np.zeros(shape=(bits, bits))
    A = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])
    def coupling(t): #represents the attenuation between VCSEL coupling
        if t <= couplingTime:
            return np.zeros(shape=(bits,bits))
        if t> couplingTime:
            return A
    AHWP = np.array([
        [0,0,0],
        [0,0,1],
        [0,1,0]
    ])
    def HWPcoupling(t):
        if t<= couplingTime:
            return np.zeros(shape = (bits, bits))
        if t> couplingTime:
            return AHWP
        
    AMatrix = coupling
    AMatrixHWP = HWPcoupling
    if not mutual:
        AMatrix = lambda x: 0
        AMatrixHWP = lambda x:0

    params = kappa, yp, ya, ys, y, alpha, Kinj, bsp, eta, \
        dw, Einjx, Einjy, delta, xi, AMatrix, AMatrixHWP, deltaCouplingMatrix
    var = np.array([Ex, Ey, phix, phiy, N, m])
    filename = f'3bit_graph_{run}'
    wsol = solve(filename = filename, t=t)
    return True
if __name__ == "__main__":
    repeats = 5000
    pool = multiprocessing.Pool(os.cpu_count())
    pool.map(solve_parallel, range(repeats))

