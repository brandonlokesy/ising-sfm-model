import numpy as np
import pandas as pd
import os

def unpack_var(var, bits):
    t = var[0]
    left = var[1:]
    bits = int(left.shape[0]/6)
    Ex = left[:bits] #Split "var" in to each category
    Ey = left[bits:2*bits]
    phix = left[2*bits:3*bits]
    phiy = left[3*bits:4*bits]
    N = left[4*bits:5*bits]
    m = left[5*bits:6*bits]
    return t, Ex, Ey, phix, phiy, N, m

path = 'Data'
currentpath = os.getcwd()
newpath = os.path.join(currentpath, path)
fileList = [f for f in os.listdir(newpath) if os.path.isfile(os.path.join(newpath, f))]
idx = 35000 #hardcoded, I know the index where t=35ns is

dataframe_values = np.array([])

bits = 3
cols_avg = [f"VCSEL{i}_avg" for i in range(bits)]
cols_last = [f"VCSEL{i}_last" for i in range(bits)]
columns = ["run"] + cols_avg + cols_last

for f in fileList:
    filename = os.path.join(newpath, f)
    if not os.path.isfile(filename):
        continue
        
    run = f.split("_")[2]
    sol = np.loadtxt(f'{filename}', unpack=True)
    t, Ex, Ey, phix, phiy, N, m = unpack_var(sol, bits)
    Ex = np.abs(Ex)
    Ey = np.abs(Ey)
    bits = len(Ex)
    
    states_avg = []
    states_last = []
    for bit in range(bits):
        bitState_avg = 0
        bitState_last = 0
        
        ExTemp = Ex[bit]
        EyTemp = Ey[bit]
        
        Ex_avg = np.average(ExTemp[idx:])
        Ey_avg = np.average(EyTemp[idx:])
        
        ExLast = ExTemp[-1]
        EyLast = EyTemp[-1]
        
        if Ey_avg >= Ex_avg:
            bitState_avg = 1
        else:
            bitState_avg = -1
        
        if EyLast >= ExLast:
            bitState_last = 1
        else:
            bitState_last = -1
            
        states_avg += [bitState_avg]
        states_last += [bitState_last]
    row = [run] + states_avg + states_last
    dataframe_values=np.append(dataframe_values, row)
    
dataframe_values_reshaped=dataframe_values.reshape(int(dataframe_values.shape[0]/len(columns)), len(columns))
df = pd.DataFrame(dataframe_values_reshaped, columns = columns)
df.to_csv(f'{bits}bit_states.csv', encoding='utf-8')