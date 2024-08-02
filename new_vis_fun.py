def visibility_function(t):
    sin_t = np.sin(10*t)
    return sin_t


t_samples = np.arange(0,1.01,0.01)
num_samples= len(t_samples)
a_values = np.array([visibility_function(t) for t in t_samples])
#print(num_samples)

def find_zeros ():
    t_samples = np.arange(0,1.01,0.01)
    num_samples= len(t_samples)
    a_values = np.array([visibility_function(t) for t in t_samples])
    zeros= []

    for i in range(num_samples-1):
        if abs(a_values [i])<1e-10:
            zeros.append(t_samples[i])
        elif abs(a_values[i+1])< 1e-10:
            zeros.append(t_samples[i+1])
        elif a_values[i] *a_values[i+1] <0:
            # when sign change detected, compute the zero using linear interpolation
            t1 = t_samples [i]
            t2 = t_samples [i+1]
            a1 = a_values[i]
            a2 = a_values [i+1]
            # Linear interpolation to find the zero 
            zero = t1 -a1*(t2-t1)/ (a2-a1)
            #zero= np.interp(0, [a1, a2], [t1, t2])
            zeros.append(zero)
            


    return zeros


zeros = find_zeros()
