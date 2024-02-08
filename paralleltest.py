
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def func(i,mult):
	return [mult*i, i/mult]

mult = 2
arr = np.arange(0,10,1)
pool = mp.Pool(mp.cpu_count())
x = partial(func,mult=mult) # keeps arg "mult" as a constant in parallel loop over i
print(x)
res = np.array(pool.map_async(x,arr).get(99999))
m1,m2=np.split(res,2,axis=1)
print(m1,m2)
pool.close()
plt.show()
