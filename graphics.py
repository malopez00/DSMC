# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:42:28 2018

@author: malopez
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import theoretical_a2


results = np.array(results)
#np.savetxt('results.dat', results)

d = 3

x_theor = np.linspace(0,1, num=1000)
y_theor = theoretical_a2(x_theor, d)

#plt.errorbar(results[:,0], results[:,1], yerr=results[:,2], fmt='o', color='black',
#             ecolor='lightgray', elinewidth=3, capsize=0)


fig = plt.figure(figsize=(8, 6), dpi=300)
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel(r'$a_2$')
ax2 = ax1

ax1.axhline(y=0, color='black', linestyle='dashed')
ax1.errorbar(results[:,0], results[:,1], yerr=results[:,2], fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
ax2.plot(x_theor, y_theor, color='black')
