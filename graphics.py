# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:42:28 2018

@author: malopez
"""

import numpy as np
import matplotlib.pyplot as plt

np.savetxt('results.dat', results)

plt.errorbar(results[:,0], results[:,0], yerr=results[:,2], fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)