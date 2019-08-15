from pygef import ParseGEF
import os
import scipy
import matplotlib.pyplot as plt

basedir = 'data/cpt-grid/'
for cpt_path in os.listdir(basedir):
    cpt = ParseGEF(basedir + cpt_path)
    cpt.x
    cpt.y

    print(cpt)