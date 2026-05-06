# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:23:36 2026

@author: Acer
"""
import numpy as np
from PlatePlant1 import PlatePlant
from rk4 import integrate
import matplotlib.pyplot as plt

plant = PlatePlant()

rng = np.random.default_rng(0)
x, _ = plant.reset(rng)

dt = 0.001
t = 0.0

u = np.array([50.00, 1.0])   # ورودی نمونه

trajectory = [x.copy()]

for k in range(10000):
    x = integrate(plant.dynamics, t, x, u, dt, n_steps=1)
    t += dt
    trajectory.append(x.copy())

trajectory = np.array(trajectory)




# زمان متناظر
steps = np.arange(len(trajectory))
time = steps * dt

# فرض کنیم K = تعداد مودها
K = plant.K

# فقط جابجایی مودها را نمایش بده (نیمه‌ی اول بردار حالت)
plt.figure(figsize=(8,5))
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,6))
ax[0].plot(time, trajectory[:, 0], 'b')
ax[0].set_ylabel("q₁ (x)")

ax[1].plot(time, trajectory[:, 0], 'r')
ax[1].set_ylabel("dq₁/dt (dx/dt)")
ax[1].set_xlabel("زمان (s)")
plt.show()
