
import numpy as np
import os
import pandas as pd
import glob
import re
from datetime import datetime
from rttData import rttData
import matplotlib.pyplot as plt

#anchor positions
AP0_pos = np.array([0,17.32])
AP1_pos = np.array([-10,0])
AP2_pos = np.array([10,0])
#ground truth
range_ground_truth = np.array([
        [15.26, 33.96, 25.30],
        [7.52,19.01,12.84],
        [13.67, 15.28, 7.06],
        [10.13, 12.32, 12.39],
        [5.39, 15, 16.45],
        [20.85, 7.12, 14],
        [12.42, 8.05, 15.44],
        ])

fig, ax = plt.subplots()
# ax.axis('equal')
ax.scatter(AP0_pos[0], AP0_pos[1], c='#00FF00', s=50, alpha=0.7,marker="s",edgecolors='#FF0000',label="Anchors")
ax.scatter(AP1_pos[0], AP1_pos[1], c='#00FF00', s=50, alpha=0.7,marker="s",edgecolors='#FF0000')
ax.scatter(AP2_pos[0], AP2_pos[1], c='#00FF00', s=50, alpha=0.7,marker="s",edgecolors='#FF0000')


def get_LLS(AP0_pos,AP1_pos,AP2_pos,r0,r1,r2):
        row0 = [2*AP0_pos[0]-2*AP1_pos[0],2*AP0_pos[1]-2*AP1_pos[1]]
        row1 = [2*AP0_pos[0]-2*AP2_pos[0],2*AP0_pos[1]-2*AP2_pos[1]]
        row2 = [2 * AP1_pos[0] - 2 * AP2_pos[0], 2 * AP1_pos[1] - 2 * AP2_pos[1]]

        A = np.matrix([row0,row1,row2])
        row0 = [r1**2-r0**2+AP0_pos[0]**2+AP0_pos[1]**2-AP1_pos[0]**2-AP1_pos[1]**2]
        row1 = [r2**2-r0**2+AP0_pos[0]**2+AP0_pos[1]**2-AP2_pos[0]**2-AP2_pos[1]**2]
        row2 = [r2 ** 2 - r1 ** 2 + AP1_pos[0] ** 2 + AP1_pos[1] ** 2 - AP2_pos[0] ** 2 - AP2_pos[1] ** 2]

        b = np.matrix([row0,row1,row2])
        res = np.linalg.inv(np.transpose(A)*A)*np.transpose(A)*b
        x = res.item(0)
        y = res.item(1)
        return x,y

for i in range(0,len(range_ground_truth)):
        # print(range_ground_truth.shape)
        r0 = range_ground_truth[i,0]
        r1 = range_ground_truth[i,1]
        r2 = range_ground_truth[i,2]
        [x, y] = get_LLS(AP0_pos, AP1_pos, AP2_pos, r0, r1, r2)
        print(f'{x=},{y=}')


data_path =os.path.join('.\\','data','RTT_measurements','2D_positioning','rtt*')
# reading csv file
print(data_path)
clabels = ['Date','Time','Truth','Distance','St Dev','Successes','Attempts','RSSI(range)','RTT(range)','RSSI(scan)','RTT(scan)','Frequency','CenterFreq0','CenterFreq1','BW','Standard','BSSID','SSID']
files = glob.glob(data_path)
print(files)
#mac ID
#AP1 = ["9c:4f:5f:4c:3a:45"]

#
# true_range = np.linspace(start=2,stop=24,num=12,endpoint=True)
# range_measurements = []
# RSSI_measurements = []
# std_dev = []
#
Nfiles = len(files)
files = sorted(files)

true_pos = np.zeros((7,2))
est_pos = np.zeros((7,2))

for fidx in range(0,Nfiles,3):
    df = pd.read_csv(files[fidx], sep=',', skipfooter=1, engine='python')
    # print(f'{df.columns=},columns:{len(df.columns)},rows:{len(df)}')
    AP0 = rttData(df)
    # print(f'MAC: {AP0.mac}, SSID: {AP0.SSID},BSSID: {AP0.BSSID} BW: {AP0.BW}, freq: {AP0.fc}MHz')
    # print(AP1.range)
    r0 = np.mean(AP0.range)+0.5

    df = pd.read_csv(files[fidx+1], sep=',', skipfooter=1, engine='python')
    # print(f'{df.columns=},columns:{len(df.columns)},rows:{len(df)}')
    AP1 = rttData(df)
    # print(f'MAC: {AP1.mac}, SSID: {AP1.SSID},BSSID: {AP1.BSSID} BW: {AP1.BW}, freq: {AP1.fc}MHz')
    # print(AP1.range)
    r1 = np.mean(AP1.range)+0.8

    df = pd.read_csv(files[fidx+2], sep=',', skipfooter=1, engine='python')
    # print(f'{df.columns=},columns:{len(df.columns)},rows:{len(df)}')
    AP2 = rttData(df)
    # print(f'MAC: {AP2.mac}, SSID: {AP2.SSID},BSSID: {AP2.BSSID} BW: {AP2.BW}, freq: {AP2.fc}MHz')
    # print(AP1.range)
    r2 = np.mean(AP2.range)+0.8
    # print(f'{r0=},{r1=},{r2=}')
    [x_est, y_est] = get_LLS(AP0_pos, AP1_pos, AP2_pos, r0, r1, r2)
    r0 = range_ground_truth[int(fidx/3),0]
    r1 = range_ground_truth[int(fidx/3),1]
    r2 = range_ground_truth[int(fidx/3),2]
    [x_true, y_true] = get_LLS(AP0_pos, AP1_pos, AP2_pos, r0, r1, r2)
    # print(f'{r0=},{r1=},{r2=}')
    print(f'{int(fidx/3)}. {x_est=},{x_true=},{y_est=},{y_true=}')
    est_pos[int(fidx / 3),0] = x_est
    est_pos[int(fidx / 3), 1] = y_est
    true_pos[int(fidx / 3),0] = x_true
    true_pos[int(fidx / 3), 1] = y_true

ax.scatter(true_pos[:,0], true_pos[:,1], c='#0000FF', s=50, alpha=1, marker='x',label="$True\;Position\;(node)$")
ax.scatter(est_pos[:,0], est_pos[:,1], c='#d62728', s=50, alpha=0.5, marker='o',label="$Estimated\;Position\;(node)$")

ax.set_xlabel(r'$X (m)$', fontsize=15)
ax.set_ylabel(r'$Y (m)$', fontsize=15)
ax.set_title('$2D\;Positioning\;Performance$')
# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(-16, 25, 2)
minor_ticks = np.arange(-6, 35, 2)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')
plt.xlim(left=-15, right=20)
plt.ylim(bottom=-5, top=30)
ax.grid(True)
ax.legend()
fig.tight_layout()
plt.gca().set_aspect('equal')
plt.show()

    # print(f'pause')
#     range_measurements.append(np.mean(AP1.range))
#     RSSI_measurements.append(AP1.medianRSSI)
#     std_dev.append(AP1.stddev)
#     # print(AP1.medianRSSI)
#     # counts, bins = np.histogram(AP1.range, bins=100, range=(0,Rmax))
#     # plt.hist(bins[:-1], bins, weights=counts)
#     # # plt.plot()
#     # plt.ylabel("pdf")
#     # plt.xlabel("Range (m)")
#     # plt.xlim((0, Rmax))
#     # plt.show()
#
# range_measurements = np.array(range_measurements)
#
# fig1 = plt.figure()
# plt.plot(true_range,range_measurements,"ro",label='Measurements')
# plt.plot(true_range,true_range,'b-',label='Ideal')
# plt.grid(visible=True)
# plt.xticks(true_range)
# plt.yticks(true_range)
# plt.xlim((0,Rmax))
# plt.ylim((0,Rmax))
# plt.xlabel("True Range (m)")
# plt.ylabel("Measured Range (m)")
# plt.title("Range Calibration")
# plt.legend()
# plt.show()
# fig1.savefig('Range_plot.png')
#
#
# fig2=plt.figure()
# plt.plot(true_range,RSSI_measurements,"ro",label='RSSI measurements')
# plt.plot(true_range,true_range,'b-')
# plt.grid(visible=True)
# plt.xticks(true_range)
# plt.xlim((0,Rmax))
# plt.ylim((-50,-90))
# plt.xlabel("True Range (m)")
# plt.ylabel("Measured RSSI (dBm)")
# plt.title("RSSI vs range")
# plt.legend()
# plt.show()
# fig2.savefig('RSSI_plot.png')
#
#
# fig3=plt.figure()
# plt.plot(true_range,std_dev,"ro",label='std dev (range)')
# plt.grid(visible=True)
# plt.xticks(true_range)
# plt.yticks(np.arange(0,1,0.1))
# plt.xlim((0,Rmax))
# plt.ylim((0,1))
# plt.ylabel("Std Dev (m)")
# plt.xlabel("True Range (m)")
# plt.title("Std Dev of range measurements")
# plt.legend()
# plt.show()
# fig3.savefig('range_stddev.png')