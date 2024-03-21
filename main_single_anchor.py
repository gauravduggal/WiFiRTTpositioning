import numpy as np
import os
import pandas as pd
import glob
import re
from datetime import datetime
from rttData import rttData
import matplotlib.pyplot as plt

data_path =os.path.join('.\\','data','RTT_measurements','Durham_4_single_anchor','rtt*')
# reading csv file
print(data_path)
clabels = ['Date','Time','Truth','Distance','St Dev','Successes','Attempts','RSSI(range)','RTT(range)','RSSI(scan)','RTT(scan)','Frequency','CenterFreq0','CenterFreq1','BW','Standard','BSSID','SSID']
files = glob.glob(data_path)
#mac ID
AP1 = ["9c:4f:5f:4c:3a:45"]
#max range
Rmax = 25

# #generate ground truth measurement labels
# file_date_dict = {}
# for file in files:
#     head,tail = os.path.split(file)
#     # print(tail)
#     date_str = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', tail)
#     # print(date_str.group())
#     file_time = datetime.strptime(date_str.group(), '%Y-%m-%d-%H-%M-%S').date()
#     # print(file_time)
#     file_date_dict[file] = file_time
#
# sorted_files = sorted(file_date_dict.values())
# print(sorted_files)
#
#
true_range = np.linspace(start=2,stop=24,num=12,endpoint=True)
range_measurements = []
RSSI_measurements = []
std_dev = []

for file in sorted(files):
    df = pd.read_csv(file, sep=',', skipfooter=1, engine='python')
    # print(f'{df.columns=},columns:{len(df.columns)},rows:{len(df)}')
    AP1 = rttData(df)
    # print(f'MAC: {AP1.mac}, SSID: {AP1.SSID},BSSID: {AP1.BSSID} BW: {AP1.BW}, freq: {AP1.fc}MHz')
    # print(AP1.range)
    range_measurements.append(np.mean(AP1.range))
    RSSI_measurements.append(AP1.medianRSSI)
    std_dev.append(AP1.stddev)
    # print(AP1.medianRSSI)
    # counts, bins = np.histogram(AP1.range, bins=100, range=(0,Rmax))
    # plt.hist(bins[:-1], bins, weights=counts)
    # # plt.plot()
    # plt.ylabel("pdf")
    # plt.xlabel("Range (m)")
    # plt.xlim((0, Rmax))
    # plt.show()

range_measurements = np.array(range_measurements)

fig1 = plt.figure()
plt.plot(true_range,range_measurements,"ro",label='Measurements')
plt.plot(true_range,true_range,'b-',label='Ideal')
plt.grid(visible=True)
plt.xticks(true_range)
plt.yticks(true_range)
plt.xlim((0,Rmax))
plt.ylim((0,Rmax))
plt.xlabel("True Range (m)")
plt.ylabel("Measured Range (m)")
plt.title("Range Calibration")
plt.legend()
plt.show()
fig1.savefig('Range_plot.png')


fig2=plt.figure()
plt.plot(true_range,RSSI_measurements,"ro",label='RSSI measurements')
plt.plot(true_range,true_range,'b-')
plt.grid(visible=True)
plt.xticks(true_range)
plt.xlim((0,Rmax))
plt.ylim((-50,-90))
plt.xlabel("True Range (m)")
plt.ylabel("Measured RSSI (dBm)")
plt.title("RSSI vs range")
plt.legend()
plt.show()
fig2.savefig('RSSI_plot.png')


fig3=plt.figure()
plt.plot(true_range,std_dev,"ro",label='std dev (range)')
plt.grid(visible=True)
plt.xticks(true_range)
plt.yticks(np.arange(0,1,0.1))
plt.xlim((0,Rmax))
plt.ylim((0,1))
plt.ylabel("Std Dev (m)")
plt.xlabel("True Range (m)")
plt.title("Std Dev of range measurements")
plt.legend()
plt.show()
fig3.savefig('range_stddev.png')