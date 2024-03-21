import numpy as np
class rttData:
    def __init__(self, df):
        self.SSID = df['SSID'][20]
        self.BSSID = df['BSSID'][20]
        self.mac = df['BSSID'][20]
        self.BW = df['BW'][20]
        self.fc = df['Frequency'][20]
        self.range = self.get_range(df)
        self.medianRSSI = self.get_median_rssi(df)
        self.stddev = np.std(self.range)

    def get_range(self, df):
        rm = []
        for ridx in range(0, len(df)):
            if df['Successes'][ridx] > 1 : #and  df['St Dev'][ridx] < 0.5:
                rm.append(df['Distance'][ridx])
        return np.array(rm)

    def get_median_rssi(self, df):
        rm = []
        for ridx in range(0, len(df)):
            if df['Successes'][ridx] > 1 : #and  df['St Dev'][ridx] < 0.5:
                rm.append(df['RSSI(range)'][ridx])
        return np.array(np.median(rm))