import numpy as np
from scipy import fftpack, signal, interpolate
import matplotlib.pyplot as plt


class Timeseries:
    def __init__(self, datapoints, timestamps=None, samplerate_Hz=1.0, starttime=0.0, units=None):
        self.values = datapoints
        self.units = units
        self.samplerate_Hz = samplerate_Hz
        if timestamps is not None and (len(datapoints) == len(timestamps)):
            self.times = timestamps
        else:
            self.times = np.arange(starttime, samplerate_Hz*len(datapoints), samplerate_Hz)
        self.interp = interpolate.interp1d(self.times, self.values)
            
    def plot(self):
        plt.plot(self.times, self.values)
        plt.xlabel("Seconds")
        plt.ylabel(str(self.units))
                
class SensorData:
    def __init__(self, name, rawdata):
        self.name = name
        self.rawdata = rawdata
        self.cleandata = self.process(self.rawdata)
        
    def process(self, data):
        return data
    
    def plot(self):
        self.rawdata.plot()
        self.cleandata.plot()
        plt.legend(["Raw Data", "Cleaned Data"])
        plt.title(self.name)
     

class PeriodicSensorData(SensorData):
    def __init__(self, name, rawdata):
        super().__init__(name, rawdata)

        self.fft = fftpack.fftshift(fftpack.fft(self.cleandata.values))
        bins = len(self.fft)
        nyquist = self.cleandata.samplerate_Hz/2
        freqmax = nyquist/2
        self.freqrange = np.linspace(-freqmax, freqmax, bins)
        
        self.peaks = self.find_peaks(self.cleandata)
        
    def plot(self):
        super().plot()
        plt.plot(self.peaks.times, self.peaks.values, linestyle="None", marker="^", markersize=10)
        plt.legend(["Raw Data", "Cleaned Data", "Peaks"])
    
    def plotfft(self):
        plt.plot(self.freqrange, self.fft)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(str(self.cleandata.units))
    
    def find_peaks(self,data):
        peaks, properties = signal.find_peaks(data.values)
        if len(peaks)>0:
            return Timeseries(data.values[peaks], timestamps=data.times[peaks], samplerate_Hz=None)
        else:
            return None

class BloodPressure(PeriodicSensorData):
    def __init__(self, data, times):
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="mmHg")
        super().__init__("Blood Pressure", rawdata)
    
    def process(self, data):
        cutoff_Hz = 10000
        nyquist_Hz = data.samplerate_Hz/2
        order = 1
        Wn = (cutoff_Hz/data.samplerate_Hz)/nyquist_Hz
        b, a = signal.butter(order, Wn, btype='low', analog=False)
        filtereddata = signal.lfilter(b, a, data.values)
        return Timeseries(filtereddata, timestamps=data.times, samplerate_Hz=data.samplerate_Hz, units=data.units)
    
    def find_peaks(self,data):
        peaks, properties = signal.find_peaks(data.values, distance=500, height=20)
        if len(peaks)>0:
            return Timeseries(data.values[peaks], timestamps=data.times[peaks], samplerate_Hz=None)
        else:
            return None


class RSP(PeriodicSensorData):
    def __init__(self, data, times):
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="volts")
        super().__init__("RSP", rawdata)
        
class PPG(PeriodicSensorData):
    def __init__(self, data, times):
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="volts")
        super().__init__("PPG", rawdata)

class AvgBloodPressure(SensorData):
    def __init__(self, data, times):
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="mmHg")
        super().__init__("Mean Arterial Pressure", rawdata)
        
class AvgPulse(SensorData):
    def __init__(self, data, times):
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="BPM")
        super().__init__("Pulse", rawdata)
    
class ElectrodermalActivity(SensorData):
    def __init__(self, data, times):
        rawdata = Timeseries(data, timestamps=times, samplerate_Hz=1000, units="microsiemens")
        super().__init__("Electrodermal Activity", rawdata)

class Parameter():
    def __init__(self, sensordata):
        self.data = self.extract(sensordata.cleandata)
    def extract(self, data):
        return np.diff(data)
    
class Test():
    def __init__(self, Trials):
        pass
    def procedure():
        pass

class Trial:
    def __init__(self, subjectname, datafilename, eventfilename):
        self.subjectname = subjectname
        self.rawdata = np.genfromtxt(datafilename, delimiter='	', skip_header=17)
        
        rawevents = np.loadtxt(eventfilename, dtype=str, delimiter=',', skiprows=2)
        self.events = dict()
        for i in range(len(rawevents)):
            self.events[rawevents[i, 1]]=rawevents[i,0].astype(float)
        
        times = (self.rawdata[:,0]*60)
        bp_data = self.rawdata[:,1]
#        rsp_data = self.rawdata[:,2]
#        ppg_data = self.rawdata[:,3]
#        mbp_data = self.rawdata[:,4]
#        bpm_data = self.rawdata[:,5]
#        eda_data = self.rawdata[:,6]
        
        self.sensors = dict()
        self.parameters = dict()
        
        bp = BloodPressure(bp_data, times)        
        self.sensors[bp.name] = bp
#        
#        rsp = RSP(rsp_data, times)        
#        self.sensors[rsp.name] = rsp
#        
#        ppg = PPG(ppg_data, times)        
#        self.sensors[ppg.name] = ppg
#        
#        mbp = AvgBloodPressure(mbp_data, times)        
#        self.sensors[mbp.name] = mbp
#        
#        bpm = AvgPulse(bpm_data, times)        
#        self.sensors[bpm.name] = bpm
#        
#        eda = ElectrodermalActivity(eda_data, times)        
#        self.sensors[eda.name] = eda


def load():
    names = ["Subject A","Subject B","Subject C","Subject D","Subject E","Subject F","Subject G"]
    datafiles = ["Subject A.txt","Subject B.txt","Subject C.txt","Subject D.txt","Subject E.txt","Subject F.txt","Subject G.txt"]
    eventfiles = ["SubjectA_events.csv","SubjectB_events.csv","SubjectC_events.csv","SubjectD_events.csv","SubjectE_events.csv","SubjectF_events.csv","SubjectG_events.csv"]
    
    subjects = dict()
    for i in range(len(names)):
        subjects[names[i]] = Trial(names[i], "data/"+datafiles[i], "events/"+eventfiles[i])
    return subjects


subjects = load()
for subject in subjects:
    plt.figure()
    subject.sensors["Blood Pressure"].plot()