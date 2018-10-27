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
        
        print("        - Cleaning data")
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

#        print("        - Processing FFT")
#        self.fft = fftpack.fftshift(fftpack.rfft(self.cleandata.values))
#        bins = len(self.fft)
#        nyquist = self.cleandata.samplerate_Hz/2
#        freqmax = nyquist/2
#        self.freqrange = np.linspace(-freqmax, freqmax, bins)
        
        
        print("        - Finding Peaks")
        self.peaks = self.find_peaks(self.cleandata)
        
    def plot(self):
        super().plot()
        plt.plot(self.peaks.times, self.peaks.values, linestyle="None", marker="^", markersize=10)
        plt.legend(["Raw Data", "Cleaned Data", "Peaks"])
    
#    def plotfft(self):
#        plt.plot(self.freqrange, self.fft)
#        plt.xlabel("Frequency (Hz)")
#        plt.ylabel(str(self.cleandata.units))
    
    def find_peaks(self,data):
        peaks, properties = signal.find_peaks(data.values)
        if len(peaks)>0:
            return Timeseries(data.values[peaks], timestamps=data.times[peaks], samplerate_Hz=None)
        else:
            return None

class BloodPressure(PeriodicSensorData):
    def __init__(self, data, times, peakinterval=500):
        self.peakinterval = peakinterval
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
        peaks, properties = signal.find_peaks(data.values, distance=self.peakinterval, height=20)
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
    def __init__(self, subjectname, rawdata, param):
        
        print("+ Creating "+subjectname)
        self.subjectname = subjectname
                
        print("    - Reading event data from file")        
        rawevents = np.loadtxt(param["eventfile"], dtype=str, delimiter=',', skiprows=2)
        
        print("    - Processing event data")
        self.events = dict()
        for i in range(len(rawevents)):
            self.events[rawevents[i, 1]]=rawevents[i,0].astype(float)
        
        times = (rawdata[:,0]*60)
        bp_data = rawdata[:,1]
#        rsp_data = self.rawdata[:,2]
#        ppg_data = self.rawdata[:,3]
#        mbp_data = self.rawdata[:,4]
#        bpm_data = self.rawdata[:,5]
#        eda_data = self.rawdata[:,6]
        
        self.sensors = dict()
        self.parameters = dict()
        
        
        print("    - Processing Blood Pressure data")
        bp = BloodPressure(bp_data, times, param["bp_peakdist"])        
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


def load_data(params):
    print("+ Loading trial data")
    rawdata = dict()
    for subject in iter(params):
        print("    - loading "+subject)
        rawdata[subject] = np.genfromtxt(params[subject]["datafile"], delimiter='	', skip_header=17)
    return rawdata

def generate_subjects(params, rawdata):
    subjects = dict()
    for subject in iter(params):
        subjects[subject] = Trial(subject, rawdata[subject], params[subject])
    return subjects



params = {"Subject A": {"datafile":     "data/Subject A.txt",
                            "eventfile":    "events/SubjectA_events.csv",
                            "bp_peakdist":  500},

              "Subject B": {"datafile":     "data/Subject B.txt",
                            "eventfile":    "events/SubjectB_events.csv",
                            "bp_peakdist":  300},
                            
              "Subject C": {"datafile":     "data/Subject C.txt",
                            "eventfile":    "events/SubjectC_events.csv",
                            "bp_peakdist":  500},
                            
              "Subject D": {"datafile":     "data/Subject D.txt",
                            "eventfile":    "events/SubjectD_events.csv",
                            "bp_peakdist":  500},
                            
              "Subject E": {"datafile":     "data/Subject E.txt",
                            "eventfile":    "events/SubjectE_events.csv",
                            "bp_peakdist":  500},
                            
              "Subject F": {"datafile":     "data/Subject F.txt",
                            "eventfile":    "events/SubjectF_events.csv",
                            "bp_peakdist":  500},
                            
              "Subject G": {"datafile":     "data/Subject G.txt",
                            "eventfile":    "events/SubjectG_events.csv",
                            "bp_peakdist":  500}
             }


rawdata = load_data(params)
subjects = generate_subjects(params, rawdata)
for subject in subjects.values():
    plt.figure()
    subject.sensors["Blood Pressure"].plot()