import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

filename = './example_data/223_1b1_Al_sc_Meditron.wav'
senal, fm = librosa.load(filename) #signal and sampling rate

print(senal.shape)
print(fm)
librosa.display.waveplot(senal, sr=fm);


#Filtrado lineal
from linearFIR import filter_design, mfreqz
import scipy.signal as signal

def filtrar_senal(senal_f,fs):
    order, lowpass = filter_design(fs, locutoff = 0, hicutoff = 1000, revfilt = 0);
    order, highpass = filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1);
    senal_hp = signal.filtfilt(highpass, 1,senal);
    senal_lp = signal.filtfilt(lowpass, 1,senal_hp);
    return senal_lp

z = filtrar_senal(senal,fm)
z1 = np.asfortranarray(z)
librosa.display.waveplot(z1, sr=fm);

plt.figure()
plt.plot(z1[0:35],label='Umbralizada por Wavelet')
plt.show()

#Wavelet 
import pywt

def wthresh(coeff,thr):
    y   = list();
    s = wnoisest(coeff);
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])));
    return y;
    
def thselect(signal):
    Num_samples = 0;
    for i in range(0,len(signal)):
        Num_samples = Num_samples + signal[i].shape[0];
    
    thr = np.sqrt(2*(np.log(Num_samples)))
    return thr

def wnoisest(coeff):
    stdc = np.zeros((len(coeff),1));
    for i in range(1,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745;
    return stdc;


LL = int(np.floor(np.log2(z.shape[0])));

coeff = pywt.wavedec(z , 'db6', level=LL );

thr = thselect(coeff);
coeff_t = wthresh(coeff,thr);

x_rec = pywt.waverec( coeff_t, 'db6');
x_rec = x_rec[0:z.shape[0]];

plt.plot(x_rec,label='Umbralizada por Wavelet')
librosa.display.waveplot(x_rec, sr=fm);
  
#Ambos filtros
def Preprocesamiento_senal(senal_P, fmo):
    
    #Aplicación de los filtros lineales a la señal sin ruido cardíaco
    senal_filt = filtrar_senal(senal_P,fmo) 

    # aplicación del Wavelet    
    LL = int(np.floor(np.log2(senal_filt.shape[0])))
    
    coeff = pywt.wavedec(senal_filt, 'db36', level=LL );
    
    thr = thselect(coeff);
        
    coeff_t = wthresh(coeff,thr);
    
    #Señal a con filtro wavelet
    senalf = pywt.waverec(coeff_t, 'db36');
    senalf= senalf[0:senal_filt.shape[0]];
    
    return senalf

ffff = Preprocesamiento_senal(senal, fm)

librosa.display.waveplot(ffff, sr=fm);

plt.figure()

t = np.arange(0,len(ffff))

f, Pxx = signal.welch(ffff,fm,'hamming',fm*5, 0, fm*5, scaling='density')

plt.plot(f[(f >= 100) & (f <= 1000)],Pxx[(f >= 100) & (f <= 1000)])

plt.title("Señal con ruido eléctrico")
plt.ylabel('Amplitud')
plt.xlabel('Tiempo [s]')
plt.show()

librosa.display.waveplot(z1, sr=fm);

def desfase(senal_sf,fmu):
    
    senal_fil_des = Preprocesamiento_senal(senal_sf, fmu)
    x = senal_sf[0:fmu]
    x2 = senal_fil_des[0:fmu]
    return x,x2

xx,xx2 = desfase(senal,fm)

librosa.display.waveplot(xx, sr=fm);

librosa.display.waveplot(xx2, sr=fm);



