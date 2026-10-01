
################################
#    
#   Interfaccia per caratterizzazione spettroscopio
#   TWINS. 
#
#   Dipendenze:
#       - pipy
#       - seabreeze
#       - numpy
#       - pyqt
#   Dipendenze HamamatsuMiniSpec (C. Frank):
#       - numpyFFFF
#       - pyusb
#       - libusb
#
#   CONDA ENV: flame
#   Anno: 2024
#   Autore: Marco Gamba
#
################################
#   Install:
#    > conda create -n twins python=3.6.5
#    > conda activate twins
#    > conda install pip
#
#    > pip install --upgrade pip
#    > pip install pyqt5 pyqtgraph==0.11.0 pipython seabreeze numpy pyusb libusb
#
#
#   Usage:
#    > conda activate twins
#    > python caratteriz.py [motore spettrometro]
#
#   Options:
#   1. motore: e873 / c863 
#       quale motore attaccato (solo UN motore per volta)
#   2. spettrometro: flame / off
#       quale spettrometro attaccato
#       (flame=OceanOptics, off=Nessuno, default=Hamamatsu)
#
#

import sys
# import seabreeze
# seabreeze.use('pyseabreeze')    # da' problemi, forse PyUSB?
import seabreeze.spectrometers as sb
import HamamatsuMiniSpectrometer as hm      # Hamamatsu lib by Frank Carsten
# PhyiskInstrumente
from pipython import GCSDevice, datarectools, pitools

from pathlib import Path
from datetime import datetime
from time import sleep

import numpy as np
from numpy import ma

import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import (QSize, Qt,
                          QTimer, pyqtSlot,
                          QThreadPool, QRunnable,
                          QObject, pyqtSignal)
from PyQt5.QtWidgets import ( QApplication,
            QWidget, QMainWindow, QPushButton,
            QHBoxLayout, QVBoxLayout, QGridLayout,  # QFormLayout,
            QLabel, QToolBar, QAction, QStatusBar,
            QGroupBox, QLineEdit, QSpinBox, QCheckBox, 
            QProgressBar
)

from PyQt5.QtGui import QPalette, QColor, QCloseEvent


"""
## S/N dei nostri controller
sn_e873 = '0185500006'       #bianco -> collegato a Q-521.330
sn_mercury = '0185500006'   #nero -> collegato a
"""
sn_e873 = '024550347' #'0185500162'       #bianco -> collegato a Q-521.330
sn_mercury = '024550348' #'0185500162'   #nero -> collegato a

 
## DEFAULT parameters (sizes in mm)
START = 10
END = 15
STEP = 0.5

T_ACQ_MS = 100
MEAN_N = 1
W_MIN = 400
W_MAX = 1000

# Spettroscopio: OceanOptics Flame (USB2000+)
if sys.argv[-1] == 'flame':
    flame = sb.Spectrometer.from_first_available()
    flame.integration_time_micros(20000)
    
    l = len(flame.wavelengths())      # n. of pixels (solitamente 2048)

    print("Connesso spettrometro: ", flame)


# Spettroscopio: Hamamatsu C10082CA (default)
elif sys.argv[-1] != 'off':
    ham = hm.HamamatsuMiniSpecLibusb()
    ham.setIntegrationTime(20000) # 20 ms

    l = len(ham.wlArr)      # n. of pixels (solitamente 2048)

    print("Connesso spettrometro: ", ham)



REFMODES = ['FNL', ]  # reference the connected stages

# PRIMO motore (E-873)
# -> collegato a Q-521.330

if 'e873' in sys.argv:    
    e873 = GCSDevice()
    print( "Motori attaccati:", e873.EnumerateUSB() )
    #e873.InterfaceSetupDlg()           # solo su Windows
    e873.ConnectUSB(serialnum = sn_e873)
    print('Connesso a: {}'.format(e873.qIDN().strip()))

    STAGES = ['L-402.10SD', ]  # connect stages to axes
    print('initialize connected stages...')
    pitools.startup(e873, STAGES, REFMODES) 

    epox = e873.qPOS()
    epox = epox['1']
    print('E-873 PosX {} mm'.format(epox))

# SECONDO motore (C-863)
# -> collegato a M-112.1DG
if 'c863' in sys.argv:

    c863 = GCSDevice()
    print( "Motori attaccati:", c863.EnumerateUSB() )
    c863.ConnectUSB(serialnum = sn_mercury)
    print('Connesso a: {}'.format(c863.qIDN().strip()))

    STAGES = ['L-505.023212' ]#STAGES = ['L-402.10SD','L-505.023212' ]  # connect stages to axes
    print('initialize connected stages...')
    pitools.startup(c863, STAGES, REFMODES) 

    cpox = c863.qPOS()
    cpox = cpox['1']
    print('C-663 PosX: {} mm'.format(cpox))


#
# Oggetto SIGNALS - essenziale per IO fra thread
#
class WorkerSignals(QObject):
    '''
    Necessario per comunicare tra thread. Segnali custom sono possibili
    solo su QObject
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        ritorna str: posizione finale dell'asse

    error
        tuple (exctype, value, traceback.format_exc() )

    status
        object data returned from processing, anything

    '''
    finished = pyqtSignal(str)
    error = pyqtSignal(tuple)
    status = pyqtSignal(object)


#
#   Main thread - qua avviene tutta l'action
#
class Worker(QRunnable):
    '''
    Worker thread:
    qua dentro vanno le chiamate ai due controller 
    '''

    def __init__(self, plot, line, config, progress_l, go_btn, t_acq, mean, w_range, *args, **kwargs):
        super(Worker, self).__init__()
        self.args = args
        self.kwargs = kwargs

        self.grafico = plot
        self.linea = line
        self.config = config
        self.progress_l = progress_l

        self.go_btn  = go_btn
        self.counter = 0

        self.mean = mean.value()
        self.t_acq = t_acq.value()*1000     # in us
        self.t_acq_in = t_acq

        self.w_min, self.w_max = w_range

        self.signals = WorkerSignals()


    @pyqtSlot()
    def run(self):
        '''
        Funzione che esegue quando parte il thread
        '''
        print("Thread start")

        self.spec_dir = Path.home() / "OneDrive/Desktop/Spettri_Twins/spectra_{}/".format(datetime.now().strftime("%y%m%d"))
        self.spec_dir.mkdir(parents=True, exist_ok=True)      # crea se non esiste

        print("Destination directory: {}".format(self.spec_dir))

        self.counter += 1
        print("Run #{}".format( self.counter ))

        if 'e873' in sys.argv:
            epox = e873.qPOS()  # motore 1 (E-873)
            epox = epox['1']
            print(str(epox))

            if (self.counter % 10 == 1):
                print('PosX1: {}'.format( epox ))
            #sleep(1.5)
        
        
        self.go_btn.setEnabled(False)       # blocco per evitare double tap
        self.t_acq_in.setEnabled(False)
        self.start_scan()
        self.go_btn.setEnabled(True)
        self.t_acq_in.setEnabled(True)
 
        print("Thread complete")


    def start_scan(self):
        try:
            self.START = float(self.config.children()[2].text())
            STEP = float(self.config.children()[6].text())
            self.END = float(self.config.children()[4].text())
        except ValueError:
            print("Valori non numerici")

        if self.START > self.END:         # tiene in conto riavvolgimento
            STEP = -STEP
        # Boundaries del motore PI
        # TODO: andrebbero aggiustati PER ogni dispositivo PI attaccato
        if self.START < 0:
            self.START = 0
        if self.END > 30.800:
            self.END = 30.800

        print("Nuova scansione: {0}mm -> {2}mm (step {1})".format(self.START, STEP, self.END))

        #
        #  Gestione motore qua  
        #   muovo da : START a END 
        #
        
        # arange() meglio di linspace() perche' so esattamente su quali valori vado a finire
        for mm in np.arange(self.START, self.END+STEP, STEP):

            # lettura POS da sensore 
            if 'e873' in sys.argv: 
                epox = e873.qPOS()  # motore 1 (E-873)
                epox = epox['1']
                print('E-873 PosX [mm]:', epox)

            # Muovo a POS e acquisisco lettura da sensore 
            if 'c863' in sys.argv: 
                c863.MOV(1, mm)     # motore 2 (C-863)
                print('Comandato!')
                pitools.waitontarget(c863, 1)
                sleep(0.5)          # perche' non si ferma subito dopo il waitontarget()
                cpox = c863.qPOS()
                cpox = cpox['1']
                
                print('C-663 PosX [mm]:', cpox)

            # 9 places totale / 6 decimali
            self.filename="spectr_at_%09.6fmm.csv" % cpox       # importante: quanta precisione vuoi?
            

            # Ora parlo con spettroscopio (se collegato)
            if sys.argv[-1] == 'flame':
                spec.integration_time_micros(self.t_acq)  # setta tempo di integrazione

                
                self.I = np.empty((0,l))        # e' un array per ospitare più misure
                self.W = []
                
                #print(spec.intensities())
                for i in range(self.mean):
                    self.I = np.vstack( [self.I, spec.intensities()] )  # acquisisce spettro
                    self.W = spec.wavelengths()

                # ATTENZIONE! Butto via un po' di segnale perche' ha un picco off the charts
                self.I = self.I[:, 50:]
                self.W = self.W[50:]

                # print(self.I)
                # MEDIA: se ho settato self.mean
                self.I = np.mean( self.I , axis=0 )
                print(f"media su {self.mean} spettri: ",self.I)
                          
                # CUT the signal
                # TODO ottimizzazione: taglia il segnale PRIMA di mediare
                m = ma.masked_outside(self.W, self.w_min,self.w_max)        # maschera di crop
                self.W = m.compressed()
                self.I = ma.masked_array(self.I, m.mask).compressed()       # applico la stessa maschera alle I

                # print( self.I[400:405] )      # stampa 4 valori a caso

                np.savetxt(self.spec_dir / self.filename, (self.I,self.W), delimiter=",")
                

                self.linea.setData(self.W, self.I)

            # Hamamatsu
            elif sys.argv[-1] != 'off':
                ham.setIntegrationTime(self.t_acq)   # 20 ms
                self.I = np.empty((0,l))        # e' un array per ospitare più misure
                self.W = []
                
                #print(spec.intensities())
                for i in range(self.mean):
                    self.I = np.vstack( [self.I, ham.getSpectrum()] )  # acquisisce spettro
                    self.W = ham.wlArr

                # MEDIA: se ho settato self.mean
                self.I = np.mean( self.I , axis=0 )
                print(f"media su {self.mean} spettri: ",self.I)
                          
                # CUT the signal
                # TODO ottimizzazione: taglia il segnale PRIMA di mediare
                m = ma.masked_outside(self.W, self.w_min,self.w_max)        # maschera di crop
                self.W = m.compressed()
                self.I = ma.masked_array(self.I, m.mask).compressed()       # applico la stessa maschera alle I

                # TODO: salvo spettri su un file unico?
                np.savetxt(self.spec_dir / self.filename, (self.I,self.W), delimiter=",")
                
                # Update pyplot
                self.linea.setData(self.W, self.I)

            now = datetime.now()
            print("Written file {} at {}\n".format(self.filename, now.strftime("%H:%M:%S\n\n")))

            # self.progress.setValue(int(mm*10))    #  NOOOOOO
            self.signals.status.emit(cpox)# {'min':START,'max':END,'posx':cpox} )
            self.progress_l.setText('C-863 PosX: {} mm (target: {:.7f})'.format(cpox, mm))
            sleep(0.5)

        self.signals.finished.emit(str(cpox))

    def update_live(self, status):
        '''
            status: bool   (se attivare o no grafico live - ma lo sto usando?)
        '''
        #t_acq = int(t_acq) * 1000          # to int and convert in microseconds
        self.live = status

        print("Grafico {}".format("ON" if status else "OFF") )


#
# Oggetto SIGNALS - essenziale per IO fra thread
#
class LivePlotSignals(QObject):
    '''
    finished
        ritorna bool: 0 se terminato

    error
        tuple (exctype, value, traceback.format_exc() )

    status
        object data returned from processing, anything

    '''
    finished = pyqtSignal(bool)
    error = pyqtSignal(tuple)
    status = pyqtSignal(object)

class LivePlot(QRunnable):
    '''
    LivePlot thread:
        il suo unico compito e' di aggiornare lo spettro in tempo reale,
        se chiamato 
    '''

    def __init__(self, plot, line, t_acq, mean, *args, **kwargs):
        super(LivePlot, self).__init__()
        self.args = args
        self.kwargs = kwargs

        self.plot = plot
        self.line = line

        self.mean = mean
        self.status = Qt.Checked       # se attivato il thread vuol dire che e' checked
        self.t_acq = t_acq*1000        # in us

        self.w_min = W_MIN
        self.w_max = W_MAX

        self.signals = LivePlotSignals()

    @pyqtSlot()
    def run(self):
        '''
        Il worker aggiorna continuamente il grafico finche' non interrotto
        '''
        print("LiveGrafico start")
        try:
            while self.status == Qt.Checked:
                
                # Parlo con spettroscopio (se collegato)
                ham.setIntegrationTime(self.t_acq)
                #spec.integration_time_micros(self.t_acq)  # setta tempo di integrazione

                
                self.I = np.empty((0,l))        # e' un array per ospitare più misure
                self.W = []
                
                #print(spec.intensities())
                for i in range(self.mean):
                    self.I = np.vstack( [self.I, ham.getSpectrum()] )  # acquisisce spettro
                    self.W = ham.wlArr

                # ATTENZIONE! Butto via un po' di segnale perche' ha un picco off the charts
                #self.I = self.I[:, 50:]
                #self.W = self.W[50:]

                # print(self.I)
                self.I = np.mean( self.I , axis=0 )
                #print(f"media su {self.mean} spettri: ",self.I)
                          
                # CUT the signal
                # TODO ottimizzazione: taglia il segnale PRIMA di mediare
                m = ma.masked_outside(self.W, self.w_min,self.w_max)        # maschera di crop
                self.W = m.compressed()
                self.I = ma.masked_array(self.I, m.mask).compressed()       # applico la stessa maschera alle I

                # print( self.I[400:405] )      # stampa 4 valori a caso
                # print(self.W, self.I)
                self.line.setData(self.W, self.I)

                sleep(0.5)
                
        except NameError:
            print("Spettrometro non connesso")

        self.signals.finished.emit(False)

        print("LiveGrafico ended")

    def update_tacq(self, t_acq):
        '''
            t_acq: str   (new acquisition time in milliseconds)
        '''
        t_acq = t_acq * 1000          # to int and convert in microseconds
        self.t_acq = t_acq

        print("Updated acq time: {}us".format(t_acq) )

    def update_mean(self, mean):
        '''
            mean: str   (new n of means)
        '''
        mean = mean
        self.mean = mean

        print("Updated n of means: {}".format(mean) )
        
    def update_range(self, val, which):
        '''
            val: int   (beginning or end limit)
            which: int (0 = beginning, 1 = end)
        '''
        if which == 0 and val < self.w_max:
            self.w_min = val
        elif which == 1 and val > self.w_min:
            self.w_max = val

        print( f"Updated range: {self.w_min} - {self.w_max}" )

#
# Classe finestra principale Qt
#
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()       # non ho idea di perche' ma...
        
        self.setWindowTitle("Caratterizzazione spettro TWINS")

        self.config_group = QGroupBox("TWINS")
        
        # CONFIG TWINS
        self.start_in = QLineEdit()
        self.start_in.setText("{}".format(START))
        self.end_in = QLineEdit()
        self.end_in.setText("{}".format(END))
        self.step_in = QLineEdit()
        self.step_in.setText("{}".format(STEP))

        
        # SPECTROMETER
        self.acq_time = QSpinBox()
        self.acq_time.setRange(10,10000)
        self.acq_time.setValue(T_ACQ_MS)
        self.mean_n = QSpinBox()
        self.mean_n.setRange(1,100)
        self.mean_n.setValue(MEAN_N)
        self.live_spect = QCheckBox()
        self.live_spect.setCheckState(Qt.Unchecked)
        #if sys.argv[-1] == 'off':
        self.live_spect.stateChanged.connect(self.toggle_spect)
        
        self.w_min = QSpinBox()
        self.w_min.setRange(350,1050)       # min/max wavelength supportate
        self.w_min.setValue(W_MIN)
        self.w_max = QSpinBox()
        self.w_max.setRange(350,1100)
        self.w_max.setValue(W_MAX)


        self.spectr_group = QGroupBox("Spectrometer")

        self.spectr_layout = QGridLayout()
        
        self.spectr_layout.addWidget(QLabel("Tempo di acquisizione [ms]"), 0,0)
        self.spectr_layout.addWidget(self.acq_time, 0,1)
        self.spectr_layout.addWidget(QLabel("Media su spettri:"), 1,0)
        self.spectr_layout.addWidget(self.mean_n, 1,1)
        self.spectr_layout.addWidget(QLabel("da [nm]: "), 2,0)
        self.spectr_layout.addWidget(QLabel("a [nm]: "), 2,1)
        self.spectr_layout.addWidget(self.w_min, 3,0)
        self.spectr_layout.addWidget(self.w_max, 3,1)
        self.spectr_layout.addWidget(QLabel("Spettro live: "), 4,0)
        self.spectr_layout.addWidget(self.live_spect, 4,1)


        self.spectr_group.setLayout(self.spectr_layout)

        # BUTTON
        self.go_btn = QPushButton("Acquisisci")
        self.go_btn.setCheckable(True)
        # .clicked e' il(uno dei) "signal" del button
        self.go_btn.clicked.connect(self.start_acquisition) # start_acquisition e' lo "slot"
        self.go_btn.setCheckable(False)
        
        # STATUS
        self.status_l = QLabel("Motore a: ")
        self.status_l.setAlignment(Qt.AlignRight) 
        
        self.config_layout = QGridLayout()
        

        self.config_layout.addWidget(QLabel("Inizio [mm]"),0,0)
        self.config_layout.addWidget(self.start_in, 0,1)
        self.config_layout.addWidget(QLabel("Fine [mm]"),1,0)
        self.config_layout.addWidget(self.end_in, 1,1)
        self.config_layout.addWidget(QLabel("Step [mm]"),2,0)
        self.config_layout.addWidget(self.step_in, 2,1)
        self.config_layout.addWidget(self.go_btn, 3,0)
        self.config_layout.addWidget(self.status_l, 3,1)
        
        self.config_group.setLayout(self.config_layout)
        
        

        # PROGRESS BAR
        self.progress = QProgressBar()
        self.progress.setOrientation(Qt.Horizontal)
        self.progress.setRange(0,100)
        self.progress.setValue(50)

        self.progress_l = QLabel("In attesa...")

        self.footer_layout = QHBoxLayout()
        self.footer_layout.addWidget(self.progress)
        self.footer_layout.addWidget(self.progress_l)
        

        self.grafico = self.crea_plot()       # dummy values

        

        # inizializzo thread - per parlare con PI & Spectrometer
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())


        # Config
        self.conf_layout = QHBoxLayout()
        self.conf_layout.addWidget(self.config_group)
        self.conf_layout.addWidget(self.spectr_group)
        # aggiustare un po' le proporzioni
        # from: https://stackoverflow.com/a/71609363
        self.conf_layout.setStretch(0,1)  # 0th col stretch is 1
        self.conf_layout.setStretch(1,1)  # 1th col stretch is 2

        self.window_wdg = QVBoxLayout()
        self.window_wdg.addLayout(self.conf_layout)
        # Grafico
        self.window_wdg.addWidget(self.grafico)
        # Footer (progress bar)
        self.window_wdg.addLayout(self.footer_layout)


        window = QWidget()
        window.setLayout(self.window_wdg)

        self.setCentralWidget(window)


    def crea_plot(self):
        plot_graph = pg.PlotWidget()
        plot_graph.setTitle("Acquired spectrum")

        plot_graph.setLabel("left", "Intensity [#counts]")
        plot_graph.setLabel("bottom", "Wavelength [nm]")

        plot_graph.showGrid(x=True, y=True)

        pen = pg.mkPen("b", width=1)                                # linea blu
        
        time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 30]
        
        # Get a line reference
        self.line = plot_graph.plot(
            time,
            temperature,
            name="Spectrometer USB2000+",
            pen=pen,
            # symbol="o",
            # symbolSize=1,
            # symbolBrush="b",
        )
       

        return plot_graph
    
    def start_acquisition(self):
        if self.live_spect.isChecked():     # interrompe LivePlot durante l'acquisizione
            self.plotworker.status = Qt.Unchecked

        # WORKER: Lanca processo separato per acquisizione spettro/spostamenti
        self.worker = Worker(self.grafico, self.line, self.config_group, self.progress_l, self.go_btn, self.acq_time, self.mean_n, (self.w_min.value(), self.w_max.value()))   # chiamata al thread LETTURA
        # NOTA: si possono anche passare intere funzioni (sono oggetti in python!)

        self.threadpool.start(self.worker)   # Avvia il thread

        self.worker.signals.status.connect(self.aggiorna_barra)     #aggiorno automaticamente la barra!
        self.worker.signals.finished.connect(self.status_l.setText)
        
        # Not a good idea: non cambio il t_acq DURANTE l'acquisizione

    def aggiorna_barra(self, prog_value):
        if self.worker.END < self.worker.START:
            start = self.worker.END
            end = self.worker.START
        else:
            start = self.worker.START
            end = self.worker.END

        start = start*10        # aumento risoluzione
        end = end*10        #(ci dev'essere un modo migliore?)
        prog_value = prog_value*10
        
        self.progress.setRange(start, end)
        self.progress.setValue(prog_value)

    def toggle_spect(self, status):
        '''
            status: bool (specifies wether live spectrum on or off)
        '''
        if status == Qt.Checked:
            self.plotworker = LivePlot( self.grafico, self.line, self.acq_time.value(), self.mean_n.value() )
            self.threadpool.start(self.plotworker)          # Avvia il thread
            self.plotworker.signals.finished.connect(self.live_spect.setChecked)    #unchecka box se terminato
            print("Toggled worker (Status {})".format(status))
        else:
            # shuts down LivePlot
            self.plotworker.status = status
        
        if self.plotworker:
            self.acq_time.valueChanged.connect(self.plotworker.update_tacq)
            self.mean_n.valueChanged.connect(self.plotworker.update_mean)
            self.w_min.valueChanged.connect(lambda x: self.plotworker.update_range(x, 0))
            self.w_max.valueChanged.connect(lambda x: self.plotworker.update_range(x, 1))
            # custom signals from: https://www.pythonguis.com/tutorials/transmitting-extra-data-qt-signals/


# Qt stuff
app = QtWidgets.QApplication(sys.argv)

main = MainWindow()
main.show()

app.exec()

# Termina
if 'e873' in sys.argv:
    e873.CloseConnection()
if 'c663' in sys.argv:
    c863.CloseConnection()

if sys.argv[-1] != 'off':
    spec.close()



