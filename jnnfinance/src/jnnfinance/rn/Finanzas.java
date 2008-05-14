/*
 * JNNFinance
 * 
 */

package jnnfinance.rn;

import java.util.Calendar;
import java.util.Date;
import org.joone.engine.DelayLayer;
import org.joone.engine.FullSynapse;
import org.joone.engine.Layer;
import org.joone.engine.Monitor;
import org.joone.engine.NeuralNetEvent;
import org.joone.engine.NeuralNetListener;
import org.joone.engine.SigmoidLayer;
import org.joone.engine.Synapse;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.YahooFinanceInputSynapse;
import org.joone.net.NeuralNet;

/**
 *
 * @author Lisandro
 */
public class Finanzas implements NeuralNetListener {
    
    /* Variables relacionadas con la red */
    private int neuronasEntrada;
    private int neuronasOculta1;
    private int neuronasOculta2;
    private int neuronasSalida;
    private int ventanaTemporal;
    
    /* Variables relacionadas con la ejecución y el aprendizaje */
    private double tasaDeAprendisaje;
    private double momentum;
    private int epochs;
    private int patronesDeEntrenamiento;
    
    /* Variables relacionadas con Yahoo */
    private Date fechaInicio;
    private Date fechaFin;
    private int primeraFila;
    private String simbolo;
    private String columnaYahoo;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
         
        Finanzas finanzas = new Finanzas(); 
        finanzas.setNeuronasEntrada(1);
        finanzas.setNeuronasOculta1(15);
        finanzas.setNeuronasOculta2(5);
        finanzas.setNeuronasSalida(1);
        finanzas.setVentanaTemporal(5);
        
        finanzas.setTasaDeAprendisaje(0.5);
        finanzas.setMomentum(0.6);
        finanzas.setEpochs(10000);
        finanzas.setPatronesDeEntrenamiento(200);
        
        Calendar inicioCal = Calendar.getInstance();        
        inicioCal.set(2008,4,1);
        finanzas.setFechaInicio(inicioCal.getTime());
        Calendar finCal = Calendar.getInstance();        
        finCal.set(2008,4,30);
        finanzas.setFechaFin(finCal.getTime());
        finanzas.setSimbolo("MSFT");
        finanzas.setPrimeraFila(2);
        finanzas.setColumnaYahoo("4");
        
        NeuralNet red = finanzas.inicializar();
        finanzas.entrenar(red);
         
    }
    
    private NeuralNet inicializar() {

        /* Definición de las capas de la red */
        DelayLayer entrada = new DelayLayer();
        SigmoidLayer oculta1 = new SigmoidLayer();
        SigmoidLayer oculta2 = new SigmoidLayer();
        SigmoidLayer salida = new SigmoidLayer();
        
        entrada.setLayerName("Capa de entrada");
        oculta1.setLayerName("Capa oculta 1");
        oculta2.setLayerName("Capa oculta 2");
        salida.setLayerName("Capa de salida");

        /* Definición de la cantidad de neuronas de cada capa */
        entrada.setRows(getNeuronasEntrada());
        oculta1.setRows(getNeuronasOculta1());
        oculta2.setRows(getNeuronasOculta2());
        salida.setRows(getNeuronasSalida());
        
        /* Definición de la cantidad de días anteriores a tener en cuenta para las predicciones*/
        entrada.setTaps(getVentanaTemporal() - 1);

        /* Creación de las uniones entre las capas (sinapsis) */
        FullSynapse sinapsisEO1 = new FullSynapse(); /* entrada -> oculta1 */
        FullSynapse sinapsisO1O2 = new FullSynapse(); /* oculta1 -> oculta2 */
        FullSynapse sinapsisO2S = new FullSynapse(); /* oculta2 -> salida */

        sinapsisEO1.setName("Sinapsis entrada-oculta1");
        sinapsisO1O2.setName("Sinapsis oculta1-oculta2");
        sinapsisO2S.setName("Sinapsis oculta2-salida");
        
        conectarCapas(entrada,sinapsisEO1, oculta1);
        conectarCapas(oculta1,sinapsisO1O2, oculta2);
        conectarCapas(oculta2,sinapsisO2S, salida);
              
        /* Seteo de todo lo relacinado con la entrada de datos de Yahoo*/
        YahooFinanceInputSynapse flujoEntrada = iniciarYahoo(getFechaInicio(), getFechaFin(), getPrimeraFila(), getSimbolo());
        entrada.addInputSynapse(flujoEntrada);
        
        /* Definición de los datos de validación para el teacher*/
        YahooFinanceInputSynapse flujoEntrenamiento = iniciarYahoo(getFechaInicio(), getFechaFin(), getPrimeraFila()+1, getSimbolo());
        
        /* Definición de un teacher para que entrene la red */
        TeachingSynapse profe = new TeachingSynapse();
        profe = iniciarTeacher(flujoEntrenamiento);
        
        /* Definición de un monitor para que coordine la red */
        Monitor monitor = new Monitor();
        
        /* Definición de la estructura de la red */
        NeuralNet red = new NeuralNet();
        red.addLayer(entrada, NeuralNet.INPUT_LAYER);
        red.addLayer(oculta1, NeuralNet.HIDDEN_LAYER);
        red.addLayer(oculta2, NeuralNet.HIDDEN_LAYER);
        red.addLayer(salida, NeuralNet.OUTPUT_LAYER);
        red.setMonitor(monitor);
        red.setTeacher(profe);
        
        /* Conección entre la capa de salida y el teacher */
        salida.addOutputSynapse(profe);
                
        return red;  
        
    }
    
    private void entrenar(NeuralNet red) {

        /* Inicialización del un monitor para coordina la red */
        Monitor monitor = red.getMonitor();
        monitor.setLearningRate(getTasaDeAprendisaje());
        monitor.setMomentum(getMomentum());
        monitor.setLearning(true);       
        monitor.setTrainingPatterns(patronesDeEntrenamiento);
        
        /* Cantidad de filas que tiene el archivo de entrada, como no es un archivo pongo 0 */
        monitor.setTrainingPatterns(0);

        /* Definición de la cantidad de epochs, o lo que es lo mismo la cantidad de ejecuciones de la red */
        monitor.setTotCicles(getEpochs());

        red.addNeuralNetListener(this);
        red.start();
        red.getMonitor().Go();
        red.join();
        
        System.out.println("Red detenida."); 
        System.out.println("Ultimo RMSE: "+red.getMonitor().getGlobalError());
        
    }
    
     private void conectarCapas(Layer origen, Synapse sinapsis, Layer destino) {

        origen.addOutputSynapse(sinapsis);
        destino.addInputSynapse(sinapsis);
    }

    public void netStarted(NeuralNetEvent evento) {
        System.out.println("Entrenando...");
    }

    public void cicleTerminated(NeuralNetEvent evento) {
        //throw new UnsupportedOperationException("Not supported yet.");
    }

    public void netStopped(NeuralNetEvent evento) {
        
        long delay = System.currentTimeMillis()/3600000000l;
        
        System.out.println("Entrenamiento Finalizado después de "+delay+" hs");
        System.exit(0);
        
    }

    public void errorChanged(NeuralNetEvent evento) {
        
        /* Obtención del monitor a partir del evento */
        Monitor mon = (Monitor)evento.getSource();
        
        /* Epoch actual */
        long c = (mon.getTotCicles()-mon.getCurrentCicle());
        
        System.out.println("Ciclo: "+ c + " - RMSE: " + mon.getGlobalError());
        
        
    }

    public void netStoppedError(NeuralNetEvent evento, String arg1) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    private TeachingSynapse iniciarTeacher(YahooFinanceInputSynapse flujoEntrenamiento) {

        /* Definición de un teacher para que entrene la red */
        TeachingSynapse profe = new TeachingSynapse();

        profe.setDesired(flujoEntrenamiento);

//        /* Creación de un archivo para guardar el error de la red calculado por el teacher */
//        FileOutputSynapse archivoError = new FileOutputSynapse();
//        archivoError.setFileName("Archivo de error");
//        profe.addResultSynapse(archivoError);

        return profe;
    }

    @SuppressWarnings("deprecation")
    private YahooFinanceInputSynapse iniciarYahoo(Date fechaInicio, Date fechaFin, int primeraColumna,String simbolo) {

        YahooFinanceInputSynapse flujoEntrada = new YahooFinanceInputSynapse();
        
        flujoEntrada.setAdvancedColumnSelector(getColumnaYahoo());
        flujoEntrada.setName("Yahoo");
        flujoEntrada.setFirstRow(primeraColumna);
        flujoEntrada.setLastRow(0);
                
        flujoEntrada.setStartDate(fechaInicio);
        flujoEntrada.setEndDate(fechaFin);
        flujoEntrada.setSymbol(simbolo);
        
       return flujoEntrada;
        
    }

    public int getNeuronasEntrada() {
        return neuronasEntrada;
    }

    public void setNeuronasEntrada(int neuronasEntrada) {
        this.neuronasEntrada = neuronasEntrada;
    }

    public int getNeuronasOculta1() {
        return neuronasOculta1;
    }

    public void setNeuronasOculta1(int neuronasOculta1) {
        this.neuronasOculta1 = neuronasOculta1;
    }

    public int getNeuronasOculta2() {
        return neuronasOculta2;
    }

    public void setNeuronasOculta2(int NeuronasOculta2) {
        this.neuronasOculta2 = NeuronasOculta2;
    }

    public int getNeuronasSalida() {
        return neuronasSalida;
    }

    public void setNeuronasSalida(int neuronasSalida) {
        this.neuronasSalida = neuronasSalida;
    }

    public int getVentanaTemporal() {
        return ventanaTemporal;
    }

    public void setVentanaTemporal(int ventanaTemporal) {
        this.ventanaTemporal = ventanaTemporal;
    }

    public double getTasaDeAprendisaje() {
        return tasaDeAprendisaje;
    }

    public void setTasaDeAprendisaje(double tasaDeAprendisaje) {
        this.tasaDeAprendisaje = tasaDeAprendisaje;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public Date getFechaInicio() {
        return fechaInicio;
    }

    public void setFechaInicio(Date fechaInicio) {
        this.fechaInicio = fechaInicio;
    }

    public Date getFechaFin() {
        return fechaFin;
    }

    public void setFechaFin(Date fechaFin) {
        this.fechaFin = fechaFin;
    }

    public int getPrimeraFila() {
        return primeraFila;
    }

    public void setPrimeraFila(int primeraFila) {
        this.primeraFila = primeraFila;
    }

    public String getSimbolo() {
        return simbolo;
    }

    public void setSimbolo(String simbolo) {
        this.simbolo = simbolo;
    }

    public String getColumnaYahoo() {
        return columnaYahoo;
    }

    public void setColumnaYahoo(String columnaYahoo) {
        this.columnaYahoo = columnaYahoo;
    }

    public int getPatronesDeEntrenamiento() {
        return patronesDeEntrenamiento;
    }

    public void setPatronesDeEntrenamiento(int patronesDeEntrenamiento) {
        this.patronesDeEntrenamiento = patronesDeEntrenamiento;
    }

}
