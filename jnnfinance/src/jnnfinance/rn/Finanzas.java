/*
 * JNNFinance
 * 
 */

package jnnfinance.rn;

import org.joone.engine.DelayLayer;
import org.joone.engine.FullSynapse;
import org.joone.engine.Monitor;
import org.joone.engine.NeuralNetEvent;
import org.joone.engine.NeuralNetListener;
import org.joone.engine.SigmoidLayer;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.FileOutputSynapse;
import org.joone.io.YahooFinanceInputSynapse;
import org.joone.net.NeuralNet;

/**
 *
 * @author Lisandro
 */
public class Finanzas implements NeuralNetListener {
    
    int ventanaTemporal;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
         
        Finanzas finanzas = new Finanzas(); 
        NeuralNet red = finanzas.inicializar();
        finanzas.entrenar(red);
         
    }

    private void entrenar(NeuralNet red) {

        /* Inicialización del un monitor para coordina la red */
        Monitor monitor = red.getMonitor();
        monitor.setLearningRate(0.5);
        monitor.setMomentum(0.6);
        monitor.setLearning(true);
        
        
        /* Cantidad de filas que tiene el archivo de entrada, como no es un archivo pongo 0 */
        monitor.setTrainingPatterns(0);

        /* Definición de la cantidad de epochs, o lo que es lo mismo la cantidad de ejecuciones de la red */
        monitor.setTotCicles(10000);

        red.addNeuralNetListener(this);
        red.start();
        red.getMonitor().Go();
        red.join();
        
        System.out.println("Red detenida."); 
        System.out.println("Ultimo RMSE: "+red.getMonitor().getGlobalError());
        
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
        entrada.setRows(1);
        oculta1.setRows(15);
        oculta2.setRows(5);
        salida.setRows(1);
        
        /* Definición de la cantidad de días anteriores a tener en cuenta para las predicciones*/
        entrada.setTaps(ventanaTemporal - 1);

        /* Creación de las uniones entre las capas (sinapsis) */
        FullSynapse sinapsisEO1 = new FullSynapse(); /* entrada -> oculta1 */
        FullSynapse sinapsisO1O2 = new FullSynapse(); /* oculta1 -> oculta2 */
        FullSynapse sinapsisO2S = new FullSynapse(); /* oculta2 -> salida */

        sinapsisEO1.setName("Sinapsis entrada-oculta1");
        sinapsisO1O2.setName("Sinapsis oculta1-oculta2");
        sinapsisO2S.setName("Sinapsis oculta2-salida");

        entrada.addOutputSynapse(sinapsisEO1);
        oculta1.addInputSynapse(sinapsisEO1);
        oculta1.addOutputSynapse(sinapsisO1O2);
        oculta2.addInputSynapse(sinapsisO1O2);
        oculta2.addOutputSynapse(sinapsisO2S);
        salida.addInputSynapse(sinapsisO2S);
        
        /* Seteo de todo lo relacinado con la entrada de datos de Yahoo*/
        String fechaInicio = "30-apr-2007";
        String fechaFin="30-apr-2008";
        int primeraFila = 2;
        String simbolo = "MSFT";
        
        YahooFinanceInputSynapse flujoEntrada = iniciarYahoo(fechaInicio, fechaFin, primeraFila, simbolo);
        entrada.addInputSynapse(flujoEntrada);
        
        /* Definición de los datos de validación para el teacher*/
        YahooFinanceInputSynapse flujoEntrenamiento = iniciarYahoo(fechaInicio, fechaFin, primeraFila+1, simbolo);
        
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
        
//        /* Creación de un archivo para guardar los resultados de la red */
//        FileOutputSynapse flujoResultados = new FileOutputSynapse();
//        flujoResultados.setFileName(".\resultado.txt");
//        salida.addOutputSynapse(flujoResultados);
//        /* Inicio de la ejecución de las capas, se ejecutan en paralelo porque cada una es un hilo separado */
//        entrada.start();
//        oculta1.start();
//        oculta2.start();
//        salida.start();
//
//        entrada.setMonitor(monitor);
//        oculta1.setMonitor(monitor);
//        oculta2.setMonitor(monitor);
//        salida.setMonitor(monitor);
//
//        profe.setMonitor(monitor);
        
        return red;  
        
    }

    public void netStarted(NeuralNetEvent evento) {
        System.out.println("Entrenando...");
    }

    public void cicleTerminated(NeuralNetEvent evento) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void netStopped(NeuralNetEvent evento) {
        
        long delay = System.currentTimeMillis();
        System.out.println("Entrenamiento Finalizado después de "+delay+" ms.");
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

        /* Creación de un archivo para guardar el error de la red calculado por el teacher */
        FileOutputSynapse archivoError = new FileOutputSynapse();
        archivoError.setFileName("Archivo de error");
        profe.addResultSynapse(archivoError);

        return profe;
    }

    private YahooFinanceInputSynapse iniciarYahoo(String fechaInicio, String fechaFin, int primeraColumna,String simbolo) {

        YahooFinanceInputSynapse flujoEntrada = new YahooFinanceInputSynapse();
        
        flujoEntrada.setAdvancedColumnSelector("4");
        flujoEntrada.setName("Yahoo");
        flujoEntrada.setFirstRow(primeraColumna);
        flujoEntrada.setLastRow(0);
        flujoEntrada.setDateStart(fechaInicio);
        flujoEntrada.setDateEnd(fechaFin);
        flujoEntrada.setSymbol(simbolo);
        
       return flujoEntrada;
        
    }

}
