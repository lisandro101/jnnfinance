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
import org.joone.io.YahooFinanceInputSynapse;

/**
 *
 * @author Lisandro
 */
public class Finanzas implements NeuralNetListener {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
         
        Finanzas finanzas = new Finanzas(); 
        finanzas.iniciar();
         
    }

    private void iniciar() {

        DelayLayer entrada = new DelayLayer();
        SigmoidLayer oculta1 = new SigmoidLayer();
        SigmoidLayer oculta2 = new SigmoidLayer();
        SigmoidLayer salida = new SigmoidLayer();

        entrada.setLayerName("entrada");
        oculta1.setLayerName("oculta 1");
        oculta2.setLayerName("oculta 2");
        salida.setLayerName("salida");

        entrada.setRows(1);
        oculta1.setRows(15);
        oculta2.setRows(5);
        salida.setRows(1);

        FullSynapse sinapsisEO1 = new FullSynapse(); /* entrada -> oculta1 */
        FullSynapse sinapsisO1O2 = new FullSynapse(); /* oculta1 -> oculta2 */
        FullSynapse sinapsisO2S = new FullSynapse(); /* oculta2 -> salida */

        sinapsisEO1.setName("entrada-oculta1");
        sinapsisO1O2.setName("oculta1-oculta2");
        sinapsisO2S.setName("oculta2-salida");

        entrada.addOutputSynapse(sinapsisEO1);
        oculta1.addInputSynapse(sinapsisEO1);
        oculta1.addOutputSynapse(sinapsisO1O2);
        oculta2.addInputSynapse(sinapsisO1O2);
        oculta2.addOutputSynapse(sinapsisO2S);
        salida.addInputSynapse(sinapsisO2S);
        
        Monitor monitor = iniciarMonitor();

        entrada.setMonitor(monitor);
        oculta1.setMonitor(monitor);
        oculta2.setMonitor(monitor);
        salida.setMonitor(monitor);

        monitor.addNeuralNetListener(this);
        
        String fechaInicio = "30-apr-2007";
        String fechaFin="30-apr-2008";
        int primeraColumna = 2;
        String simbolo = "MSFT";
        YahooFinanceInputSynapse flujoEntrada = iniciarYahoo(fechaInicio, fechaFin, primeraColumna, simbolo);
        entrada.addInputSynapse(flujoEntrada);
        
        
        
        
        
        
        
    }

    public void netStarted(NeuralNetEvent arg0) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void cicleTerminated(NeuralNetEvent arg0) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void netStopped(NeuralNetEvent arg0) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void errorChanged(NeuralNetEvent arg0) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void netStoppedError(NeuralNetEvent arg0, String arg1) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    private Monitor iniciarMonitor() {

        Monitor monitor = new Monitor();
        monitor.setLearningRate(0.5);
        monitor.setMomentum(0.6);

        return monitor;
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
