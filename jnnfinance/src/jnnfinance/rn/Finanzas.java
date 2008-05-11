/*
 * JNNFinance
 * 
 */

package jnnfinance.rn;

import org.joone.engine.DelayLayer;
import org.joone.engine.FullSynapse;
import org.joone.engine.NeuralNetEvent;
import org.joone.engine.NeuralNetListener;
import org.joone.engine.SigmoidLayer;

/**
 *
 * @author Lisandro
 */
public class Finanzas implements NeuralNetListener {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
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

}
