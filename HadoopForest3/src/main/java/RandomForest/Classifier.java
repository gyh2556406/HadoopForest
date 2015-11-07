package RandomForest;/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.Serializable;

/**
 *
 * @author daq
 */
public abstract class Classifier implements Cloneable, Serializable {

    public abstract void train();

    public abstract double predict(double[] features);

    public abstract void setOutputDirName(String outputDirName) ;

}
