package RandomForest;

import RandomForeatUtils.DFUtils;
import TrainingMapRed.PredictModelMR;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
/**
 * 
 *  Random Forest
 */

public class RandomForest extends Classifier implements Writable {
    private static final Logger log = LoggerFactory.getLogger(PredictModelMR.PredictModelMapper.class);

    private String outputDirName = "output";
    private int classifier; // 生成分类器的数量
    private boolean[] isCategoryTrain;
    private double[][] featuresTrain;
    private double[] labelsTrain;
    public RandomDecisionTree[] forestTrees;

    public RandomForest(int nTree,boolean[] isCategoryTrain,double[][] featuresTrain,double[] labelsTrain){
        this.classifier = nTree;
        this.isCategoryTrain = isCategoryTrain;
        this.featuresTrain = featuresTrain;
        this.labelsTrain = labelsTrain;
        this.forestTrees = new RandomDecisionTree[nTree];

    }


    public boolean[] getIsCategoryTrain() {
        return isCategoryTrain;
    }

    public RandomForest(){
    }


    @Override
    public void write(DataOutput dataOutput) throws IOException {
        //log.info( "Writing RandomForest...");
        //log.info( "Writing isCategoryTrain..."+isCategoryTrain.length);
        dataOutput.writeInt(isCategoryTrain.length);
        for (boolean anIsCategoryTrain : isCategoryTrain) {
            dataOutput.writeBoolean(anIsCategoryTrain);
        }

        //log.info( "Writing isCategoryTrain Done");

        //log.info( "Writing classifier..."+classifier);
        dataOutput.writeInt(classifier);
        for (int i = 0 ;i < classifier;i++) {
            //log.info( "Start to write Tree NO...."+i);

            forestTrees[i].write(dataOutput);
        }

    }
    /**
     * Reads the trees from the input and adds them to the existing trees
     */
    @Override
    public void readFields(DataInput dataInput) throws IOException {
        //log.info( "Readfield of RandomForest...");
        //log.info( "Read isCategoryTrain...");

        int isCategoryLen = dataInput.readInt();
        isCategoryTrain = new boolean[isCategoryLen];
        for (int i = 0 ;i<isCategoryLen;i++){
            isCategoryTrain[i] = dataInput.readBoolean();
        }
        //log.info( "Read isCategoryTrain Done...");


        int size = dataInput.readInt();
        forestTrees = new RandomDecisionTree[size];
        //log.info( "the size of Datainput.readInt..."+size);
        for (int i = 0; i < size; i++) {
            //log.info( "start to read Tree No."+i);
            forestTrees[i] = new RandomDecisionTree();
            forestTrees[i].set_ifCategory(isCategoryTrain);
            forestTrees[i].read(dataInput);
        }
    }

    /**
     * Read the forest from inputStream
     * @param dataInput - input forest
     * @return {@link RandomForest}
     * @throws IOException
     */
    public static RandomForest read(DataInput dataInput) throws IOException {
        log.info( "Start to read Forest");

        RandomForest forest = new RandomForest();
        forest.readFields(dataInput);
        return forest;
    }

    /**
     * Load the forest from a single file or a directory of files
     * @throws java.io.IOException
     */
    public static RandomForest load(Configuration conf, Path forestPath) throws IOException {
        FileSystem fs = forestPath.getFileSystem(conf);
        Path[] files;
        if (fs.getFileStatus(forestPath).isDir()) {
            files = DFUtils.listOutputFiles(fs, forestPath);
        } else {
            files = new Path[]{forestPath};
        }
        log.info( "files length:\t"+files.length);

        RandomForest forest = null;
        for (Path path : files) {
            try (FSDataInputStream dataInput = new FSDataInputStream(fs.open(path))) {
                if (forest == null) {
                    forest = read(dataInput);
                } else {
                    forest.readFields(dataInput);
                }
            }
        }

        return forest;

    }


    // 存储可放回抽取生成的新样本集
    public class RepickSamples {
        double[][] features;
        double[] labels;
        int[] index;

        public double[][] getFeatures(){
            return features;
        }

        public double[] getLabels(){
            return labels;
        }
    }


    public void train() {
        for (int i = 0; i < classifier; ++i) {
            RepickSamples samples = repickSamples(featuresTrain, labelsTrain);
            forestTrees[i] = new RandomDecisionTree();
            forestTrees[i].train(isCategoryTrain, samples.features, samples.labels);
        }
        //System.out.println(classifier+"SubDecision Trees Have Builded");

    }

    // 可放回收取新样本集
    public RepickSamples repickSamples(double[][] features, double[] labels) {
        RepickSamples samples = new RepickSamples();
        int size = labels.length;
        Random random = new Random();

        samples.features = new double[size][];
        samples.labels = new double[size];
        samples.index = new int[size];
        for (int i = 0; i < size; ++i) {
            int index = random.nextInt(size);
            samples.features[i] = features[index].clone();
            samples.labels[i] = labels[index];
            samples.index[i] = index;
        }
        //System.out.println("Samples Have bean Repicked");

        return samples;
    }



    public double predict(double[] features) {
        HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
        for (int i = 0; i < forestTrees.length; ++i) {
            double label = forestTrees[i].predict(features);
            if (counter.get(label) == null) {
                counter.put(label, 1);
            } else {
                int count = counter.get(label) + 1;
                counter.put(label, count);
            }
        }

        int temp_max = 0;
        double label = 0;
        Iterator<Double> iterator = counter.keySet().iterator();
        while (iterator.hasNext()) {
            double key = iterator.next();
            int count = counter.get(key);
            if (count > temp_max) {
                temp_max = count;
                label = key;
            }
        }
        //System.out.println("Samples Have bean Predicted");

        return label;
    }

    public void setOutputDirName(String name) {
        this.outputDirName = name;
    }
}


