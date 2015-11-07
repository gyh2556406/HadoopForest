package Dataset;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;


public class DataSet {

    private boolean[] isCategory;
    private double[][] features;
    private double[] labels;
    private int numAttributes;
    private int numInstnaces;

    public DataSet(String path) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String[] attInfo = reader.readLine().split(","); // attributes info
            System.out.print(attInfo.length);
            numAttributes = attInfo.length - 1;
            isCategory = new boolean[numAttributes + 1];
            for (int i = 0; i < isCategory.length; i++) {
                isCategory[i] = Integer.parseInt(attInfo[i]) == 1 ;
            }

            numInstnaces = 0;
            while (reader.readLine() != null) {
                numInstnaces++;
            }


            features = new double[numInstnaces][numAttributes];
            labels = new double[numInstnaces];
            System.out.println("reading " + numInstnaces + " exmaples with " + numAttributes + " attributes");

            reader = new BufferedReader(new FileReader(path));
            reader.readLine();
            String line;
            int ind = 0;
            while ((line = reader.readLine()) != null) {
                String[] atts = line.split(",");
                try {
                    for (int i = 0; i < atts.length - 1; i++) {
                        features[ind][i] = Double.parseDouble(atts[i]);
                    }
                }catch (NumberFormatException e){
                    e.printStackTrace();
                }

                labels[ind] = Double.parseDouble(atts[atts.length - 1]);
                ind++;
            }

            reader.close();


        } catch (FileNotFoundException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public ArrayList<Object> devideDataSet(){
        ArrayList<Object> devideResult = new ArrayList<Object>();

        int fold = 3;

        Random random = new Random(2015);
        int[] permutation = new int[10000];
        for (int i = 0; i < permutation.length; i++) {
            permutation[i] = i;
        }
        for (int i = 0; i < 10 * permutation.length; i++) {
            int repInd = random.nextInt(permutation.length);
            int ind = i % permutation.length;

            int tmp = permutation[ind];
            permutation[ind] = permutation[repInd];
            permutation[repInd] = tmp;
        }

        int[] perm = new int[numInstnaces];
        int ind = 0;
        for (int i = 0; i < permutation.length; i++) {
            if (permutation[i] < numInstnaces) {
                perm[ind++] = permutation[i];
            }
        }

        int share = numInstnaces / fold;

        boolean[] isCategoryDevide = isCategory;
        double[][] featuresDevide = features;
        double[] labelsDevide = labels;

        for (int f = 0; f < fold; f++) {
            //System.out.println("No." + (f + 1) + "  time for runing");

            int numTest = f < fold - 1 ? share : numInstnaces - (fold - 1) * share;
            double[][] trainFeatures = new double[numInstnaces - numTest][numAttributes];
            double[] trainLabels = new double[numInstnaces - numTest];
            double[][] testFeatures = new double[numTest][numAttributes];
            double[] testLabels = new double[numTest];
            //System.out.println("We get " + (dataset.getNumInstnaces() - numTest) + "examples for training");
            //System.out.println("We get " + numTest + "examples for testing");

            int indTrain = 0, indTest = 0;
            for (int j = 0; j < numInstnaces; j++) {
                if ((f < fold - 1 && (j < f * share || j >= (f + 1) * share)) || (f == fold - 1 && j < f * share)) {
                    System.arraycopy(featuresDevide[perm[j]], 0, trainFeatures[indTrain], 0, numAttributes);
                    trainLabels[indTrain] = labelsDevide[perm[j]];
                    indTrain++;
                } else {
                    System.arraycopy(featuresDevide[perm[j]], 0, testFeatures[indTest], 0, numAttributes);
                    testLabels[indTest] = labelsDevide[perm[j]];
                    indTest++;
                }
            }
            devideResult.add(trainFeatures);
            devideResult.add(trainLabels);
            devideResult.add(testFeatures);
            devideResult.add(testLabels);
            devideResult.add(isCategoryDevide);

        }
        return devideResult;
    }

    public boolean[] getIsCategory() {
        return isCategory;
    }

    public double[][] getFeatures() {
        return features;
    }

    public double[] getLabels() {
        return labels;
    }

    public int getNumAttributes() {
        return numAttributes;
    }

    public int getNumInstnaces() {
        return numInstnaces;
    }
}
