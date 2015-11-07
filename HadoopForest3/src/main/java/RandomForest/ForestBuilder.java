package RandomForest;

import Dataset.DataSet;
import RandomForeatUtils.DFUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by geyanhao801 on 8/18/15.
 */
public class ForestBuilder extends Configured {
    public ForestBuilder(String path,String ouputPath){
        this.dataPath = new Path(path);
        this.outputPath = new Path(ouputPath);
    }

    private static final Logger log = LoggerFactory.getLogger(ForestBuilder.class);

    private Path dataPath;

    private Path outputPath;

    public Path getOutputPath(){
        return outputPath;
    }


    public void buildForest(Configuration conf) throws IOException, ClassNotFoundException, InterruptedException {

        //Set nTree
        int nTrees = 150;

        DataSet dataSet = new DataSet(dataPath.getName());
        RandomForest forest = new RandomForest(nTrees,dataSet.getIsCategory(),dataSet.getFeatures(),dataSet.getLabels());
        forest.setOutputDirName(outputPath.getName());

        log.info("Building the forest...");
        long time = System.currentTimeMillis();
        forest.train();
        double[] feature = new double[]{1d,1d,1d,1d,41d,18d,11d,2.5d,1d,1d,1d,10d,44d};

        log.info("Predict Result: {}", forest.predict(feature));

        time = System.currentTimeMillis() - time;
        log.info("Build Time: {}", DFUtils.elapsedTime(time));
        //log.info("Forest num Nodes: {}", forest.nbNodes());
        //log.info("Forest mean num Nodes: {}", forest.meanNbNodes());
        //log.info("Forest mean max Depth: {}", forest.meanMaxDepth());

        // store the decision forest in the output path
        Path forestPath = new Path(outputPath, "forest.seq");
        outputPath = forestPath;
        log.info("Storing the forest in: {}", forestPath);
        /**
        for(int i =0;i< forest.forestTrees[1].root.childrenNodes.length;i++){
            if(forest.forestTrees[0].root.childrenNodes != null){
                log.info("forestTrees[1] ChildNode Scan split_attr :"+forest.forestTrees[0].root.childrenNodes[i].split_attr);
                log.info("forestTrees[1] ChildNode Scan label :"+forest.forestTrees[0].root.childrenNodes[i].label);
                log.info("forestTrees[1] ChildNode Scan split_points :"+forest.forestTrees[0].root.childrenNodes[i].split_points[0]);
                log.info("forestTrees[1] ChildNode Scan childrenNodes.length :"+forest.forestTrees[0].root.childrenNodes.length);
            }
        }*/

        DFUtils.storeWritable(conf, forestPath, forest);
    }
    /**
    protected static Data loadData(Configuration conf, Path dataPath, Dataset dataset) throws IOException {
        log.info("Loading the data...");
        FileSystem fs = dataPath.getFileSystem(conf);
        Data data = DataLoader.loadData(dataset, fs, dataPath);
        log.info("Data Loaded");

        return data;
    }*/
}
