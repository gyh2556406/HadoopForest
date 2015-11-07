/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

import RandomForest.ForestBuilder;
import TrainingMapRed.PredictModelMR;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Main {
    private static final Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws Exception {
        System.out.println("for RandomForest");
        String[] dataPaths = new String[]{"HadoopForest2/resources/traindata0804(DWXZDM).csv"};
        Path testDataPath = new Path("HadoopForest2/resources/testData0804(DWXZDM).data");

        int partitionDate = 1;

        String path = dataPaths[0];
        String outputPath = "HadoopForest2/forestPath";
        Configuration conf = new Configuration();

        ForestBuilder randomForest = new ForestBuilder(path,outputPath);
        randomForest.buildForest(conf);
        log.info( "Writing RandomForest To the CacheFile...");

        DistributedCache.addCacheFile(randomForest.getOutputPath().toUri(), conf);
        System.out.println("DistributedCache.addCacheFile  Done");

        String[] params = new String[3];
        params[0] = testDataPath.toString();
        params[1] = "HadoopForest2/ResultForPath"+partitionDate;
        params[2] = String.valueOf(partitionDate);
        System.out.println("PredictModelMR build:");

        int res = -1;
        try {
            res = ToolRunner.run(conf, new PredictModelMR(), params);
        } catch (Exception e) {
              // TODO Auto-generated catch block
            e.printStackTrace();
        }

        System.out.println("Success or not:" + res);

    }
}
