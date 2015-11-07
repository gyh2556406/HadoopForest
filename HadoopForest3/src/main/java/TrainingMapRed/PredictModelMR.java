package TrainingMapRed;

import RandomForest.RandomForest;
import com.sun.istack.internal.Nullable;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;

/**
 * Created by geyanhao801 on 8/13/15.
 */
public class PredictModelMR extends Configured implements Tool {
    public static class PredictModelMapper extends Mapper<LongWritable,Text,LongWritable,Text> {
        private static double C = 0.05;
        private static final Logger log = LoggerFactory.getLogger(PredictModelMapper.class);
        private RandomForest randomForestMR ;
        private LongWritable outKey = new LongWritable();
        private Text outValue = new Text();
        public static void checkState(boolean expression, @Nullable Object errorMessage) {
            if(!expression) {
                throw new IllegalStateException(String.valueOf(errorMessage));
            }
        }
        /**
         * Load the Training data and Bagging
         */
        protected void setup(Context context) throws IOException,InterruptedException{
            super.setup(context);
            Configuration conf = context.getConfiguration();
            LocalFileSystem localFs = FileSystem.getLocal(conf);
            Path[] cacheFiles = DistributedCache.getLocalCacheFiles(conf);

            URI[] fallbackFiles = DistributedCache.getCacheFiles(conf);

            // fallback for local execution
            if (cacheFiles == null) {

                checkState(fallbackFiles != null, "Unable to find cached files!");

                cacheFiles = new Path[fallbackFiles.length];
                for (int n = 0; n < fallbackFiles.length; n++) {
                    cacheFiles[n] = new Path(fallbackFiles[n].getPath());
                }
            } else {
                for (int n = 0; n < cacheFiles.length; n++) {
                    cacheFiles[n] = localFs.makeQualified(cacheFiles[n]);
                    // fallback for local execution
                    if (!localFs.exists(cacheFiles[n])) {
                        cacheFiles[n] = new Path(fallbackFiles[n].getPath());
                    }
                }
            }

            checkState(cacheFiles.length > 0, "Unable to find cached files!");

            log.info("Loading the Forest...");

            randomForestMR = RandomForest.load(conf,cacheFiles[0]) ;
            log.info("Loading the data... Done");

        }

        protected void map(LongWritable key,Text value,Context context) throws IOException,InterruptedException{
            //use mapper to predict instance;
            //key == input rowNumber;
            //value == features+realLabel;

            String[] valueSplit = StringUtils.splitPreserveAllTokens(value.toString(), ",");

            double[] feature = new double[(valueSplit.length-1)];


            if(feature.length == randomForestMR.getIsCategoryTrain().length -1){
                try {
                    for (int i = 0; i < feature.length; i++) {
                        //log.info("valueSplit[i] is  ..." + valueSplit[i]);

                        feature[i] = Double.parseDouble(valueSplit[i]);
                    }
                }catch (NumberFormatException e){
                    e.printStackTrace();
                    return;
                }
                double realLabel = Double.parseDouble(valueSplit[valueSplit.length - 1]);

                double predictLabel = randomForestMR.predict(feature);
                //the output is the result comparing predict label with reallabel
                Boolean res = Math.abs(predictLabel - realLabel) < C;

                outKey.set(key.get());
                outValue.set(value.toString()+" "+predictLabel+" "+res.toString());
                context.write(outKey, outValue);
            }

        }

    }

    public static class PredictModelReducer extends Reducer<BooleanWritable,LongWritable,BooleanWritable,LongWritable>{
        private BooleanWritable outKey = new BooleanWritable();
        private LongWritable outValue = new LongWritable();
        protected void reduce(BooleanWritable key,Iterable<LongWritable> values,Context context) throws IOException, InterruptedException {

            long count = 0l;
            for(LongWritable item : values){
                count += item.get();
            }

            outKey = key;
            outValue.set(count);
            context.write(outKey, outValue);
        }
    }


    public int run(String[] args) throws Exception {
        String inputs = args[0];
        String outputPath = args[1];
        String partitionDate = args[2];

        Configuration conf = this.getConf();
        //conf.set("path",inputs);
        //conf.set("mapred.job.priority", "VERY_HIGH");
        //conf.set("mapred.child.java.opts", "-Xmx2048m");
        //conf.set("mapreduce.map.memory.mb", "1024");
        //conf.set("mapreduce.reduce.memory.mb", "4096");
        //conf.set("hadoop.tmp.dir", "/wls/applications/loki-app/tmp");

        Job job = new Job(conf, "PredictMR@" + partitionDate);
        job.setJarByClass(PredictModelMR.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setMapperClass(PredictModelMapper.class);
        job.setMapOutputKeyClass(BooleanWritable.class);
        job.setMapOutputValueClass(LongWritable.class);

        //job.setCombinerClass(PredictModelReducer.class);

        //job.setReducerClass(PredictModelReducer.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(TextOutputFormat.class);

        job.setNumReduceTasks(0);


        FileInputFormat.addInputPath(job, new Path(inputs));
        FileOutputFormat.setOutputPath(job, new Path(outputPath,"FinalResultFile"));

        return job.waitForCompletion(true) ? 1 : 0;
    }


}
