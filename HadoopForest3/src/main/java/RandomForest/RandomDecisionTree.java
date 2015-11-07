package RandomForest;

import TrainingMapRed.PredictModelMR;
import node.Node;
import node.TreeNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.*;


public class RandomDecisionTree {
    private static final Logger log = LoggerFactory.getLogger(PredictModelMR.PredictModelMapper.class);


    // 存储分割信息
    class SplitData {
        int splitAttr;
        double[] splitPoints;
        int[][] splitSets;       // 分割后新的样本集合的数组
    }

    class DataMSE {
        double floatValue;        // 存储增益率或MSE
        SplitData splitInfo;
    }

    // 当分割出现错误时抛出此异常
    class SplitException extends Exception {
    }

    private boolean ifClassification;
    private double[][] featureSplit;
    private boolean[] ifCategory;
    private double[] labelSplit;
    private double[] defaults;
    public TreeNode root;

    public RandomDecisionTree() {
    }

    public void set_ifCategory(boolean[] ifCategory){
        this.ifCategory = ifCategory;
    }


    public final void write(DataOutput out) throws IOException {
        //log.info( "Get to the  RandomDecisionTree Class...");

        root.write(out);
    }


    public void read(DataInput in) throws IOException {
        //log.info( "Get to the  RandomDecisionTree Class...");

        root = Node.read(in);
    }


    public void train(boolean[] isCategory, double[][] features, double[] labels) {
        ifClassification = isCategory[isCategory.length - 1];
        featureSplit = features;
        ifCategory = isCategory;
        labelSplit = labels;

        int set[] = new int[featureSplit.length];
        for (int i = 0; i < set.length; ++i) {
            set[i] = i;
        }

        int attr_index[] = new int[featureSplit[0].length];
        for (int i = 0; i < attr_index.length; ++i) {
            attr_index[i] = i;
        }

        // 处理缺失属性
        defaults = killMissingData();

        root = buildDecisionTree(set, attr_index);
    }

    private double[] killMissingData() {
        int num = ifCategory.length - 1;
        double[] defaults = new double[num];

        for (int i = 0; i < defaults.length; ++i) {
            if (ifCategory[i]) {
                // 离散属性取最多的值
                HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
                for (int j = 0; j < featureSplit.length; ++j) {
                    double feature = featureSplit[j][i];
                    if (!Double.isNaN(feature)) {
                        if (counter.get(feature) == null) {
                            counter.put(feature, 1);
                        } else {
                            int count = counter.get(feature) + 1;
                            counter.put(feature, count);
                        }
                    }
                }

                int max_time = 0;
                double value = 0;
                Iterator<Double> iterator = counter.keySet().iterator();
                while (iterator.hasNext()) {
                    double key = iterator.next();
                    int count = counter.get(key);
                    if (count > max_time) {
                        max_time = count;
                        value = key;
                    }
                }
                defaults[i] = value;
            } else {
                // 连续属性取平均值
                int count = 0;
                double total = 0;
                for (int j = 0; j < featureSplit.length; ++j) {
                    if (!Double.isNaN(featureSplit[j][i])) {
                        count++;
                        total += featureSplit[j][i];
                    }
                }
                defaults[i] = total / count;
            }
        }

        // 代换
        for (int i = 0; i < featureSplit.length; ++i) {
            for (int j = 0; j < defaults.length; ++j) {
                if (Double.isNaN(featureSplit[i][j])) {
                    featureSplit[i][j] = defaults[j];
                }
            }
        }
        return defaults;
    }

    public double predict(double[] features) {
        //log.info("Entering to the DecisionTree.predict...");

        // 处理缺失属性
        for (int i = 0; i < features.length; ++i) {
            if (Double.isNaN(features[i])) {
                features[i] = defaults[i];
            }
        }

        return predict_with_decision_tree(features, root);
    }



    private double predict_with_decision_tree(double[] features, TreeNode node) {
        if (node.childrenNodes == null) {
            return node.label;
        }

        double feature = features[node.splitAttr];

        if (ifCategory[node.splitAttr]) {
            // 离散属性
            for (int i = 0; i < node.splitPoints.length; ++i) {
                if (node.splitPoints[i] == feature) {
                    return predict_with_decision_tree(features, node.childrenNodes[i]);
                }
            }

            return node.label; // 不存在的属性取父节点样本的标签，减少叶子结点
        } else {
            // 连续属性
            if (feature < node.splitPoints[0]) {
                return predict_with_decision_tree(features, node.childrenNodes[0]);
            } else {
                return predict_with_decision_tree(features, node.childrenNodes[1]);
            }
        }

    }

    private TreeNode buildDecisionTree(int[] set, int[] attr_index) {
        TreeNode node = new TreeNode();
        node.set = set;
        node.attrIndex = attr_index;
        node.label = 0;
        node.childrenNodes = null;

        // 都为同类返回直接返回
        double label = labelSplit[node.set[0]];
        boolean flag = true;
        for (int i = 0; i < node.set.length; ++i) {
            if (labelSplit[node.set[i]] != label) {
                flag = false;
                break;
            }
        }
        if (flag) {
            node.label = label;
            return node;
        }

        // 没有可用属性标记为大多数(离散)或平均值(连续)
        if (ifClassification) {
            node.label = mostLabel(set);
        } else {
            node.label = meanValue(set);
        }
        if (node.attrIndex == null || node.attrIndex.length == 0) {
            return node;
        }

        // 寻找最优切割属性
        SplitData split_info = attribute_selection(node);
        node.splitAttr = split_info.splitAttr;
        // 没有可以分割的属性
        if (node.splitAttr < 0) {
            return node;
        }

        node.splitPoints = split_info.splitPoints;

        // 去掉已使用的离散属性，连续属性不做删除
        int[] child_attr_index = null;
        if (ifCategory[node.splitAttr]) {
            child_attr_index = new int[attr_index.length - 1];
            int t = 0;
            for (int index : attr_index) {
                if (index != node.splitAttr) {
                    child_attr_index[t++] = index;
                }
            }
        } else {
            child_attr_index = node.attrIndex.clone();
        }

        // 递归建立子节点
        node.childrenNodes = new TreeNode[split_info.splitSets.length];
        for (int i = 0; i < split_info.splitSets.length; ++i) {
            node.childrenNodes[i] = buildDecisionTree(split_info.splitSets[i], child_attr_index);
        }

        return node;
    }

    // 给定样本中出现最多的标签
    private double mostLabel(int[] set) {
        HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
        for (int item : set) {
            double label = labelSplit[item];
            if (counter.get(label) == null) {
                counter.put(label, 1);
            } else {
                int count = counter.get(label) + 1;
                counter.put(label, count);
            }
        }

        int maxTimes = 0;
        double label = 0;
        Iterator<Double> iterator = counter.keySet().iterator();
        while (iterator.hasNext()) {
            double key = iterator.next();
            int count = counter.get(key);
            if (count > maxTimes) {
                maxTimes = count;
                label = key;
            }
        }
        return label;
    }

    // 给定样本的标签平均值
    private double meanValue(int[] set) {
        double temp = 0;
        for (int index : set) {
            temp += labelSplit[index];
        }
        return temp / set.length;
    }

    private SplitData attribute_selection(TreeNode node) {
        SplitData result = new SplitData();
        result.splitAttr = -1;

        // 前剪枝
        double referenceValue = ifClassification ? 0.05 : -1;
        int numAttributeToPick = 4;
        if (node.set.length < numAttributeToPick) return result;

        // 生成随机选取的属性
        numAttributeToPick = (int) (Math.log(1 + node.attrIndex.length) / Math.log(2));
        int attrs[] = new int[numAttributeToPick];
        Random random = new Random();
        HashSet<Integer> hash = new HashSet<Integer>();
        for (int i = 0; i < numAttributeToPick; ++i) {
            int index = 0;
            do {
                index = random.nextInt(node.attrIndex.length);
            } while (hash.contains(index));
            hash.add(index);
            attrs[i] = node.attrIndex[index];
        }
        if (ifClassification) {
            for (int attribute : attrs) {
                try {
                    DataMSE gain_ratio_info = gain_ratio_use_attribute(node.set, attribute); // 分割错误会抛出分割异常
                    if (gain_ratio_info.floatValue > referenceValue) {
                        referenceValue = gain_ratio_info.floatValue;
                        result = gain_ratio_info.splitInfo;
                    }
                } catch (SplitException ex) { // 捕获异常，直接丢弃
                }
            }
        } else {
            for (int attribute : attrs) {
                try {
                    DataMSE mse_info = mseByAttribute(node.set, attribute);
                    if (referenceValue < 0 || mse_info.floatValue < referenceValue) {
                        referenceValue = mse_info.floatValue;
                        result = mse_info.splitInfo;
                    }
                } catch (SplitException ex) {
                }
            }
        }
        return result;
    }

    private SplitData splitWithAttribute(int[] set, int attribute) throws SplitException {
        SplitData result = new SplitData();
        result.splitAttr = attribute;

        if (ifCategory[attribute]) {
            // 离散属性
            int amountOfFeatures = 0;
            HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
            HashMap<Double, Integer> indexRecorder = new HashMap<Double, Integer>();
            for (int item : set) {
                double feature = featureSplit[item][attribute];
                if (counter.get(feature) == null) {
                    counter.put(feature, 1);
                    indexRecorder.put(feature, amountOfFeatures++);
                } else {
                    int count = counter.get(feature) + 1;
                    counter.put(feature, count);
                }
            }

            // 记录切割点
            result.splitPoints = new double[amountOfFeatures];

            for (Double key : indexRecorder.keySet()) {
                int value = indexRecorder.get(key);
                result.splitPoints[value] = key;
            }

            result.splitSets = new int[amountOfFeatures][];
            int[] tIndex = new int[amountOfFeatures];
            for (int i = 0; i < amountOfFeatures; ++i) tIndex[i] = 0;

            for (int item : set) {
                int index = indexRecorder.get(featureSplit[item][attribute]);
                if (result.splitSets[index] == null) {
                    result.splitSets[index] = new int[counter.get(featureSplit[item][attribute])];
                }
                result.splitSets[index][tIndex[index]++] = item;
            }
        } else {
            // 连续属性
            double[] features = new double[set.length];
            for (int i = 0; i < features.length; ++i) {
                features[i] = featureSplit[set[i]][attribute];
            }
            Arrays.sort(features);

            double referenceValue = ifClassification ? 0 : -1;
            double bestSplitPoint = 0;
            result.splitSets = new int[2][];
            for (int i = 0; i < features.length - 1; ++i) {
                if (features[i] == features[i + 1]) continue;
                double split_point = (features[i] + features[i + 1]) / 2;
                int[] subSetA = new int[i + 1];
                int[] subSetB = new int[set.length - i - 1];

                int aIndex = 0;
                int bIndex = 0;
                for (int j = 0; j < set.length; ++j) {
                    if (featureSplit[set[j]][attribute] < split_point) {
                        subSetA[aIndex++] = set[j];
                    } else {
                        subSetB[bIndex++] = set[j];
                    }
                }

                if (ifClassification) {
                    double temp = gainRatioUseNumericalAttribute(set, attribute, subSetA, subSetB);
                    if (temp > referenceValue) {
                        referenceValue = temp;
                        bestSplitPoint = split_point;
                        result.splitSets[0] = subSetA;
                        result.splitSets[1] = subSetB;
                    }
                } else {
                    double temp = (subSetA.length * mse(subSetA) + subSetB.length * mse(subSetB)) / set.length;
                    if (referenceValue < 0 || temp < referenceValue) {
                        referenceValue = temp;
                        bestSplitPoint = split_point;
                        result.splitSets[0] = subSetA;
                        result.splitSets[1] = subSetB;
                    }
                }
            }
            // 没有分割点，抛出分割异常
            if (result.splitSets[0] == null && result.splitSets[1] == null) throw new SplitException();
            result.splitPoints = new double[1];
            result.splitPoints[0] = bestSplitPoint;
        }
        return result;
    }

    // 计算给定样本集合的熵
    private double entropy(int[] set) {
        HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
        for (int item : set) {
            double label = labelSplit[item];
            if (counter.get(label) == null) {
                counter.put(label, 1);
            } else {
                int count = counter.get(label) + 1;
                counter.put(label, count);
            }
        }

        double result = 0;
        Iterator<Double> iterator = counter.keySet().iterator();
        while (iterator.hasNext()) {
            int count = counter.get(iterator.next());
            double p = (double) count / set.length;
            result += -p * Math.log(p);
        }

        return result;
    }

    // 增益率 C4.5
    private DataMSE gain_ratio_use_attribute(int[] set, int attribute) throws SplitException {
        DataMSE result = new DataMSE();
        double entropyBeforeSplit = entropy(set);

        double entropyAfterSplit = 0;
        double splitInformation = 0;
        result.splitInfo = splitWithAttribute(set, attribute);
        for (int[] subSet : result.splitInfo.splitSets) {
            entropyAfterSplit += (double) subSet.length / set.length * entropy(subSet);
            double p = (double) subSet.length / set.length;
            splitInformation += -p * Math.log(p);
        }
        result.floatValue = (entropyBeforeSplit - entropyAfterSplit) / splitInformation;
        return result;
    }

    private double gainRatioUseNumericalAttribute(int[] set, int attribute, int[] partA, int[] partB) {
        double entropyBeforeSplit = entropy(set);
        double entropyAfterSplit = (partA.length * entropy(partA) + partB.length * entropy(partB)) / set.length;

        double splitInformation = 0;
        double p = (double) partA.length / set.length;
        splitInformation += -p * Math.log(p);
        p = (double) partB.length / set.length;
        splitInformation += -p * Math.log(p);

        return (entropyBeforeSplit - entropyAfterSplit) / splitInformation;
    }

    private double mse(int[] set) {
        double mean = meanValue(set);

        double temp = 0;
        for (int index : set) {
            double t = labelSplit[index] - mean;
            temp += t * t;
        }
        return temp / set.length;
    }

    private DataMSE mseByAttribute(int[] set, int attribute) throws SplitException {
        DataMSE mseInfo = new DataMSE();
        mseInfo.floatValue = 0;
        mseInfo.splitInfo = splitWithAttribute(set, attribute);
        for (int[] sub_set : mseInfo.splitInfo.splitSets) {
            mseInfo.floatValue += (double) sub_set.length / set.length * mse(sub_set);
        }
        return mseInfo;
    }
}