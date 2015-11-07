package node;

import RandomForeatUtils.DFUtils;
import TrainingMapRed.PredictModelMR;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class TreeNode extends Node{
    private static final Logger log = LoggerFactory.getLogger(PredictModelMR.PredictModelMapper.class);

    public int[] set;                // 样本下标集合
    public int[] attrIndex;         // 可用属性下标集合
    public double label;             // 标签
    public  int splitAttr;           // 该节点用于分割的属性下标
    public double[] splitPoints;    // 切割点 离散属性为多值，连续属性只有一个值
    public  TreeNode[] childrenNodes; // 子节点

    @Override
    protected Type getType() {
        return null;
    }

    @Override
    protected String getString() {
        return null;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        //log.info( "Start to Write Node...");

        writeNode(out);
    }

    @Override
    public void writeNode(DataOutput out) throws IOException {
        out.writeDouble(label);
        //log.info( "label Writen Done"+label);

        out.writeInt(splitAttr);
        //log.info("splitAttr Writen Done" + splitAttr);

        if(splitAttr >= 0) {
            if(splitPoints != null){
                out.writeInt(1);
                DFUtils.writeArray(out, splitPoints);

            }else{
                out.writeInt(-1);
            }


            if(childrenNodes != null){
                out.writeInt(1);
                //log.info( "childrenNodes != null childrenNodes.length to be write..."+ childrenNodes.length);

                DFUtils.writeArray(out, childrenNodes);
                //log.info( "childrenNodes Writen Done");
            }else{
                out.writeInt(-1);
            }

        }

    }

    @Override
    public void readFields(DataInput in) throws IOException {

        label = in.readDouble();
        //log.info("Read label Done \t" + label);

        splitAttr = in.readInt();
        //log.info("Read splitAttr Done \t" + splitAttr);

        if(splitAttr >= 0){
            int splitPointFlag = in.readInt();

            if(splitPointFlag == 1){
                splitPoints = DFUtils.readDoubleArray(in);
            }
            int childrenNodesFlag = in.readInt();

            if(childrenNodesFlag == 1) {
                childrenNodes = DFUtils.readNodeArray(in);
                //log.info("splitAttr > 0  Load the Four Done:\t" + splitAttr + "\t" + splitPoints.length + "\t" + childrenNodes.getClass() + "\t" + label);
            }
        }


    }
        
}
