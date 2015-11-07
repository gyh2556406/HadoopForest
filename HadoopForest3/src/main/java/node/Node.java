package node;

import org.apache.hadoop.io.Writable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Represents an abstract node of a decision tree
 */
public abstract class Node implements Writable {
    private static final Logger log = LoggerFactory.getLogger(Node.class);

    protected enum Type {;}

  protected abstract Type getType();

  public static TreeNode read(DataInput in) throws IOException {
      log.info( "Read Field of Node...");

      TreeNode node = new TreeNode();
      node.readFields(in);

      return node;
  }
  
  @Override
  public final String toString() {
    return getType() + ":" + getString() + ';';
  }
  
  protected abstract String getString();
  
  @Override
  public void write(DataOutput out) throws IOException {
    writeNode(out);
  }
  
  public abstract void writeNode(DataOutput out) throws IOException;
  
}
