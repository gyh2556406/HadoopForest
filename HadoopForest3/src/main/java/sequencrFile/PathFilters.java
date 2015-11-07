package sequencrFile;

/**
 * Created by geyanhao801 on 8/18/15.解析集群文件路径
 */
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;

/**
 * Supplies some useful and repeatedly-used instances of {@link PathFilters}.
 */
public final class PathFilters {

    private static final PathFilter PART_FILE_INSTANCE = new PathFilter() {
        public boolean accept(Path path) {
            String name = path.getName();
            return name.startsWith("part-") && !name.endsWith(".crc");
        }
    };

    /**
     * Pathfilter to read the final clustering file.
     */
    private static final PathFilter CLUSTER_FINAL = new PathFilter() {
        public boolean accept(Path path) {
            String name = path.getName();
            return name.startsWith("clusters-") && name.endsWith("-final");
        }
    };

    private static final PathFilter LOGS_CRC_INSTANCE = new PathFilter() {
        public boolean accept(Path path) {
            String name = path.getName();
            return !(name.endsWith(".crc") || name.startsWith(".") || name.startsWith("_"));
        }
    };

    private PathFilters() {
    }

    /**
     * @return {@link PathFilters} that accepts paths whose file name starts with "part-". Excludes
     * ".crc" files.
     */
    public static PathFilter partFilter() {
        return PART_FILE_INSTANCE;
    }

    /**
     * @return {@link PathFilters} that accepts paths whose file name starts with "part-" and ends with "-final".
     */
    public static PathFilter finalPartFilter() {
        return CLUSTER_FINAL;
    }

    /**
     * @return {@link PathFilters} that rejects paths whose file name starts with "_" (e.g. Cloudera
     * _SUCCESS files or Hadoop _logs), or "." (e.g. local hidden files), or ends with ".crc"
     */
    public static PathFilter logsCRCFilter() {
        return LOGS_CRC_INSTANCE;
    }

}

