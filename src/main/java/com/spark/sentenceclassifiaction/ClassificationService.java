package com.spark.sentenceclassifiaction;

import com.spark.sentenceclassifiaction.util.DataUtils;
import lombok.Getter;
import lombok.experimental.Accessors;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import scala.Tuple2;
import scala.collection.JavaConversions;
import scala.collection.mutable.WrappedArray;

@Accessors(fluent = true)
@Getter
public class ClassificationService {

    private static final String SPARK_APP_NAME = "Sentence Similarity Classification";
    private static final String SPARK_MASTER = "local[*]";

    private PreProcessingService preProcessingService;

    private SparkSession spark;

    private SparkContext sc;

    public ClassificationService() {
        SparkConf conf = new SparkConf().set("spark.sql.crossJoin.enabled", "true");
        spark = SparkSession.builder()
                .config(conf)
                .appName(SPARK_APP_NAME)
                .master(SPARK_MASTER)
                .getOrCreate();

        spark.udf().register("dotProduct",
                (WrappedArray<Double> vecA, WrappedArray<Double> vecB) -> {
                    return DataUtils.dotProduct((Double[]) JavaConversions.seqAsJavaList(vecA).toArray(),
                            (Double[]) JavaConversions.seqAsJavaList(vecB).toArray());
                }, DataTypes.DoubleType);

        sc = spark.sparkContext();
        preProcessingService = new PreProcessingService(spark);
    }


    /*
     *  process data ("Sentence", "Category") through data and
     *  return <testSentID, <trainSentLabel, Matched>>
     *  @trainData train data
     *  @testData test data
     */
    public JavaPairRDD<Integer, Tuple2<Integer, Boolean>> modelProcessing(Dataset<Row> trainData,
                                                                          Dataset<Row> testData) {

        PipelineModel preProcessingFit = preProcessingService.preProcessingPipeline().fit(trainData);

        trainData = preProcessingFit.transform(trainData);
        testData = preProcessingFit.transform(testData);

        trainData.createOrReplaceTempView("train_data_table");
        testData.createOrReplaceTempView("val_data_table");

        trainData.show();
        testData.show();

        Dataset<Row> joinedData = spark.sql("SELECT val_data_table.Sentence as sent, " +
                "SUM(1) AS count, train_data_table.Category AS train_cat, FROM train_data_table " +
                "CROSS JOIN val_data_table" +
                "WHERE dotProduct(train_data_table.NormEmbeddings, val_data_table.NormEmbeddings) >= 0.7 " +
                "GROUP BY sent, train_cat");

        joinedData.show();

        return null;
    }

}
