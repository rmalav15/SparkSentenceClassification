package com.spark.sentenceclassifiaction.util;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lower;
import static org.apache.spark.sql.types.DataTypes.StringType;
import static org.apache.spark.sql.types.DataTypes.createStructField;

public class DataUtils {

    /*
     *  Data loading function specific to Kaggle San Francisco Crime Classification dataset
     *  @csvPath path to csv file
     */
    public static Dataset<Row> readData(SparkSession spark, String csvPath) {
        return spark.read().option("header", true).csv(csvPath)
                .select("Descript", "Category")
                .withColumn("Descript", lower(col("Descript")))
                .withColumnRenamed("Descript", "Sentence");
    }

    public static Dataset<Row> generateDummyData(SparkSession spark,
                                                 String csvPath, int totalTrainingCount, int totalQueryCount) {
        return null;
    }

    public static Dataset<Row> generateDummyData(SparkSession spark) {
        List<Row> data = Arrays.asList(
                RowFactory.create("0c", "Hi I heard about Spark."),
                RowFactory.create("1c", "I wish Java could use case classes."),
                RowFactory.create("2c", "There is PySpark also."),
                RowFactory.create("0c", "Spark modules."),
                RowFactory.create("1c", "Java interfaces."),
                RowFactory.create("1c", "Spring boot Java."),
                RowFactory.create("2c", "Something PySpark."),
                RowFactory.create("1c", "Something Java.")
        );

        StructType schema = new StructType(new StructField[]{
                createStructField("Category", StringType, false),
                createStructField("Sentence", StringType, false)
        });

        return spark.createDataFrame(data, schema);
    }

    public static double dotProduct(Double[] vecA, Double[] vecB) {
        assert vecA.length == vecB.length;
        return IntStream.range(0, vecA.length)
                .parallel()
                .mapToDouble(id -> vecA[id] * vecB[id])
                .reduce(0.0, Double::sum);
    }

    public static MulticlassMetrics evaluate(JavaPairRDD<String, String> prediction,
                                             JavaPairRDD<String, String> gt) {

        JavaPairRDD<Object, Object> predictionAndLabels = gt.join(prediction)
                .mapToPair(t -> new Tuple2<>(t._2._2, t._2._1));
        return new MulticlassMetrics(predictionAndLabels.rdd());
    }
}
