package com.spark.sentenceclassifiaction.util;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
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
                RowFactory.create("0", "Hi I heard about Spark."),
                RowFactory.create("1", "I wish Java could use case classes."),
                RowFactory.create("2", "There is PySpark also."),
                RowFactory.create("0", "Spark modules."),
                RowFactory.create("1", "Java interfaces."),
                RowFactory.create("1", "Spring boot Java."),
                RowFactory.create("2", "Something PySpark."),
                RowFactory.create("1", "Something Java.")
        );

        StructType schema = new StructType(new StructField[]{
                createStructField("Category", StringType, false),
                createStructField("Sentence", StringType, false)
        });

        return spark.createDataFrame(data, schema);
    }

    public static double dotProduct(Double[] vecA, Double[] vecB) {
        return IntStream.range(0, vecA.length)
                .parallel()
                .mapToDouble( id -> vecA[id] * vecB[id])
                .reduce(0.0, Double::sum);
    }

}
