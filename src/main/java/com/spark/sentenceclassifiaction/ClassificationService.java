package com.spark.sentenceclassifiaction;

import lombok.Getter;
import lombok.experimental.Accessors;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import scala.Tuple2;

@Accessors(fluent = true)
@Getter
public class ClassificationService {

    private static final String SPARK_APP_NAME = "Sentence Similarity Classification";
    private static final String SPARK_MASTER = "local[*]";

    private PreProcessingService preProcessingService;

    private SparkSession spark;

    private SparkContext sc;

    ClassificationService() {
        SparkConf conf = new SparkConf().set("spark.sql.crossJoin.enabled", "true");
        spark = SparkSession.builder()
                .config(conf)
                .appName(SPARK_APP_NAME)
                .master(SPARK_MASTER)
                .getOrCreate();

        /*spark.udf().register("dotProduct",
                (WrappedArray<Double> vecA, WrappedArray<Double> vecB) -> {
                    return DataUtils.dotProduct((Double[]) JavaConversions.seqAsJavaList(vecA).toArray(),
                            (Double[]) JavaConversions.seqAsJavaList(vecB).toArray());
                }, DataTypes.DoubleType);*/

        spark.udf().register("toArray",
                (UDF1<DenseVector, Object>) DenseVector::toArray, DataTypes.createArrayType(DataTypes.DoubleType));

        sc = spark.sparkContext();
        preProcessingService = new PreProcessingService(spark);
    }


    /*
     *  process data ("Sentence", "Category") through data and
     *  return <testSentID, <trainSentLabel, Matched>>
     *  @trainData train data
     *  @testData test data
     */
    JavaPairRDD<String, String> modelProcessing(Dataset<Row> trainData,
                                                Dataset<Row> testData) {

        PipelineModel preProcessingFit = preProcessingService.preProcessingPipeline().fit(trainData);

        trainData = preProcessingFit.transform(trainData);
        testData = preProcessingFit.transform(testData);

        trainData.createOrReplaceTempView("train_data_table");
        testData.createOrReplaceTempView("val_data_table");

        // Filtering cosine value >= 0.7, it may happen that it doesnt find any matching sentences
        Dataset<Row> joinedData = spark.sql("SELECT v.Sentence, " +
                "SUM(1), t.Category FROM train_data_table t " +
                "CROSS JOIN val_data_table v " +
                "WHERE AGGREGATE(ZIP_WITH(toArray(t.NormEmbeddings), toArray(v.NormEmbeddings), (x, y) -> x*y)," +
                "CAST(0 AS DOUBLE), (acc, x) -> acc+x) >= 0.7 " +
                "GROUP BY v.Sentence, t.Category");

        JavaPairRDD<String, String> resultRdd = joinedData.javaRDD()
                .mapToPair(r -> new Tuple2<>(r.getString(0), new Tuple2<>(r.getString(2), r.getLong(1))))
                .reduceByKey((cat1, cat2) -> cat2._2 >= cat1._2 ? cat2 : cat1)
                .mapToPair(t -> new Tuple2<>(t._1, t._2._1));

        /*joinedData.show();
        resultRdd.foreach(v -> System.out.println(v));*/

        return resultRdd;
    }


}
