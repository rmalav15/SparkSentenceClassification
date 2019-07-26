package com.spark.sentenceclassifiaction;

import lombok.experimental.Accessors;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.sql.SparkSession;

@Accessors(fluent = true)
public class ClassificationModel {

    private static final String SPARK_APP_NAME = "Sentence Similarity Classification";
    private static final String SPARK_MASTER = "local[*]";

    private SparkSession spark;

    private Pipeline pipeline;

    public ClassificationModel() {
        spark = SparkSession.builder()
                .appName(SPARK_APP_NAME)
                .master(SPARK_MASTER)
                .getOrCreate();
        pipeline = null;
    }

    public Pipeline getPipeline() {

        return pipeline;
    }
}
