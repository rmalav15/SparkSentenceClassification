package com.spark.sentenceclassifiaction;

import com.spark.sentenceclassifiaction.util.DataUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

public class Main {

    private static final int SEED = 123;

    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        ClassificationService classificationService = new ClassificationService();

        Dataset<Row> dummyData = DataUtils.generateDummyData(classificationService.spark());

        Dataset<Row>[] randomSplit = dummyData.randomSplit(new double[]{0.8, 0.2}, SEED);

        Dataset<Row> trainData = randomSplit[0].persist(StorageLevel.MEMORY_AND_DISK());
        Dataset<Row> testData = randomSplit[1].persist(StorageLevel.MEMORY_AND_DISK());

        JavaPairRDD<String, Double> predictionRdd = classificationService.modelProcessing(trainData,
                testData);

        testData = classificationService.trainLabelIndexerModel().transform(testData);

        JavaPairRDD<String, Double> gtRdd = testData.javaRDD()
                .mapToPair(r -> new Tuple2<>(r.getAs("Sentence"), r.getAs("Label")));

        MulticlassMetrics metrics = DataUtils.evaluate(predictionRdd, gtRdd);

        // Confusion matrix
        Matrix confusion = metrics.confusionMatrix();
        System.out.println("Confusion matrix: \n" + confusion);

        // Overall statistics
        System.out.println("Accuracy = " + metrics.accuracy());
    }

}
