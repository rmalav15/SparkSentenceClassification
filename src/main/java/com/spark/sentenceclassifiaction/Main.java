package com.spark.sentenceclassifiaction;

import com.spark.sentenceclassifiaction.util.DataUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

public class Main {

    private static final int SEED = 123;

    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        ClassificationService classificationService = new ClassificationService();

        // Dataset<Row> data = DataUtils.generateDummyData(classificationService.spark());

        Dataset<Row> data = DataUtils.readData(classificationService.spark(),
                "src/main/resources/train.csv");

        Dataset<Row>[] randomSplit = data.randomSplit(new double[]{0.6, 0.4}, SEED);

        Dataset<Row> trainData = randomSplit[0].persist(StorageLevel.MEMORY_AND_DISK());
        Dataset<Row> testData = randomSplit[1].persist(StorageLevel.MEMORY_AND_DISK());

        trainData.show();

        // Execution time can be seen on Spark History Server.
        JavaPairRDD<String, Double> predictionRdd = classificationService.classifyTestData(trainData,
                testData);

        testData = classificationService.trainLabelIndexerModel().transform(testData);
        JavaPairRDD<String, Double> gtRdd = testData.javaRDD()
                .mapToPair(r -> new Tuple2<>(r.getAs("Sentence"), r.getAs("Label")));

        MulticlassMetrics metrics = DataUtils.evaluate(predictionRdd, gtRdd);

        // Confusion matrix
        /*Matrix confusion = metrics.confusionMatrix();
        System.out.println("Confusion matrix: \n" + confusion);*/

        // Overall statistics
        System.out.println("Accuracy = " + metrics.accuracy());

        // Stats by labels
        for (int i = 0; i < metrics.labels().length; i++) {
            System.out.println("-----------------------------------------------------------------");
            System.out.format("Class %f precision = %f\n", metrics.labels()[i],metrics.precision(
                    metrics.labels()[i]));
            System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
                    metrics.labels()[i]));
            System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
                    metrics.labels()[i]));
        }

    }

}
