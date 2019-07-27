package com.spark.sentenceclassifiaction;

import com.spark.sentenceclassifiaction.util.DataUtils;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class Main {

    private static final int SEED = 123;

    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        ClassificationService classificationService = new ClassificationService();

        Dataset<Row> dummyData = DataUtils.generateDummyData(classificationService.spark());

        Dataset<Row>[] randomSplit = dummyData.randomSplit(new double[]  {0.8, 0.2}, SEED);

        classificationService.modelProcessing(randomSplit[0], randomSplit[1]);
    }

}
