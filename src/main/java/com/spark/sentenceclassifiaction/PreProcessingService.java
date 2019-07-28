package com.spark.sentenceclassifiaction;

import lombok.Getter;
import lombok.experimental.Accessors;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.SparkSession;

@Accessors(fluent = true)
public class PreProcessingService {

    private static final int EMB_SIZE = 16;

    private SparkSession spark;

    @Getter
    private Pipeline preProcessingPipeline;

    public PreProcessingService(SparkSession sparkSession) {
        spark = sparkSession;

        // Not a good tokenizer, will fail in cases of ",", ".", "?" etc.
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("Sentence")
                .setOutputCol("RawWords")
                .setPattern("\\W");

        /*Tokenizer tokenizer = new Tokenizer()
                .setInputCol("Sentence")
                .setOutputCol("RawWords");*/

        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol("RawWords")
                .setOutputCol("FilteredWords");

        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("FilteredWords")
                .setOutputCol("Embeddings")
                .setVectorSize(EMB_SIZE)
                .setMinCount(0);

        Normalizer l2Normalizer = new Normalizer()
                .setInputCol("Embeddings")
                .setOutputCol("NormEmbeddings")
                .setP(2.0);

        preProcessingPipeline = new Pipeline()
        .setStages(new PipelineStage[]{tokenizer, stopWordsRemover, word2Vec, l2Normalizer});
    }
}
