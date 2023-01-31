package edu.gatech.cse6250.main
import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import org.apache.spark.ml.classification.{ LogisticRegression, RandomForestClassifier }
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

/**
 * @author Anita Rao <arao338@gatech.edu>
 */
object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark

    val train0 = CSVHelper.loadCSVAsTable(spark, "data/train0.csv", "train0")
    val train1 = CSVHelper.loadCSVAsTable(spark, "data/train1.csv", "train1")
    val train2 = CSVHelper.loadCSVAsTable(spark, "data/train2.csv", "train2")
    val train3 = CSVHelper.loadCSVAsTable(spark, "data/train3.csv", "train3")
    val train4 = CSVHelper.loadCSVAsTable(spark, "data/train4.csv", "train4")

    val test0 = CSVHelper.loadCSVAsTable(spark, "data/test0.csv", "test0")
    val test1 = CSVHelper.loadCSVAsTable(spark, "data/test1.csv", "test1")
    val test2 = CSVHelper.loadCSVAsTable(spark, "data/test2.csv", "test2")
    val test3 = CSVHelper.loadCSVAsTable(spark, "data/test3.csv", "test3")
    val test4 = CSVHelper.loadCSVAsTable(spark, "data/test4.csv", "test4")

    val train_data = Array(train0, train1, train2, train3, train4)
    val test_data = Array(test0, test1, test2, test3, test4)

    var accuracy_train_lr = 0.0
    var precision_train_lr = 0.0
    var recall_train_lr = 0.0
    var f1score_train_lr = 0.0
    var accuracy_train_rf = 0.0
    var precision_train_rf = 0.0
    var recall_train_rf = 0.0
    var f1score_train_rf = 0.0

    var accuracy_test_lr = 0.0
    var precision_test_lr = 0.0
    var recall_test_lr = 0.0
    var f1score_test_lr = 0.0
    var accuracy_test_rf = 0.0
    var precision_test_rf = 0.0
    var recall_test_rf = 0.0
    var f1score_test_rf = 0.0

    for (i <- 0 to 4) {
      val castedTrainData = train_data(i).toDF.columns.foldLeft(train_data(i).toDF)((current, c) => current.withColumn(c, col(c).cast("float")))
      val castedTestData = test_data(i).toDF.columns.foldLeft(test_data(i).toDF)((current, c) => current.withColumn(c, col(c).cast("float")))

      val assembler_train = new VectorAssembler()
        .setInputCols(castedTrainData.drop("results").columns)
        .setOutputCol("features")

      val assembler_test = new VectorAssembler()
        .setInputCols(castedTestData.drop("results").columns)
        .setOutputCol("features")

      val featureDf_train = assembler_train.transform(castedTrainData)
      val featureDf_test = assembler_test.transform(castedTestData)

      val trainingData = featureDf_train.withColumnRenamed("results", "label")
      val testData = featureDf_test.withColumnRenamed("results", "label")

      val logisticRegression = new LogisticRegression()
        .setMaxIter(10)
        .setElasticNetParam(0.8)
        .setFamily("multinomial")
      val logisticRegressionModel = logisticRegression.fit(trainingData)

      val rf = new RandomForestClassifier()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setNumTrees(30)
        .setMaxDepth(6)
      val rfModel = rf.fit(trainingData)

      val predictionDf_lr = logisticRegressionModel.transform(trainingData)
      val predictionDf_rf = rfModel.transform(trainingData)

      val evaluator_acc = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val evaluator_prec = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("weightedPrecision")
      val evaluator_recall = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("weightedRecall")
      val evaluator_f1 = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("f1")

      accuracy_train_lr += evaluator_acc.evaluate(predictionDf_lr)
      precision_train_lr += evaluator_prec.evaluate(predictionDf_lr)
      recall_train_lr += evaluator_recall.evaluate(predictionDf_lr)
      f1score_train_lr += evaluator_f1.evaluate(predictionDf_lr)

      accuracy_train_rf += evaluator_acc.evaluate(predictionDf_rf)
      precision_train_rf += evaluator_prec.evaluate(predictionDf_rf)
      recall_train_rf += evaluator_recall.evaluate(predictionDf_rf)
      f1score_train_rf += evaluator_f1.evaluate(predictionDf_rf)

      val predictionDf_test_lr = logisticRegressionModel.transform(testData)
      val predictionDf_test_rf = rfModel.transform(testData)

      val evaluator_acc_test = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val evaluator_prec_test = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("weightedPrecision")
      val evaluator_recall_test = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("weightedRecall")
      val evaluator_f1_test = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("f1")

      accuracy_test_lr += evaluator_acc_test.evaluate(predictionDf_test_lr)
      precision_test_lr += evaluator_prec_test.evaluate(predictionDf_test_lr)
      recall_test_lr += evaluator_recall_test.evaluate(predictionDf_test_lr)
      f1score_test_lr += evaluator_f1_test.evaluate(predictionDf_test_lr)

      accuracy_test_rf += evaluator_acc_test.evaluate(predictionDf_test_rf)
      precision_test_rf += evaluator_prec_test.evaluate(predictionDf_test_rf)
      recall_test_rf += evaluator_recall_test.evaluate(predictionDf_test_rf)
      f1score_test_rf += evaluator_f1_test.evaluate(predictionDf_test_rf)
    }

    println("Logistic Regression Performance")
    println("Accuracy Train: " + accuracy_train_lr / 5.0)
    println("Precision Train: " + precision_train_lr / 5.0)
    println("Recall Train: " + recall_train_lr / 5.0)
    println("F1Score Train: " + f1score_train_lr / 5.0)
    println("")
    println("Accuracy Test: " + accuracy_test_lr / 5.0)
    println("Precision Test: " + precision_test_lr / 5.0)
    println("Recall Test: " + recall_test_lr / 5.0)
    println("F1Score Test: " + f1score_test_lr / 5.0)
    println("")
    println("Random Forest Performance")
    println("Accuracy Train: " + accuracy_train_rf / 5.0)
    println("Precision Train: " + precision_train_rf / 5.0)
    println("Recall Train: " + recall_train_rf / 5.0)
    println("F1Score Train: " + f1score_train_rf / 5.0)
    println("")
    println("Accuracy Test: " + accuracy_test_rf / 5.0)
    println("Precision Test: " + precision_test_rf / 5.0)
    println("Recall Test: " + recall_test_rf / 5.0)
    println("F1Score Test: " + f1score_test_rf / 5.0)

  }

}
