package io.github.t73liu.titanic

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{SaveMode, SparkSession}

import scala.util.Random

object Titanic extends App {
  val spark = SparkSession.builder()
    .appName("Spark Machine Learning")
    .master("local[*]")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val booleanToInt = udf((boolean: Boolean) => if (boolean) 1 else 0)

  val originalDF = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("./src/test/resources/titanic/train.csv")
  originalDF.cache()

  // Determined the fill values used
  println("Averages to use when defaulting invalid values")
  originalDF.na.drop(Array("Sex", "Age", "Pclass", "Fare"))
    .groupBy("Sex")
    .agg(count("Sex"), avg("Age"), avg("Pclass"), avg("Fare"))
    .show()

  val cleanedDF = originalDF
    .na
    .fill(Map(
      "Sex" -> "male",
      "Age" -> 30.7,
      "Pclass" -> 2,
      "Fare" -> 27.27
    ))
    .drop("Name", "Embarked", "Ticket", "PassengerId")
    .withColumn("Cabin", booleanToInt($"Cabin".isNotNull))
    .withColumn("Sex", booleanToInt($"Sex".eqNullSafe("male")))

  // 15% used for testing
  val Array(trainDF, testDF) = cleanedDF.randomSplit(Array(0.85, 0.15))
  trainDF.cache()
  testDF.cache()
  originalDF.unpersist()

  val inputColumns = trainDF.columns.filter(col => col != "Survived")
  println(s"Features Selected: ${inputColumns.mkString(", ")}")

  // Transformer: applies a transformation onto a DataFrame
  val featuresTransformer = new VectorAssembler().setInputCols(inputColumns).setOutputCol("features")

  // Estimator: algorithm with can be fit onto a DataFrame to produce a Transformer
  val decisionTreeEstimator = new DecisionTreeClassifier()
    .setSeed(Random.nextLong())
    .setLabelCol("Survived")
    .setFeaturesCol("features")
    .setPredictionCol("prediction")

  // Pipeline acts as an estimator and is made up of a series of Estimator/Transformer stages
  val pipeline = new Pipeline().setStages(Array(featuresTransformer, decisionTreeEstimator))

  // Estimator parameter combinations to train
  val estimatorParameters = new ParamGridBuilder()
    .addGrid(decisionTreeEstimator.impurity, Seq("entropy", "gini"))
    .addGrid(decisionTreeEstimator.maxDepth, Seq(10, 12, 14, 16))
    .addGrid(decisionTreeEstimator.maxBins, Seq(25, 30))
    .build()

  // Evaluation of model using true positive rate and false positive rate
  val binaryEvaluator = new BinaryClassificationEvaluator()
    .setLabelCol("Survived")
    .setMetricName("areaUnderROC")

  val validator = new CrossValidator()
    .setSeed(Random.nextLong())
    .setEstimator(pipeline)
    .setEvaluator(binaryEvaluator)
    .setEstimatorParamMaps(estimatorParameters)
    .setNumFolds(4)

  val validatorModel = validator.fit(trainDF)

  val bestModel = validatorModel.bestModel
  println(s"Model Selected: ${bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap}")

  val testAccuracy = binaryEvaluator.evaluate(bestModel.transform(testDF))
  println(s"Test Accuracy: $testAccuracy")

  val trainAccuracy = binaryEvaluator.evaluate(bestModel.transform(trainDF))
  println(s"Training Accuracy: $trainAccuracy")

  trainDF.unpersist()
  testDF.unpersist()

  // Writes output for Kaggle
  val resultDF = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("./src/test/resources/titanic/test.csv")
    .na
    .fill(Map(
      "Sex" -> "male",
      "Age" -> 30.7,
      "Pclass" -> 2,
      "Fare" -> 27.27
    ))
    .withColumn("Cabin", booleanToInt($"Cabin".isNotNull))
    .withColumn("Sex", booleanToInt($"Sex".eqNullSafe("male")))

  bestModel.transform(resultDF)
    .select("PassengerId", "prediction")
    .withColumn("prediction", $"prediction".cast(IntegerType))
    .withColumnRenamed("prediction", "Survived")
    .coalesce(1)
    .write
    .mode(SaveMode.Overwrite)
    .option("header", "true")
    .csv("./src/test/resources/titanic/result/")
}
