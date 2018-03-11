package io.github.t73liu.titanic

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

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
    .csv("./src/test/resources/kaggle-titanic/train.csv")
  originalDF.cache()

  // Determined the fill values used
  println("Averages to use when defaulting null")
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
    .drop("Name", "Embarked", "Ticket")
    .withColumn("Cabin", booleanToInt($"Cabin".isNotNull))
    .withColumn("Sex", booleanToInt($"Sex".eqNullSafe("male")))

  // 70% train, 15% validation, 15% test
  val Array(trainDF, testDF) = cleanedDF.randomSplit(Array(0.85, 0.15))
  trainDF.cache()
  testDF.cache()
  originalDF.unpersist()

  val inputColumns = trainDF.columns.filter(col => col != "Survived")
  println("Features Selected: " + inputColumns.mkString(", "))
  val features = new VectorAssembler().setInputCols(inputColumns).setOutputCol("features")
  val numericalDF = features.transform(trainDF)

  val classifier = new DecisionTreeClassifier()
    .setSeed(Random.nextLong())
    .setLabelCol("Survived")
    .setFeaturesCol("features")
    .setPredictionCol("prediction")

  val model = classifier.fit(numericalDF)

  println("Feature importance in prediction")
  model.featureImportances.toArray.zip(inputColumns).sorted.reverse.foreach(println)

  trainDF.unpersist()
  testDF.unpersist()
}
