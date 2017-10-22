package io.github.t73liu

import org.apache.spark.sql.SparkSession

object Application extends App {
  val spark = SparkSession.builder()
    .appName("Spark Machine Learning")
    .master("local[4]")
    .getOrCreate()

  // survival: 0 = No, 1 = Yes
  // pclass: social-economic status (1 = Upper, 2 = Middle, 3 = Lower)
  val trainDF = spark.read.option("header", "true").csv("src/test/resources/kaggle-titanic/train.csv")
  val testDF = spark.read.option("header", "true").csv("src/test/resources/kaggle-titanic/test.csv")

  spark.stop()
}
