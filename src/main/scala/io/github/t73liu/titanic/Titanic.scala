package io.github.t73liu.titanic

import org.apache.spark.sql.SparkSession

object Titanic extends App {
  val spark = SparkSession.builder()
    .appName("Spark Machine Learning")
    .master("local[4]")
    .getOrCreate()

  // survival: 0 = No, 1 = Yes
  // pclass: social-economic status (1 = Upper, 2 = Middle, 3 = Lower)
  val trainDF = spark.read.option("header", "true").csv("./src/test/resources/kaggle-titanic/train.csv")
  trainDF.show()
  val testDF = spark.read.option("header", "true").csv("./src/test/resources/kaggle-titanic/test.csv")
  testDF.show()

  spark.stop()
}
