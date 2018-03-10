package io.github.t73liu.digit

import org.apache.spark.sql.SparkSession

object DigitRecognizer extends App {
  val spark = SparkSession.builder()
    .appName("Spark Machine Learning")
    .master("local[4]")
    .getOrCreate()

  spark.stop()
}
