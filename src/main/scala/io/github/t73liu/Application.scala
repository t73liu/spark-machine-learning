package io.github.t73liu

import org.apache.spark.sql.SparkSession

object Application extends App {
  val spark = SparkSession.builder()
    .appName("Spark Machine Learning")
    .master("local[4]")
    .getOrCreate()

  val df = spark.read.option("header", "true").csv("src/test/resources/USDT_XRP.300.poloniex.csv")

  df.show(50)
}
