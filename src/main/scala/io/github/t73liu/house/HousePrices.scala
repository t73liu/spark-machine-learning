package io.github.t73liu.house

import org.apache.spark.sql.SparkSession

object HousePrices extends App {
  val spark = SparkSession.builder()
    .appName("Spark Machine Learning")
    .master("local[*]")
    .getOrCreate()

  // TODO Work in progress
}
