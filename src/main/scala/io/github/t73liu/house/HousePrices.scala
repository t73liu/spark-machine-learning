package io.github.t73liu.house

import org.apache.spark.sql.SparkSession

object HousePrices extends App {
  val spark = SparkSession.builder()
    .appName("Spark Machine Learning")
    .master("local[*]")
    .getOrCreate()

  val originalDF = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("./src/test/resources/house/train.csv")

  originalDF.printSchema()
  originalDF.show()
}
