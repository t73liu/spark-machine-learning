package io.github.t73liu.house

import org.apache.spark.sql.SparkSession

object HousePrices extends App {
  val spark = SparkSession.builder()
    .appName("Housing Prices")
    .master("local[*]")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  val originalDF = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("./src/test/resources/house/train.csv")
    .drop("Id")

  originalDF.printSchema()
  originalDF.show()

  // Writes output for Kaggle
  //  val resultDF = spark.read
  //    .option("header", "true")
  //    .option("inferSchema", "true")
  //    .csv("./src/test/resources/house/test.csv")
  //  bestModel.transform(resultDF)
  //    .select("Id", "prediction")
  //    .withColumnRenamed("prediction", "SalePrice")
  //    .coalesce(1)
  //    .write
  //    .mode(SaveMode.Overwrite)
  //    .option("header", "true")
  //    .csv("./src/test/resources/house/result/")
}
