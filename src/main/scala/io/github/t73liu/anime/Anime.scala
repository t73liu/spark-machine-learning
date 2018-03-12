package io.github.t73liu.anime

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

object Anime extends App {
  val spark = SparkSession.builder()
    .appName("Anime Recommendation")
    .master("local[*]")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val unknownEpisode = udf((episode: String) => if (episode.equalsIgnoreCase("Unknown")) "12" else episode)

  // Null ratings (6.47 average)
  // Null genres (Comedy has highest count 4645)
  // Null type (TV has highest count) and episodes (12 episode average)
  val animeDF = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("./src/test/resources/anime/anime.csv")
    .drop("name")
    .na
    .fill(Map(
      "rating" -> 6.47,
      "genre" -> "Comedy",
      "type" -> "TV"
    ))
    .withColumn("genre", split($"genre", ", "))
    .withColumn("episodes", unknownEpisode($"episodes").cast(IntegerType))

  // Note -1 rating means did not rate
  // Loading unzipped version of file because gzip cannot be split
  val ratingDF = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("./src/test/resources/anime/rating.csv")

  spark.close()
}
