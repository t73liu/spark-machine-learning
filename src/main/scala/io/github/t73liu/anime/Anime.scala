package io.github.t73liu.anime

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

import scala.util.Random

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
    .filter($"rating".notEqual(-1))

  val model = new ALS()
    .setSeed(Random.nextLong())
    .setImplicitPrefs(true)
    .setRank(5)
    .setMaxIter(5)
    .setRegParam(0.01)
    .setUserCol("user_id")
    .setItemCol("anime_id")
    .setRatingCol("rating")
    .setPredictionCol("prediction")

  val personalDF = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("./src/test/resources/anime/personal_rating.csv")
  personalDF.cache()

  val allRatingDF = ratingDF.union(personalDF)

  val trainedModel = model.fit(allRatingDF)

  val user = personalDF.select("user_id").distinct()

  val recommendationDF = trainedModel.recommendForUserSubset(user, 20)
    .select(explode($"recommendations"))
    .sort($"col.rating".desc)
    .select($"col.anime_id")
  val recommendedAnimeDF = recommendationDF
    .join(animeDF, "anime_id")
  personalDF.unpersist()

  recommendedAnimeDF.show()

  // Evaluate Model
  //  evaluateModel()

  spark.close()

  def evaluateModel(): Unit = {
    val Array(trainDF, testDF) = ratingDF.randomSplit(Array(0.85, 0.15))
    trainDF.cache()
    testDF.cache()

    val recommendEstimator = model.fit(trainDF)
    trainDF.unpersist()
    val predictionsDF = recommendEstimator.transform(testDF)
      .na
      .drop(Array("prediction"))
    testDF.unpersist()

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    println(s"Root Mean Squared Error: ${evaluator.evaluate(predictionsDF)}")
  }
}
