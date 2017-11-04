package io.github.t73liu

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.FunSuite

class ApplicationTest extends FunSuite with DataFrameSuiteBase {
  test("sample test") {
    val trainDF = sqlContext.read.option("header", "true").csv("src/test/resources/kaggle-titanic/train.csv")
    assertDataFrameEquals(trainDF, trainDF)
  }
}
