// Importing the packages used
// For file reading
import scala.io.Source
import tools.FileReader
// For monitoring memory
import tools.MemoryMonitor
// For logistic regression
import ai.djl.ndarray.{NDArray, NDManager}
import tools.DataPreprocessor
import logisticRegression.LogisticRegression
import tools.Evaluation
// For plotting (works only for 2D)
import plotter.Plotter

object Main extends App {

  val memoryUsage = scala.collection.mutable.ArrayBuffer.empty[Long]

  def recordMemoryUsage(): Unit = {
    memoryUsage += MemoryMonitor.getCurrentMemoryUsage()
  }

  // Beginning by setting the memory usage
  // In order to calculate relative memory usage
  recordMemoryUsage()

  // Create NDManager and declare the filenames with data
  val manager = NDManager.newBaseManager()
  val filename = "data/logistic_regression_2d_train.txt"
  val testFilename = "logistic_regression_2d_test.txt"

  try {

    // System.gc()
    // Thread.sleep(20)
    println(
      s"\nTotal Memory: ${Runtime.getRuntime.totalMemory().toDouble / 1000000}"
    )

    // Implicit manager is used to avoid errors
    // when putting file reading inside Tools.scala
    implicit val implicitManager: NDManager = manager

    // Data Loading
    recordMemoryUsage()
    val dataTrain = FileReader.readCsvToNDArray(filename)
    val dataTest = FileReader.readCsvToNDArray(testFilename)
    recordMemoryUsage()

    // Data Preprocessing: We use separate variable pattern
    // As recomennded by the scala warnings
    val preprocessedDataTrain =
      DataPreprocessor.splitFeaturesAndLabels(dataTrain)
    val XData = preprocessedDataTrain._1
    val y = preprocessedDataTrain._2
    val scaledX = DataPreprocessor.minMaxScaling(XData)
    val X = scaledX._1
    val min = scaledX._2
    val range = scaledX._3
    val preprocessedDataTest = DataPreprocessor.splitFeaturesAndLabels(dataTest)
    val XTestData = preprocessedDataTest._1
    val yTest = preprocessedDataTest._2
    val XTest = XTestData.sub(min).div(range)
    recordMemoryUsage()

    // Model Training
    val numFeatures = XData.getShape.get(1)
    val learningRate: Float = if (numFeatures == 2) 0.01 else 0.05
    val epochs = 10000
    val (weights, bias, lossHistory) =
      LogisticRegression.train(X, y, learningRate, epochs)
    recordMemoryUsage()

    // Prediction
    val predictions = LogisticRegression.predict(XTest, weights, bias)
    recordMemoryUsage()

    // Evaluation
    val accuracy = Evaluation.accuracy(predictions, yTest)
    // Visualization
    Plotter.plotLoss(lossHistory)

    // Reversing the feature scaling
    val originalWeights: NDArray = weights.div(range.reshape(-1, 1))
    val originalBias = bias.sub(min.matMul(originalWeights))
    if (numFeatures == 2) {
      Plotter.visualizeDataWithBoundary(
        XData,
        y,
        XTestData,
        yTest,
        originalWeights,
        originalBias
      )
    } else {
      println(
        s"\nData visualization is only available for 2D data. Current data has $numFeatures dimensions."
      )
    }
    recordMemoryUsage()

    println(s"\nFinal weights: $originalWeights")
    println(s"Final bias: $originalBias")
    println(f"\nTest accuracy: ${accuracy * 100}%.2f%%")
  } finally {
    // Close the NDManager
    manager.close()
    // Close the files
    Source.fromFile(filename).close()
    Source.fromFile(testFilename).close()
  }

  recordMemoryUsage()
  // val initialMemory = memoryUsage.head
  // println("\nRelative Memory Usage (relative to initial):")
  // memoryUsage.zipWithIndex.foreach { case (usage, index) =>
  //  val relativeDiff = usage - initialMemory
  //  println(f"Step $index: ${relativeDiff / 1000000}%.2f MB")
  // }
  // Print memory usage
  println(
    s"\nTotal Memory: ${Runtime.getRuntime.totalMemory().toDouble / 1000000}"
  )
  println("\nFree Memory:")
  memoryUsage.zipWithIndex.foreach { case (freeMemory, index) =>
    println(f"Step $index: ${freeMemory.toDouble / 1000000}%.2f MB")
  }

}
