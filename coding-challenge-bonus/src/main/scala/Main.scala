import scala.io.Source
import scala.util.Random

object Main extends App {

  /** I used the same memory measuring script with the non optimzed case. Even
    * if I doubt its results.
    * @return
    *   Long representing the free memory in bytes
    */
  def getCurrentMemoryUsage(): Long = {
    // System.gc()
    // Thread.sleep(20)

    val runtime = Runtime.getRuntime
    runtime.freeMemory()
  }

  /** Stream data from a file, instead of loading the whole file.
    * @param filename
    *   The name of the file
    * @return
    *   An iterator of float arrays
    */
  def streamData(filename: String): Iterator[Array[Float]] = {
    Source.fromFile(filename).getLines().map { line =>
      line.split(",").map(_.trim.toFloat)
    }
  }

  /** Create mini-batches from the input data.
    * @param data
    *   The dataset
    * @param batchSize
    *   The size of each mini-batch
    * @return
    *   An iterator of mini-batches
    */
  def createMiniBatches(
      data: Array[Array[Float]],
      batchSize: Int
  ): Iterator[Array[Array[Float]]] = {
    data.grouped(batchSize)
  }

  /** Fature scaling min-max normalization.
    * @param data
    *   The input data to be scaled
    * @return
    *   A tuple containing the scaled data, min values, and max values
    */
  def scaleFeatures(
      data: Array[Array[Float]] // This will not work for more than 3D, I doubt
  ): (Array[Array[Float]], Array[Float], Array[Float]) = {
    val numFeatures = data(0).length - 1 // Assuming last column is the label
    val min = Array.fill(numFeatures)(Float.MaxValue)
    val max = Array.fill(numFeatures)(Float.MinValue)

    // Find min and max for each feature
    data.foreach { row =>
      for (i <- 0 until numFeatures) {
        min(i) = math.min(min(i), row(i))
        max(i) = math.max(max(i), row(i))
      }
    }

    // Scale features
    val scaledData = data.map { row =>
      val scaledRow = Array.ofDim[Float](row.length)
      for (i <- 0 until numFeatures) {
        scaledRow(i) =
          if (max(i) > min(i)) (row(i) - min(i)) / (max(i) - min(i)) else 0f
      }
      scaledRow(numFeatures) = row(numFeatures) // Keep the label unchanged
      scaledRow
    }

    (scaledData, min, max)
  }

  /** Logistic Regression.
    * @param numFeatures
    *   The number of input features
    */
  class LogisticRegressionModel(numFeatures: Int) {
    private val weights: Array[Float] =
      Array.fill(numFeatures)(Random.nextFloat() * 0.01f)
    private var bias: Float = 0f

    /** Sigmoid activation function.
      * @param x
      *   Input value
      * @return
      *   Sigmoid of x
      */
    private def sigmoid(x: Float): Float = 1f / (1f + math.exp(-x).toFloat)

    /** Make a prediction for given features.
      * @param features
      *   Input features
      * @return
      *   Predicted probability
      */
    def predict(features: Array[Float]): Float = {
      val z = features.zip(weights).map { case (x, w) => x * w }.sum + bias
      sigmoid(z)
    }

    /** Make a prediction for unscaled features.
      * @param features
      *   Unscaled input features
      * @param min
      *   Min values for scaling
      * @param max
      *   Max values for scaling
      * @return
      *   Predicted probability
      */
    def predictUnscaled(
        features: Array[Float],
        min: Array[Float],
        max: Array[Float]
    ): Float = {
      val scaledFeatures = features.zipWithIndex.map { case (x, i) =>
        if (max(i) > min(i)) (x - min(i)) / (max(i) - min(i)) else 0f
      }
      predict(scaledFeatures)
    }

    /** Gradient Descent.
      * @param features
      *   Input features
      * @param label
      *   True label
      * @param learningRate
      *   Learning rate for gradient descent
      */
    def vanilla_gradient_descent(
        features: Array[Float],
        label: Float,
        learningRate: Float
    ): Unit = {
      val prediction = predict(features)
      val error = prediction - label

      for (i <- weights.indices) {
        weights(i) -= learningRate * error * features(i)
      }
      bias -= learningRate * error
    }

    // Getter methods
    def getWeights: Array[Float] = weights
    def getBias: Float = bias

    /** Get unstandardized weights for interpretability.
      * @param min
      *   Min values used for scaling
      * @param max
      *   Max values used for scaling
      * @return
      *   Unstandardized weights
      */
    def getUnstandardizedWeights(
        min: Array[Float],
        max: Array[Float]
    ): Array[Float] = {
      weights.zipWithIndex.map { case (w, i) => w / (max(i) - min(i)) }
    }

    /** Get unstandardized bias for interpretability.
      * @param min
      *   Min values used for scaling
      * @param max
      *   Max values used for scaling
      * @return
      *   Unstandardized bias
      */
    def getUnstandardizedBias(min: Array[Float], max: Array[Float]): Float = {
      bias - weights.zipWithIndex.map { case (w, i) =>
        w * min(i) / (max(i) - min(i))
      }.sum
    }
  }

  /** Train the model.
    */
  def train(
      model: LogisticRegressionModel,
      data: Array[Array[Float]],
      epochs: Int,
      learningRate: Float,
      batchSize: Int
  ): Unit = {
    for (epoch <- 1 to epochs) {
      var batchLoss = 0f
      var batchCount = 0

      for (batch <- createMiniBatches(data, batchSize)) {
        for (sample <- batch) {
          val features = sample.dropRight(1)
          val label = sample.last

          val prediction = model.predict(features)
          batchLoss = batchLoss + (-label * math.log(
            prediction
          ) - (1 - label) * math.log(1 - prediction)).toFloat

          model.vanilla_gradient_descent(features, label, learningRate)
        }
        batchCount += 1
      }

      val avgLoss = batchLoss / (batchCount * batchSize)
      if (epoch % 500 == 0) {
        println(f"Epoch $epoch%d, Loss: $avgLoss%.4f")
      }
    }
  }

  /** Main execution method.
    */
  def run(): Unit = {
    // System.gc()
    // Thread.sleep(20)

    println(
      s"\nTotal Memory: ${Runtime.getRuntime.totalMemory().toDouble / 1000000}"
    )

    val filename = "data/logistic_regression_3d_train.txt"
    val batchSize = 32
    val epochs = 10000
    val learningRate = 0.001f
    val memoryUsage = scala.collection.mutable.ArrayBuffer.empty[Long]
    memoryUsage += getCurrentMemoryUsage()

    // Load all data
    val data = streamData(filename).toArray

    // Scale all data
    val (scaledData, min, max) = scaleFeatures(data)
    memoryUsage += getCurrentMemoryUsage()

    val numFeatures = scaledData(0).length - 1
    val model = new LogisticRegressionModel(numFeatures)

    // Train the model
    train(model, scaledData, epochs, learningRate, batchSize)
    memoryUsage += getCurrentMemoryUsage()

    // Print final model parameters
    println("\nFinal weights:")
    println(model.getUnstandardizedWeights(min, max).mkString(", "))
    println(
      f"Final bias: ${model.getUnstandardizedBias(min, max)}%.4f"
    )

    // Example of making a prediction on unscaled data
    val sampleUnscaledFeatures = data(0).dropRight(1)
    val prediction = model.predictUnscaled(sampleUnscaledFeatures, min, max)
    memoryUsage += getCurrentMemoryUsage()

    // Print memory usage statistics
    println(
      s"\nTotal Memory: ${Runtime.getRuntime.totalMemory().toDouble / 1000000}"
    )

    println("\nFree Memory Usage:")
    memoryUsage.zipWithIndex.foreach { case (freeMemory, index) =>
      println(f"Step $index: ${freeMemory.toDouble / 1000000}%.2f MB")
    }
  }

  run()
}
