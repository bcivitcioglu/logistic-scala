// Importing the packages used
import scala.io.Source
import ai.djl.ndarray.{NDArray, NDManager}
import ai.djl.ndarray.types.{DataType, Shape}
// import ai.djl.metric.Accuracy

object Main extends App {
  // Create NDManager and declare the filenames with data
  implicit val manager: NDManager = NDManager.newBaseManager()

  // Files that is read for demonstration
  val filename = "data/logistic_regression_2d_train.txt"
  val testFilename = "data/logistic_regression_2d_test.txt"

  try {
    // Data Loading
    val dataTrain = LogisticRegression.readCsvToNDArray(filename)
    val dataTest = LogisticRegression.readCsvToNDArray(testFilename)

    // Data Preprocessing: separate features and labels, and feature scale
    val xData = dataTrain.get(":, :-1")
    val y = dataTrain.get(":, -1").reshape(-1, 1)

    val (x, min, range) = LogisticRegression.minMaxScaling(xData)

    val xTestData = dataTest.get(":, :-1")
    val yTest = dataTest.get(":, -1").reshape(-1, 1)

    val xTest = xTestData.sub(min).div(range)

    // Model Training
    val numFeatures = xData.getShape.get(1)
    val learningRate: Float = 0.02f // Changed to Float
    val epochs = 10000
    val (weights, bias, lossHistory) =
      LogisticRegression.train(x, y, learningRate, epochs)

    // Prediction
    val (probabilities, predictions) =
      LogisticRegression.predict(xTest, weights, bias)

    // Evaluation
    // val accuracy = new Accuracy()
    // val accuracyValue = accuracy.update(predictions, yTest)

    // Reversing the feature scaling
    val originalWeights: NDArray = weights.div(range.reshape(-1, 1))
    val originalBias = bias.sub(min.matMul(originalWeights))

    println(s"\nFinal weights: ${originalWeights}")
    println(s"Final bias: ${originalBias}")
    // println(f"\nTest accuracy: ${accuracyValue * 100}%.2f%%")
  } finally {
    // Close the NDManager
    manager.close()
    // Close the files
    Source.fromFile(filename).close()
    Source.fromFile(testFilename).close()
  }
}

object LogisticRegression {
  def train(
      x: NDArray,
      y: NDArray,
      learningRate: Float,
      epochs: Int
  ): (NDArray, NDArray, Array[Float]) = {
    val manager = x.getManager
    val numFeatures = x.getShape.get(1)
    // Initialize the weights and bias
    var w =
      manager.randomNormal(0f, 1f, new Shape(numFeatures, 1), DataType.FLOAT32)
    var b = manager.zeros(new Shape(1, 1), DataType.FLOAT32)

    // Array to store loss history
    val lossHistory = new Array[Float](epochs)

    // Perform gradient descent
    val (finalW, finalB) = vanillaGradientDescent(
      x,
      y,
      w,
      b,
      learningRate,
      epochs,
      lossHistory
    )

    (finalW, finalB, lossHistory)
  }

  private def vanillaGradientDescent(
      x: NDArray,
      y: NDArray,
      w: NDArray,
      b: NDArray,
      learningRate: Float,
      epochs: Int,
      lossHistory: Array[Float]
  ): (NDArray, NDArray) = {
    var curW = w // current value of weights
    var curB = b // current value of bias
    val numSamples = x.getShape.get(0).toFloat

    for (epoch <- 1 to epochs) {
      val yPred = hypothesis(curW, curB, x)
      var loss = binaryCrossEntropy(y, yPred)
      lossHistory(epoch - 1) = loss.getFloat()

      // Print out the loss value each 500 epochs
      if (epoch % 500 == 0 || epoch == epochs) {
        println(s"Epoch $epoch, Loss: ${loss.getFloat()}")
      }

      val dw: NDArray = x.transpose().matMul(yPred.sub(y)).div(numSamples)
      val db: NDArray = yPred.sub(y).mean()

      val updateW: NDArray = curW.sub(dw.mul(learningRate))
      val updateB: NDArray = curB.sub(db.mul(learningRate))

      curW = updateW
      curB = updateB
    }

    (curW, curB)
  }

  def predict(
      x: NDArray,
      w: NDArray,
      b: NDArray,
      threshold: Float = 0.5f
  ): (NDArray, NDArray) = {
    val probabilities = hypothesis(w, b, x)
    val predictions =
      probabilities.gte(threshold).toType(DataType.FLOAT32, false)
    (probabilities, predictions)
  }

  private def hypothesis(w: NDArray, b: NDArray, x: NDArray): NDArray = {
    sigmoid(x.matMul(w).add(b))
  }

  private def binaryCrossEntropy(y: NDArray, yPred: NDArray): NDArray = {
    y.mul(yPred.log())
      .add(y.neg().add(1).mul(yPred.neg().add(1).log()))
      .neg()
      .mean()
  }

  private def sigmoid(x: NDArray): NDArray = {
    x.neg().exp().add(1).pow(-1)
  }

  def minMaxScaling(x: NDArray): (NDArray, NDArray, NDArray) = {
    val manager = x.getManager
    val shape = x.getShape
    val min = x.min(Array(0))
    val max = x.max(Array(0))
    val range = max.sub(min)
    val scaledX = x.sub(min).div(range)
    (scaledX, min, range)
  }

  def readCsvToNDArray(filename: String, delimiter: String = ",")(implicit
      manager: NDManager
  ): NDArray = {
    try {
      val lines = Source.fromFile(filename).getLines().toArray
      val data = lines.map { line =>
        line.split(delimiter).map(_.trim).map(_.toFloat)
      }
      manager.create(data)
    } catch {
      case e: Exception =>
        println(s"Error reading file: ${e.getMessage}")
        throw e
    }
  }
}
