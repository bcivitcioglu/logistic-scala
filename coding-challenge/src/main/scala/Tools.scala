package tools

import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDArray, NDManager}

/** Binary cross-entropy loss function.
  */
object LossFunc {

  /** @param y
    *   The true labels (0 or 1).
    * @param yPred
    *   The predicted probabilities.
    * @return
    *   The binary cross-entropy loss.
    */
  def binaryCrossEntropy(y: NDArray, yPred: NDArray): NDArray = {
    y.mul(yPred.log())
      .add(y.neg().add(1).mul(yPred.neg().add(1).log()))
      .neg()
      .mean()
  }
}

/** The hypothesis function. In the logistic regression case it is a linear, and
  * subject to an activation function
  */
object Hypothesis {

  /** Computes the hypothesis function for logistic regression.
    *
    * @param w
    *   The weight vector.
    * @param b
    *   The bias term.
    * @param X
    *   The input feature matrix.
    * @return
    *   The predicted probabilities.
    */
  def hypothesis(w: NDArray, b: NDArray, X: NDArray): NDArray = {
    ActivationFunc.sigmoid(X.matMul(w).add(b))
  }
}

/** Activation functions: sigmoid and tanh
  */
object ActivationFunc {

  /** The sigmoid activation
    * @param x
    *   The input.
    */
  def sigmoid(x: NDArray): NDArray = {
    x.neg().exp().add(1).pow(-1)
  }

  /** The tanh activation
    *
    * @param x
    *   The input.
    */
  def tanh(x: NDArray): NDArray = {
    x.tanh()
  }
}

/** Tools for data preprocessing.
  */
object DataPreprocessor {

  /** Splits the input data matrix into features and labels.
    *
    * @param data
    *   The input data matrix.
    * @return
    *   A tuple containing the feature matrix (X) and the label vector (y).
    */
  def splitFeaturesAndLabels(data: NDArray): (NDArray, NDArray) = {
    val X = data.get(":, :-1")
    val y = data.get(":, -1").reshape(-1, 1)
    (X, y)
  }

  /** min-max scaling on the input features.
    *
    * @param X
    *   The input feature matrix.
    * @return
    *   A tuple containing the scaled feature matrix (scaledX), the minimum
    *   values (min), and the feature ranges (range).
    */
  def minMaxScaling(X: NDArray): (NDArray, NDArray, NDArray) = {
    val manager = X.getManager
    val shape = X.getShape
    val min = X.min(Array(0))
    val max = X.max(Array(0))
    val range = max.sub(min)
    val scaledX = X.sub(min).div(range)
    return (scaledX, min, range)
  }

  /** Reverses the min-max scaling.
    *
    * @param X
    *   The scaled feature matrix.
    * @param min
    *   The minimum values used for scaling.
    * @param range
    *   The feature ranges used for scaling.
    * @return
    *   The original feature matrix.
    */
  def unstandardizeFeatures(
      X: NDArray,
      min: NDArray,
      range: NDArray
  ): NDArray = {
    X.mul(range).add(min)
  }
}

/** The implementation of the gradient descent algorithm for logistic
  * regression. It is a simple implementation, thus the name 'vanilla'
  */
object GradientDescent {

  /** @param X
    *   The feature matrix.
    * @param y
    *   The label vector.
    * @param w
    *   The initial weight vector.
    * @param b
    *   The initial bias term.
    * @param learningRate
    *   The learning rate for gradient descent.
    * @param epochs
    *   The number of training epochs.
    * @param lossHistory
    *   An array to store the loss history.
    * @return
    *   A tuple containing the final weight vector and the bias.
    */
  def vanillaGradientDescent(
      X: NDArray,
      y: NDArray,
      w: NDArray,
      b: NDArray,
      learningRate: Float,
      epochs: Int,
      lossHistory: Array[Float]
  ): (NDArray, NDArray) = {
    var curW = w // current value of weights
    var curB = b // current value of bias
    val numSamples = X.getShape.get(0).toFloat

    for (epoch <- 1 to epochs) {
      val yPred = Hypothesis.hypothesis(curW, curB, X)
      var loss = LossFunc.binaryCrossEntropy(y, yPred)
      lossHistory(epoch - 1) = loss.getFloat()

      // Print out the loss value each 500 epochs
      if (epoch % 500 == 0 || epoch == epochs) {
        println(s"Epoch $epoch, Loss: ${loss.getFloat()}")
      }

      val dw: NDArray = X.transpose().matMul(yPred.sub(y)).div(numSamples)
      val db: NDArray = yPred.sub(y).mean()

      val updateW: NDArray = curW.sub(dw.mul(learningRate))
      val updateB: NDArray = curB.sub(db.mul(learningRate))

      curW = updateW
      curB = updateB
    }

    (curW, curB)
  }
}

/** The accuracy implemented.
  */
object Evaluation {

  /** @param predictions
    *   The predicted labels.
    * @param yTrue
    *   The true labels.
    * @return
    *   The accuracy
    */
  def accuracy(predictions: NDArray, yTrue: NDArray): Float = {
    require(
      predictions.getShape == yTrue.getShape,
      "Predictions and true labels must have the same shape"
    ) // This line is added during debugging, then I thought it can stay

    val predFloat = predictions.toType(DataType.FLOAT32, false)
    val yTrueFloat = yTrue.toType(DataType.FLOAT32, false)

    val correct = predFloat.eq(yTrueFloat).sum().toLongArray()(0)
    val total = predictions.getShape.get(0)

    correct.toFloat / total.toFloat
  }
}

/** Monitoring memory usage.
  */
object MemoryMonitor {

  /** Returns the current memory usage of the JVM.
    *
    * @return
    *   The current memory usage in bytes.
    */
  def getCurrentMemoryUsage(): Long = {
    // System.gc()
    // Thread.sleep(10)

    val runtime = Runtime.getRuntime
    runtime.freeMemory()
  }
}

import scala.io.Source

/** Reading data from CSV files.
  */
object FileReader {

  /** @param filename
    *   The path to the CSV file.
    * @param delimiter
    *   The delimiter used in the CSV file (default is ",").
    * @param manager
    *   The NDManager instance (passed implicitly).
    * @return
    *   The data read from the CSV file as an NDArray.
    */
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
