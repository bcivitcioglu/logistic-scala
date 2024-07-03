package plotter

import breeze.linalg._
import breeze.plot._
import java.awt.Color
import ai.djl.ndarray.NDArray

object Plotter {

  /** Visualizes the labeled training and test data, and the decision boundary
    * determined by the logistic regression model.
    *
    * @param X
    *   The feature matrix for the training data.
    * @param y
    *   The label vector for the training data.
    * @param XTest
    *   The feature matrix for the test data.
    * @param yTest
    *   The label vector for the test data.
    * @param w
    *   The weight vector of the logistic regression model.
    * @param b
    *   The bias term of the logistic regression model.
    */
  def visualizeDataWithBoundary(
      X: NDArray,
      y: NDArray,
      XTest: NDArray,
      yTest: NDArray,
      w: NDArray,
      b: NDArray
  ): Unit = {
    require(
      X.getShape.dimension() == 2 && X.getShape.get(1) == 2,
      "X should have 2 features"
    )
    require(
      y.getShape.dimension() == 2 && y.getShape.get(1) == 1,
      "y should be a column vector"
    )

    val x1 = X.get(":, 0").toFloatArray.map(_.toFloat)
    val x2 = X.get(":, 1").toFloatArray.map(_.toFloat)
    val labels = y.toFloatArray.map(_.toInt)
    val x1Test = XTest.get(":, 0").toFloatArray.map(_.toFloat)
    val x2Test = XTest.get(":, 1").toFloatArray.map(_.toFloat)
    val labelsTest = yTest.toFloatArray.map(_.toInt)
    val weights = w.toFloatArray.map(_.toFloat)
    val bias = b.getFloat().toFloat

    plotLabeledDataWithBoundary(
      x1,
      x2,
      labels,
      x1Test,
      x2Test,
      labelsTest,
      weights,
      bias
    )
  }

  private def plotLabeledDataWithBoundary(
      x1: Array[Float],
      x2: Array[Float],
      labels: Array[Int],
      x1Test: Array[Float],
      x2Test: Array[Float],
      labelsTest: Array[Int],
      weights: Array[Float],
      bias: Float
  ): Unit = {
    // Plot training data points
    val class0X1 = x1.zip(labels).filter(_._2 == 0).map(_._1)
    val class0X2 = x2.zip(labels).filter(_._2 == 0).map(_._1)
    val class1X1 = x1.zip(labels).filter(_._2 == 1).map(_._1)
    val class1X2 = x2.zip(labels).filter(_._2 == 1).map(_._1)

    // Plot test data points
    val class0X1Test = x1Test.zip(labelsTest).filter(_._2 == 0).map(_._1)
    val class0X2Test = x2Test.zip(labelsTest).filter(_._2 == 0).map(_._1)
    val class1X1Test = x1Test.zip(labelsTest).filter(_._2 == 1).map(_._1)
    val class1X2Test = x2Test.zip(labelsTest).filter(_._2 == 1).map(_._1)

    // Plot decision boundary
    val xRange =
      linspace(math.min(x1.min, x1Test.min), math.max(x1.max, x1Test.max), 100)
    val yBoundary = xRange.map(x => (-weights(0) * x - bias) / weights(1))

    val f = Figure()
    val p = f.subplot(0)
    p += scatter(
      class0X1,
      class0X2,
      _ => 0.5,
      _ => Color.BLUE,
      name = "Train Class 0"
    )
    p += scatter(
      class1X1,
      class1X2,
      _ => 0.5,
      _ => Color.RED,
      name = "Train Class 1"
    )
    p += scatter(
      class0X1Test,
      class0X2Test,
      _ => 1,
      _ => Color.CYAN,
      name = "Test Class 0"
    )
    p += scatter(
      class1X1Test,
      class1X2Test,
      _ => 1,
      _ => Color.MAGENTA,
      name = "Test Class 1"
    )
    p += plot(xRange, yBoundary, name = "Decision Boundary")
    p.xlabel = "X1"
    p.ylabel = "X2"
    p.title = "Labeled Data (Training and Test)"
    f.saveas("imgs/decision_boundary.png") // Save the plot as PNG
  }

  /** Plots the loss history per epoch for the logistic regression model.
    * lossHistory: The loss history array.
    */
  def plotLoss(lossHistory: Array[Float]): Unit = {
    val epochs = (1 to lossHistory.length).toArray
    plotLossPerEpoch(epochs, lossHistory.map(_.toFloat))
  }

  private def plotLossPerEpoch(
      epochs: Array[Int],
      losses: Array[Float]
  ): Unit = {
    val f = Figure()
    val p = f.subplot(0)
    p += plot(DenseVector(epochs.map(_.toFloat)), DenseVector(losses))
    p.xlabel = "Epoch"
    p.ylabel = "Loss"
    p.title = "Loss per Epoch"
    f.saveas("imgs/loss.png") // Save the plot as PNG
  }
}
