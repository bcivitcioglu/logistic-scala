package logisticRegression

import ai.djl.ndarray.types.{DataType, Shape}
import ai.djl.ndarray.{NDArray, NDManager}
import tools.LossFunc.binaryCrossEntropy
import tools.Hypothesis.hypothesis
import tools.GradientDescent

object LogisticRegression {

  /** Trains a logistic regression model on the given data.
    *
    * @param X:
    *   The feature matrix.
    * @param y:
    *   The label vector.
    * @param learningRate:
    *   The learning rate for gradient descent.
    * @param epochs:
    *   The number of training epochs.
    * @return
    *   The final weight vector, the final bias term, and the loss history.
    */
  def train(
      X: NDArray,
      y: NDArray,
      learningRate: Float,
      epochs: Int
  ): (NDArray, NDArray, Array[Float]) = {
    val manager = X.getManager
    val numFeatures = X.getShape.get(1)
    // Initialize the weights and bias
    var w =
      manager.randomNormal(0f, 1f, new Shape(numFeatures, 1), DataType.FLOAT32)
    var b = manager.zeros(new Shape(1, 1), DataType.FLOAT32)

    // Array to store loss history
    val lossHistory = new Array[Float](epochs)

    // Perform gradient descent
    val (finalW, finalB) = GradientDescent.vanillaGradientDescent(
      X,
      y,
      w,
      b,
      learningRate,
      epochs,
      lossHistory
    )

    (finalW, finalB, lossHistory)
  }

  /** Computes the probabilities of the input data belonging to the positive
    * class.
    *
    * @param X
    *   The input feature matrix.
    * @param w
    *   The weight vector.
    * @param b
    *   The bias term.
    * @return
    *   The vector of probabilities.
    */
  def predictProbabilities(X: NDArray, w: NDArray, b: NDArray): NDArray = {
    hypothesis(w, b, X)
  }

  /** Predicts the class labels for the input data based on the logistic
    * regression model.
    *
    * @param X
    *   The input feature matrix.
    * @param w
    *   The weight vector.
    * @param b
    *   The bias term.
    * @param threshold
    *   The threshold for classification (default is 0.5).
    * @return
    *   The vector of predicted class labels.
    */
  def predict(
      X: NDArray,
      w: NDArray,
      b: NDArray,
      threshold: Float = 0.5f
  ): NDArray = {
    val probabilities = hypothesis(w, b, X)
    probabilities.gte(threshold).toType(DataType.FLOAT32, false)
  }
}
