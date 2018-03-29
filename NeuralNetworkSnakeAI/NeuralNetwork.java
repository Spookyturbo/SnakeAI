import processing.core.PApplet;

public class NeuralNetwork {

  MapInput activationFunction = new MapInput() {
    public float change(float x) {
      float n = (float)(1/( 1 + Math.pow(Math.E, (-1*x))));
      return n;
    }
  };

  MapInput derivativeFunction = new MapInput() {
    public float change(float x) {
      return (x * (1 - x));
    }
  };

  MapInput averageFunction = new MapInput() {
    public float change(float x) {
      return x / 2000;
    }
  };

  int numberOfInput;
  int numberOfHidden;
  int numberOfOutput;

  float learningRate = 0.1f;

  public Matrix inputWeights; //weights from input to hidden
  public Matrix hiddenWeights; //weights from hidden to output

  public Matrix hiddenBias; //Bias on the hidden (From the input to the hidden)
  public Matrix outputBias; //Bias on the output (From the hidden to the output)

  public NeuralNetwork(int nI, int nH, int nO) {
    numberOfInput = nI;
    numberOfHidden = nH;
    numberOfOutput = nO;

    inputWeights = new Matrix(numberOfHidden, numberOfInput); //Each col is a new input, every input has a value for every hidden (The rows)
    hiddenWeights = new Matrix(numberOfOutput, numberOfHidden); //Same reasoning as above

    inputWeights.randomize();
    hiddenWeights.randomize();

    hiddenBias = new Matrix(numberOfHidden, 1); //One bias per node
    outputBias = new Matrix(numberOfOutput, 1); //One bias per node

    hiddenBias.randomize(); //Remember the bias is just your constant
    outputBias.randomize(); //without a variable in a function
  }

  public float[] feedForward(float[] inputs) { //calculates a guess based on the current weights
    Matrix inputValues = Matrix.vectorFromArray(inputs);

    //Applying the weights to the values
    Matrix hiddenValues = Matrix.multiply(inputWeights, inputValues, false); //Apply the weights to the values  false means not elementwise
    hiddenValues = Matrix.add(hiddenValues, hiddenBias);
    hiddenValues.map(activationFunction); //apply the activation function to the output

    //Applying the weights to the values
    Matrix outputValues = Matrix.multiply(hiddenWeights, hiddenValues, false);
    outputValues = Matrix.add(outputValues, outputBias);
    outputValues.map(activationFunction);

    float[] outputs = new float[outputValues.rows]; //Think about creating a Matrix.toArray()

    for (int i = 0; i < outputValues.rows; i++) {
      outputs[i] = outputValues.data[i][0];
    }

    return outputs;
  }

  public void train(float[] inputs, float[] answers) {

    Matrix inputValues = Matrix.vectorFromArray(inputs);
    Matrix correctOutputs = Matrix.vectorFromArray(answers);
    //Applying the weights to the values
    Matrix hiddenValues = Matrix.multiply(inputWeights, inputValues, false); //Apply the weights to the values  false means not elementwise
    inputValues.print();
    hiddenValues = Matrix.add(hiddenValues, hiddenBias);
    
    hiddenValues.map(activationFunction); //apply the activation function to the output

    //Applying the weights to the values
    Matrix outputValues = Matrix.multiply(hiddenWeights, hiddenValues, false);
    outputValues = Matrix.add(outputValues, outputBias);
    
    outputValues.map(activationFunction);

    Matrix error = Matrix.subtract(correctOutputs, outputValues);
    Matrix transposedHiddenWeights = Matrix.transpose(hiddenWeights);
    Matrix hiddenError = Matrix.multiply(transposedHiddenWeights, error, false);

    //calculate the gradient for the error
    PApplet.println("MappingOutputDerivative");
    outputValues.map(derivativeFunction);

    Matrix gradientHidden = Matrix.multiply(error, outputValues, true); //This IS element wise
    //apply the gradient to get the deltaWeights
    gradientHidden = Matrix.multiply(gradientHidden, learningRate);
    Matrix deltaHiddenWeights = Matrix.multiply(gradientHidden, Matrix.transpose(hiddenValues), false);

    //Apply the change to the weights
    hiddenWeights = Matrix.add(hiddenWeights, deltaHiddenWeights);
    outputBias = Matrix.add(outputBias, gradientHidden);
    PApplet.println("MappingHiddenDerivative");
    hiddenValues.map(derivativeFunction);

    Matrix gradientInput = Matrix.multiply(hiddenError, hiddenValues, true);
    gradientInput = Matrix.multiply(gradientInput, learningRate);
    Matrix deltaInputWeights = Matrix.multiply(gradientInput, Matrix.transpose(inputValues), false);

    inputWeights = Matrix.add(inputWeights, deltaInputWeights);
    hiddenBias = Matrix.add(hiddenBias, gradientInput);
  }

  public void train(float[][] inputs, float[][] answers) {
    Matrix totalError = new Matrix(answers[0].length, 1);
    Matrix totalOutputValues = new Matrix(numberOfOutput, 1);
    Matrix totalHiddenValues = new Matrix(numberOfHidden, 1);
    Matrix totalInputValues = new Matrix(numberOfInput, 1);

    for (int i = 0; i < inputs.length; i++) {
      Matrix inputValues = Matrix.vectorFromArray(inputs[i]);
      totalInputValues = Matrix.add(totalInputValues, inputValues);
      Matrix correctOutputs = Matrix.vectorFromArray(answers[i]);

      //Applying the weights to the values
      Matrix hiddenValues = Matrix.multiply(inputWeights, inputValues, false); //Apply the weights to the values  false means not elementwise
      hiddenValues.map(activationFunction); //apply the activation function to the output
      totalHiddenValues = Matrix.add(totalHiddenValues, hiddenValues);
      //Applying the weights to the values
      Matrix outputValues = Matrix.multiply(hiddenWeights, hiddenValues, false);
      outputValues.map(activationFunction);
      totalOutputValues = Matrix.add(totalOutputValues, outputValues);

      Matrix error = Matrix.subtract(correctOutputs, outputValues);
      totalError = Matrix.add(totalError, error);
    }

    totalOutputValues.map(averageFunction);
    totalHiddenValues.map(averageFunction);
    totalInputValues.map(averageFunction);

    Matrix transposedHiddenWeights = Matrix.transpose(hiddenWeights);
    Matrix hiddenError = Matrix.multiply(transposedHiddenWeights, totalError, false);

    //calculate the gradient for the error
    totalOutputValues.map(derivativeFunction);

    Matrix gradientHidden = Matrix.multiply(totalError, totalOutputValues, true); //This IS element wise
    //apply the gradient to get the deltaWeights
    Matrix deltaHiddenWeights = Matrix.multiply(gradientHidden, Matrix.transpose(totalHiddenValues), false);
    deltaHiddenWeights = Matrix.multiply(deltaHiddenWeights, learningRate);

    //Apply the change to the weights
    hiddenWeights = Matrix.add(hiddenWeights, deltaHiddenWeights);

    totalHiddenValues.map(derivativeFunction);

    Matrix gradientInput = Matrix.multiply(hiddenError, totalHiddenValues, true);

    Matrix deltaInputWeights = Matrix.multiply(gradientInput, Matrix.transpose(totalInputValues), false);
    deltaInputWeights = Matrix.multiply(deltaInputWeights, learningRate);

    inputWeights = Matrix.add(inputWeights, deltaInputWeights);
  }
}