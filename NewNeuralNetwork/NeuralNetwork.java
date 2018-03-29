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
  int[] numberOfHidden; //For knowing amount of nodes per layer
  int numberOfOutput;

  float learningRate = 0.1f;

  public Matrix inputWeights; //weights from input to hidden
  public Matrix[] hiddenWeights; //A matrix for every layer
  public Matrix[] hiddenBias; //Bias on the hidden (From the input to the hidden)
  public Matrix outputBias; //Bias on the output (From the hidden to the output)

  public NeuralNetwork(int nI, int[] nH, int nO) {
    numberOfInput = nI;
    numberOfHidden = nH;
    numberOfOutput = nO;

    //The hidden bias are applied from the input to the first hidden as well as every hidden to other hiddens
    //The last hidden to the output is its own bias
    hiddenWeights = new Matrix[nH.length];
    hiddenBias = new Matrix[hiddenWeights.length];
    inputWeights = new Matrix(numberOfHidden[0], numberOfInput); //Each col is a new input, every input has a value for every hidden (The rows)
    for (int i = 1; i < hiddenWeights.length; i++) {
      hiddenWeights[i - 1] = new Matrix(numberOfHidden[i], numberOfHidden[i - 1]); //Same reasoning as above
    }
    hiddenWeights[hiddenWeights.length - 1] = new Matrix(numberOfOutput, numberOfHidden[numberOfHidden.length - 1]); //Connect the last hidden to the outputs

    //Initialize all of the matrices
    inputWeights.randomize();
    for (Matrix weights : hiddenWeights) {
      weights.randomize();
    }

    //Initialize all biases
    for (int i = 0; i < hiddenWeights.length; i++) {
      hiddenBias[i] = new Matrix(numberOfHidden[i], 1);
      hiddenBias[i].randomize(); //Remember the bias is just your constant
    }
    outputBias = new Matrix(numberOfOutput, 1); //One bias per node
    outputBias.randomize(); //without a variable in a function
  }

  public float[] feedForward(float[] inputs) { //calculates a guess based on the current weights
    Matrix inputValues = Matrix.vectorFromArray(inputs);

    //Applying the weights to the values
    Matrix hiddenValues = Matrix.multiply(inputWeights, inputValues, false); //Apply the weights to the values  false means not elementwise
    hiddenValues = Matrix.add(hiddenValues, hiddenBias[0]); 
    hiddenValues.map(activationFunction); //apply the activation function to the output

    for (int i = 0; i < hiddenWeights.length - 1; i++) {
      hiddenValues = Matrix.multiply(hiddenWeights[i], hiddenValues, false);
      hiddenValues = Matrix.add(hiddenValues, hiddenBias[i + 1]);
      hiddenValues.map(activationFunction);
    }

    //Applying the weights to the values
    Matrix outputValues = Matrix.multiply(hiddenWeights[hiddenWeights.length - 1], hiddenValues, false);
    outputValues = Matrix.add(outputValues, outputBias);
    outputValues.map(activationFunction);

    float[] outputs = new float[outputValues.rows]; //Think about creating a Matrix.toArray()

    for (int i = 0; i < outputValues.rows; i++) {
      outputs[i] = outputValues.data[i][0];
    }

    return outputs;
  }

  public void train(float[] inputs, float[] answers) {
   
    Matrix[] steps = new Matrix[hiddenWeights.length + 2]; //For all the hidden and the output with the initial being the inputValues
    Matrix inputValues = Matrix.vectorFromArray(inputs);
    steps[0] = inputValues;
    Matrix correctOutputs = Matrix.vectorFromArray(answers);
    //Applying the weights to the values
    Matrix inputToHiddenValues = Matrix.multiply(inputWeights, inputValues, false); //Apply the weights to the values  false means not elementwise

    inputToHiddenValues = Matrix.add(inputToHiddenValues, hiddenBias[0]);

    inputToHiddenValues.map(activationFunction); //apply the activation function to the output
    steps[1] = inputToHiddenValues;
    
    for (int i = 0; i < hiddenWeights.length - 1; i++) {
      Matrix hiddenValues = Matrix.multiply(hiddenWeights[i], steps[i + 1], false);
      hiddenValues = Matrix.add(hiddenValues, hiddenBias[i + 1]);
      hiddenValues.map(activationFunction);
      steps[i + 2] = hiddenValues;
      
    }
    
    //Applying the weights to the values
    
    
    Matrix outputValues = Matrix.multiply(hiddenWeights[hiddenWeights.length - 1], steps[steps.length - 2], false); //hiddenWeights.length - 1 == steps.length - 2
    outputValues = Matrix.add(outputValues, outputBias);

    outputValues.map(activationFunction);
    steps[steps.length - 1] = outputValues;
    
    //Now that all the steps are known, start backpropagation
    //Index 0 = Error from outputs to last hidden
    Matrix[] errors = new Matrix[hiddenWeights.length + 1]; //Error for all the hiddenWeights + the inputWeights
    Matrix error = Matrix.subtract(correctOutputs, outputValues);
    errors[0] = error;
    int errorIndex = 1; //For adding to the error array
    int weightIndex = hiddenWeights.length - 1; //For going bcakwards through the hiddenWeight array
    for (int i = steps.length - 2; i >= 1; i--) { //steps.length - 2 because backpropagation requires going backwards and steps.length - 1 is handled in the initial error calc
      //Cant be 0 because the inputs dont have an error
      Matrix transposedHiddenWeights = Matrix.transpose(hiddenWeights[weightIndex]);
      Matrix hiddenError = Matrix.multiply(transposedHiddenWeights, errors[errorIndex - 1], false);
      errors[errorIndex] = hiddenError;
      errorIndex++;
      weightIndex--;
    }
    
    //calculate the gradients for the errors
    errorIndex = 0;
    weightIndex = hiddenWeights.length - 1;
    for (int i = steps.length - 1; i > 1; i--) { //G0 back through every step, calculate gradient, and apply it Do not try this at 0, once at 0, there is no farther to go backwards
      steps[i].map(derivativeFunction);
      
      Matrix gradient = Matrix.multiply(errors[errorIndex], steps[i], true);
      gradient = Matrix.multiply(gradient, learningRate);
      Matrix deltaWeights = Matrix.multiply(gradient, Matrix.transpose(steps[i - 1]), false);
      hiddenWeights[weightIndex] = Matrix.add(hiddenWeights[weightIndex], deltaWeights);
      if (i == steps.length - 1) { //This means the bias belongs to the output not the hidden
        outputBias = Matrix.add(outputBias, gradient);
      } else {
        //Because steps is two longer then hidden, the initial run through is 2 over the size of hiddenBias, so after just i - 1 can be used
        hiddenBias[i - 1] = Matrix.add(hiddenBias[i - 1], gradient);
      }
      errorIndex++;
      weightIndex--;
    }
    //The above adjusts all the hidden, but the inputWeights still need adjusted
    steps[1].map(derivativeFunction);
    Matrix gradient = Matrix.multiply(errors[errors.length - 1], steps[1], true);
    gradient = Matrix.multiply(gradient, learningRate);
    Matrix deltaWeights = Matrix.multiply(gradient, Matrix.transpose(steps[0]), false);
    inputWeights = Matrix.add(inputWeights, deltaWeights);
    hiddenBias[0] = Matrix.add(hiddenBias[0], gradient);
    //outputValues.map(derivativeFunction);

    //Matrix gradientHidden = Matrix.multiply(errors[0], outputValues, true); //This IS element wise
    ////apply the gradient to get the deltaWeights
    //gradientHidden = Matrix.multiply(gradientHidden, learningRate);
    //Matrix deltaHiddenWeights = Matrix.multiply(gradientHidden, Matrix.transpose(hiddenValues), false);

    ////Apply the change to the weights
    //hiddenWeights = Matrix.add(hiddenWeights, deltaHiddenWeights);
    //outputBias = Matrix.add(outputBias, gradientHidden);

    //hiddenValues.map(derivativeFunction);

    //Matrix gradientInput = Matrix.multiply(hiddenError, hiddenValues, true);
    //gradientInput = Matrix.multiply(gradientInput, learningRate);
    //Matrix deltaInputWeights = Matrix.multiply(gradientInput, Matrix.transpose(inputValues), false);

    //inputWeights = Matrix.add(inputWeights, deltaInputWeights);
    //hiddenBias = Matrix.add(hiddenBias, gradientInput);
  }
}