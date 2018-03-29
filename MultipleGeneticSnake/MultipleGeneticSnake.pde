ArrayList<NeuralNetwork> snakeBrains = new ArrayList<NeuralNetwork>();
float[] fitnessScores;
Snake[] snakes;

boolean trainingMode = true;
boolean snakesAlive = true; //False when all the snakes have died

int nInput = 7;
int nHidden = 1000;
int nOutput = 1;

float mutationLearningRate = 0.01f;
float maxPopulationSize = 200;
float mutationRate = 0.01f;

int generation = 1;

NeuralNetwork bestSnake = null; //The fitness and actual network for the overall best snake found so far
float bestSnakeFitness = 0;

boolean gameReady = false;

Snake snake; //This will be the snake used for showing the best performing snake
void setup() {
  fitnessScores = new float[(int)maxPopulationSize];
  snakes = new Snake[(int)maxPopulationSize];
  snake = new Snake(10);
  snake.beginGame();
  for (int i = 0; i < maxPopulationSize; i++) { //Initializes the snakes
    snakes[i] = new Snake(10);
    snakes[i].show = true;
    snakes[i].beginGame();
  }

  size(400, 400);

  initializePopulation(); //Initializes the neuralnetworks
}

void draw() {
  if (trainingMode || !gameReady) { //!gameReady allows the current population to finish before it starts showing the best snake
    if (snakesAlive) {
      testSnakes();
      gameReady = false;
    } else {
      setFitness();
      gameReady = true;
    }
  } else if (gameReady) {
    if (!snake.alive) { //Reset the snake if its died
      snake.alive = true;
      snake.reset();
      background(51);
    }

    float[] options = new float[3];
    for (int i = -1; i < 2; i++) {

      float[] objects = snake.surrounding();
      float[] inputs = new float[7];
      PVector averagePosition = snake.averagePosition();
      inputs[0] = objects[0];
      inputs[1] = objects[1];
      inputs[2] = objects[2];
      inputs[3] = angleToApple(snake);
      inputs[4] = i;
      inputs[5] = averagePosition.x;
      inputs[6] = averagePosition.y;

      options[i + 1] = bestSnake.feedForward(inputs)[0];
    }
    int largest = 0;
    for (int i = 0; i < options.length; i++) {
      if (options[i] > options[largest]) {
        largest = i;
      }
    }

    //Then move after we know (This is done because the snake will generate a new apple as soon as it lands on it, so we need to check before we call move/turn)
    if (largest == 0) {
      snake.alive = snake.turnLeft();
    } else if (largest == 1) {
      snake.alive = snake.moveForward();
    } else if (largest == 2) {
      snake.alive = snake.turnRight();
    }
  }
}

void setFitness() { //Sets the fitness for the current snake
  background(51);
  for (int i = 0; i < snakes.length; i++) {
    fitnessScores[i] = (snakes[i].fitness); 
    snakes[i].reset();
    snakes[i].alive = true;
  }
  //Generate the new population and reset the snake and fitness scores
  createNewPopulation();
  generation++;

  println("Generation: " + generation);
  snakesAlive = true;
}

void initializePopulation() {
  for (int i = 0; i < maxPopulationSize; i++) {
    snakeBrains.add(new NeuralNetwork(nInput, nHidden, nOutput));
  }
}

void createNewPopulation() {
  //Get the total fitness to assist in creating the breeding pool
  float totalFitness = 0;
  float minFitness = 0;
  int bestSnakeIndex = 0;

  //finding the minFitness so that -fitness values will not cause exceptions in the arrays
  for (float fitness : fitnessScores) {
    if (fitness < minFitness) {
      minFitness = fitness;
    }
  }

  //Adding the absolute value of the minfitness to all to normalize all the values between 0 - max
  for (int i = 0; i < fitnessScores.length; i++) {
    fitnessScores[i] = (fitnessScores[i] + abs(minFitness) + 1); //adds 1 so that every element is included atleast one time
  }

  NeuralNetwork[] breedingPool = new NeuralNetwork[6]; //we will only take the best 6 for the breeding pool

  float tmpFitness = 0;
  int best = 0;
  for (int j = 0; j < 6; j++) {
    for (int i = 0; i < fitnessScores.length; i++) {
      if (fitnessScores[i] > fitnessScores[best]) {
        best = i;
      }
    }
    breedingPool[j] = snakeBrains.get(best);
    if (j == 0) {
      tmpFitness = fitnessScores[best];
    }
    fitnessScores[best] = 0f;
  }

  //Use the breedingPool to create the new generation
  ArrayList<NeuralNetwork> newGeneration = new ArrayList<NeuralNetwork>();
  for (int i = 0; i < maxPopulationSize - 2; i++) {
    int snakeOne = floor(random(breedingPool.length));
    int snakeTwo = floor(random(breedingPool.length));
    newGeneration.add(crossProduct(breedingPool[snakeOne], breedingPool[snakeTwo]));
  }

  if (tmpFitness - abs(minFitness) > bestSnakeFitness) {
    bestSnakeFitness = tmpFitness - abs(minFitness);
    bestSnake = breedingPool[0];
  }

  newGeneration.add(breedingPool[0]);
  newGeneration.add(breedingPool[0]);

  //reset the previous list and put in the new one
  snakeBrains.clear();
  snakeBrains = newGeneration;
}

NeuralNetwork crossProduct(NeuralNetwork a, NeuralNetwork b) { //Merges the two networks and applies any mutation
  NeuralNetwork nn = new NeuralNetwork(nInput, nHidden, nOutput);

  //The values that will need to be changed in the network
  Matrix inputWeights = merge(a.inputWeights, b.inputWeights);
  Matrix hiddenWeights = merge(a.hiddenWeights, b.hiddenWeights);
  Matrix hiddenBias = merge(a.hiddenBias, b.hiddenBias);
  Matrix outputBias = merge(a.outputBias, b.outputBias);

  //Assign the new weights to the network
  nn.inputWeights = inputWeights;
  nn.hiddenWeights = hiddenWeights;
  nn.hiddenBias = hiddenBias;
  nn.outputBias = outputBias;

  return nn;
}

Matrix merge(Matrix a, Matrix b) { //randomly chooses from each matrix and also has chance of mutation
  if (a.rows != b.rows || a.cols != b.cols) {
    println("Cannot merge matrices, incompatible dimensions");
    return null;
  }

  Matrix m = new Matrix(a.rows, a.cols);

  float[][] data = new float[a.data.length][a.data[0].length];
  for (int i = 0; i < data.length; i++) { //Goes through all the rows
    for (int j = 0; j < data[0].length; j++) { //Goes through all the cols
      if (random(1) <= mutationRate) {
        data[i][j] += random(-mutationLearningRate, mutationLearningRate);
      } else {
        boolean keepA = (floor(random(2)) == 1) ? true : false; //random(2) returns between 0 - exclusive 2, flooring results in 0 - 1

        data[i][j] = (keepA) ? a.data[i][j] : b.data[i][j];
      }
    }
  }

  m.data = data;
  return m;
}

//From here down is just snake control
void testSnakes() {
  boolean allDead = true;
  for (int j = 0; j < snakes.length; j++) {
    if (snakes[j].alive) {
      allDead = false;
      float[] options = new float[3];
      for (int i = -1; i < 2; i++) {

        float[] objects = snakes[j].surrounding();
        float[] inputs = new float[7];
        PVector averagePosition = snakes[j].averagePosition();
        inputs[0] = objects[0];
        inputs[1] = objects[1];
        inputs[2] = objects[2];
        inputs[3] = angleToApple(snakes[j]);
        inputs[4] = i;
        inputs[5] = averagePosition.x;
        inputs[6] = averagePosition.y;

        options[i + 1] = snakeBrains.get(j).feedForward(inputs)[0];
      }
      int largest = 0;
      for (int i = 0; i < options.length; i++) {
        if (options[i] > options[largest]) {
          largest = i;
        }
      }

      float distanceToApple = getDistance(snakes[j].apple, snakes[j].currentPosition);
      PVector newPosition = new PVector(1, 1);
      //Use the desired output to see if the snake will eat the apple
      if (largest == 0) {
        newPosition = snakes[j].simulateLeft();
      } else if (largest == 1) {
        newPosition = snakes[j].simulateForward();
      } else if (largest == 2) {
        newPosition = snakes[j].simulateRight();
      }

      if (newPosition.x == snakes[j].apple.x && newPosition.y == snakes[j].apple.y) {
        snakes[j].fitness += 20;
      }

      //Then move after we know (This is done because the snake will generate a new apple as soon as it lands on it, so we need to check before we call move/turn)
      if (largest == 0) {
        snakes[j].alive = snakes[j].turnLeft();
      } else if (largest == 1) {
        snakes[j].alive = snakes[j].moveForward();
      } else if (largest == 2) {
        snakes[j].alive = snakes[j].turnRight();
      }

      if (getDistance(snakes[j].apple, snakes[j].currentPosition) < distanceToApple) {
        snakes[j].fitness += 1;
        snakes[j].circle += 1;
      } else {
        snakes[j].fitness -= 1f; //This way forever going in a circle will only hurt it
        snakes[j].circle -= 1.5;
      }

      if (snakes[j].circle < -10) {
        snakes[j].alive = false;
      }
    }
  }

  snakesAlive = !allDead;
}

void keyPressed() {
  if (key == 'a') {
    trainingMode = false;
  } else if (key == 's') {
    trainingMode = true;
  }

  if (key == 'f') {
    snake.show = false;
  } else if (key == 'g') {
    snake.show = true;
  }
}

float angleToApple(Snake snake) {
  PVector snakePosition = snakePosition = snake.currentPosition.copy();
  PVector apple = snake.apple.copy();


  PVector appleToSnakeVector = new PVector(snakePosition.x - apple.x, snakePosition.y - apple.y);
  PVector direction = snake.direction.copy();
  float dot = appleToSnakeVector.dot(direction);
  float angle = (float)Math.toDegrees(Math.acos(dot / appleToSnakeVector.mag()));

  if (direction.x == 0 && direction.y == 1) { //up
    if (snakePosition.x < apple.x) {
      //angle -= 180;
      angle *= -1;
    }
  } else if (direction.x == 1 && direction.y == 0) { //right
    if (snakePosition.y > apple.y) {
      //angle -= 180;
      angle *= -1;
    }
  } else if (direction.x == -1 && direction.y == 0) { //left
    if (snakePosition.y < apple.y) {
      //angle -= 180;
      angle *= -1;
    }
  } else if (direction.x == 0 && direction.y == -1) { //down
    if (snakePosition.x > apple.x) {
      //angle -= 180;
      angle *= -1;
    }
  }

  return  angle / 180;
}

float getDistance(PVector one, PVector two) {
  float a = one.x - two.x;
  float b = one.y - two.y;
  a *= a;
  b *= b;

  return (float)Math.sqrt(a + b);
}