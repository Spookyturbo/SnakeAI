class Snake {

  float timeSinceApple = 0f;
  float lastTime = 0f;
  PVector apple;
  boolean alive = true;
  ArrayList<PVector> snake = new ArrayList<PVector>();
  boolean show = true;
  int snakeSize;
  int snakeLength = 3;
  int score = -1;
  public float fitness = 0;
  public float circle = 0; //Same as the fitness score but subtracts 1.5 for the wrong direction to determine when going in circles forever
  PVector direction = new PVector(1, 0);
  PVector currentPosition;
  PVector dimensions;

  public Snake(int snakeSize) {
    this.snakeSize = snakeSize;
  }

  public void beginGame() {
    background(51);
    dimensions = new PVector(width / snakeSize, height / snakeSize);
    currentPosition = new PVector(0, 0);
    snake.add(currentPosition);
    generateApple();
    if (show) {
      drawSnake();
    }
  }

  public float distanceToApple(PVector currentPosition) {
    float a = apple.x - currentPosition.x;
    float b = apple.y - currentPosition.y;
    a *= a;
    b *= b;

    return (float)Math.sqrt(a + b);
  }

  public void generateApple() {
    score++;
    do {
      apple = new PVector(floor(random(dimensions.x)), floor(random(dimensions.y)));
    } while (snakeContains(apple));

    if (show) {
      noStroke();
      fill(255, 0, 0);
      rect(apple.x * snakeSize, map(apple.y, 0, dimensions.y - 1, dimensions.y - 1, 0) * snakeSize, snakeSize, snakeSize);
    }
    timeSinceApple = 0f;
  }

  public boolean snakeContains(PVector point) {
    for (int i = 0; i < snake.size(); i++) {
      if (snake.get(i).x == point.x && snake.get(i).y == point.y) {
        return true;
      }
    }
    return false;
  }

  public boolean updateSnake() {
    timeSinceApple += millis() - lastTime;
    move();
    lastTime = millis();
    if (show) {
      drawSnake();
    }
    if (dead(currentPosition) || timeSinceApple > 5000f) { //If the snake hasnt eaten an apple for 5 seconds, its going in circles or dumb, so kill it
      //reset();
      return false;
    }
    return true;
  }

  private void reset() {
    timeSinceApple = 0;
    lastTime = millis();
    circle = 0;
    score = -1;
    fitness = 0;
    snakeLength = 3;
    snake.clear();
    currentPosition = new PVector(0, 0);
    snake.add(currentPosition);
    direction = new PVector(1, 0);
    drawSnake();
    generateApple();
  }

  private boolean dead(PVector currentPosition) {
    return currentPosition.x >= dimensions.x || currentPosition.x < 0 || currentPosition.y >= dimensions.y || currentPosition.y < 0 || hitSelf(currentPosition);
  }

  public float[] surrounding() {
    float[] surroundingObjects = new float[3];
    float[] tmp = {direction.x, direction.y};

    Matrix dir = Matrix.multiply(left, Matrix.vectorFromArray(tmp), false);
    PVector leftSpot = new PVector();
    leftSpot.x = dir.data[0][0];
    leftSpot.y = dir.data[1][0];
    leftSpot = currentPosition.copy().add(leftSpot);

    dir = Matrix.multiply(right, Matrix.vectorFromArray(tmp), false);
    PVector rightSpot = new PVector();
    rightSpot.x = dir.data[0][0];
    rightSpot.y = dir.data[1][0];
    rightSpot = currentPosition.copy().add(rightSpot);

    surroundingObjects[0] = (dead(leftSpot)) ? 1 : 0;
    surroundingObjects[1] = (dead(currentPosition.copy().add(direction))) ? 1 : 0;
    surroundingObjects[2] = (dead(rightSpot)) ? 1 : 0;

    return surroundingObjects;
  }

  public float[] surroundingDistance() { //This version returns the distance until it will run into a wall or something that kills it
    float[] surroundingObjects = new float[3];
    float[] tmp = {direction.x, direction.y};

    PVector tmpPosition = simulateLeft(currentPosition, direction);
    surroundingObjects[0] = 0;
    while (!dead(tmpPosition)) {
      surroundingObjects[0] ++;
      tmpPosition = simulateLeft(tmpPosition, direction);
    }

    tmpPosition = simulateForward(currentPosition, direction);
    surroundingObjects[1] = 0;
    while (!dead(tmpPosition)) {
      surroundingObjects[1]++; 
      tmpPosition = simulateForward(tmpPosition, direction);
    }


    tmpPosition = simulateRight(currentPosition, direction);
    surroundingObjects[2] = 0;
    while (!dead(tmpPosition)) {
      surroundingObjects[2]++;
      tmpPosition = simulateRight(tmpPosition, direction);
    }

    for(int i = 0; i < surroundingObjects.length; i++) {
     surroundingObjects[i] /= dimensions.x; 
    }

    return surroundingObjects;
  }

  public PVector averagePosition() {
   PVector average = new PVector(0, 0);
   float totalX = 0;
   float totalY = 0;
   for(PVector position : snake) {
     totalX += position.x;
     totalY += position.y;
   }
   
   average.x = totalX / snake.size() / dimensions.x;
   average.y = totalY / snake.size() / dimensions.y;
   
   return average;
  }

  private boolean hitSelf(PVector currentPosition) {
    for (int i = 0; i < snake.size() - 1; i++) {
      if (snake.get(i).x == currentPosition.x && snake.get(i).y == currentPosition.y) {
        return true;
      }
    }
    return false;
  }

  public void drawSnake() {
    for (PVector position : snake) {
      noStroke();
      fill(0, 255, 0);
      rect(position.x * snakeSize, map(position.y, 0, dimensions.y - 1, dimensions.y - 1, 0) * snakeSize, snakeSize, snakeSize);
    }
  }

  public void move() {
    PVector newPosition = currentPosition.copy().add(direction);
    currentPosition = newPosition;
    snake.add(newPosition);
    if (currentPosition.x == apple.x && currentPosition.y == apple.y) {
      snakeLength++;
      generateApple();
    }
    while (snake.size() > snakeLength) {
      if (show) {
        noStroke();
        fill(51);
        rect(snake.get(0).x * snakeSize, map(snake.get(0).y, 0, dimensions.y - 1, dimensions.y - 1, 0) * snakeSize, snakeSize, snakeSize);
      }
      snake.remove(0);
    }
  }

  Matrix left = new Matrix(new float[][] {{0, 1}, {-1, 0}});
  Matrix right = new Matrix(new float[][] {{0, -1}, {1, 0}});

  public boolean turnRight() {
    float[] tmp = {direction.x, direction.y};
    Matrix dir = Matrix.multiply(right, Matrix.vectorFromArray(tmp), false);
    direction.x = dir.data[0][0];
    direction.y = dir.data[1][0];
    return updateSnake();
  }

  public boolean turnLeft() {
    float[] tmp = {direction.x, direction.y};
    Matrix dir = Matrix.multiply(left, Matrix.vectorFromArray(tmp), false);
    direction.x = dir.data[0][0];
    direction.y = dir.data[1][0];
    return updateSnake();
  }

  public boolean moveForward() {
    return updateSnake();
  }

  public PVector simulateLeft() {
    float[] tmp = {direction.x, direction.y};
    Matrix dir = Matrix.multiply(left, Matrix.vectorFromArray(tmp), false);
    PVector _direction = new PVector();
    _direction.x = dir.data[0][0];
    _direction.y = dir.data[1][0];

    return currentPosition.copy().add(_direction);
  }

  public PVector simulateRight() {
    float[] tmp = {direction.x, direction.y};
    Matrix dir = Matrix.multiply(right, Matrix.vectorFromArray(tmp), false);
    PVector _direction = new PVector();
    _direction.x = dir.data[0][0];
    _direction.y = dir.data[1][0];

    return currentPosition.copy().add(_direction);
  }

  public PVector simulateForward() {
    return currentPosition.copy().add(direction);
  }

  public PVector simulateLeft(PVector currentPosition, PVector direction) {
    float[] tmp = {direction.x, direction.y};
    Matrix dir = Matrix.multiply(left, Matrix.vectorFromArray(tmp), false);
    PVector _direction = new PVector();
    _direction.x = dir.data[0][0];
    _direction.y = dir.data[1][0];

    return currentPosition.copy().add(_direction);
  }

  public PVector simulateRight(PVector currentPosition, PVector direction) {
    float[] tmp = {direction.x, direction.y};
    Matrix dir = Matrix.multiply(right, Matrix.vectorFromArray(tmp), false);
    PVector _direction = new PVector();
    _direction.x = dir.data[0][0];
    _direction.y = dir.data[1][0];

    return currentPosition.copy().add(_direction);
  }

  public PVector simulateForward(PVector currentPosition, PVector direction) {
    return currentPosition.copy().add(direction);
  }
}