class Snake {

  PVector apple;
  ArrayList<PVector> snake = new ArrayList<PVector>();

  int snakeSize;
  int snakeLength = 3;

  PVector direction = new PVector(1, 0);
  PVector currentPosition;
  PVector dimensions;

  public Snake(int snakeSize) {
    this.snakeSize = snakeSize;
  }

  public void beginGame() {
    dimensions = new PVector(width / snakeSize, height / snakeSize);
    currentPosition = new PVector(0, 0);
    snake.add(currentPosition);
    generateApple();
    drawSnake();
  }

  public float distanceToApple(PVector currentPosition) {
    float a = apple.x - currentPosition.x;
    float b = apple.y - currentPosition.y;
    a *= a;
    b *= b;

    return (float)Math.sqrt(a + b);
  }

  public void generateApple() {
    apple = new PVector(floor(random(dimensions.x)), floor(random(dimensions.y))); 
    noStroke();
    fill(255, 0, 0);
    rect(apple.x * snakeSize, map(apple.y, 0, dimensions.y - 1, dimensions.y - 1, 0) * snakeSize, snakeSize, snakeSize);
  }

  public boolean updateSnake() {
    move();
    drawSnake();
    if (dead(currentPosition)) {
      //println("GAME OVER"); 
      //background(255, 0, 0);
      reset();
      return false;
    }
    return true;
  }

  private void reset() {
    background(51);
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
      noStroke();
      fill(51);
      rect(snake.get(0).x * snakeSize, map(snake.get(0).y, 0, dimensions.y - 1, dimensions.y - 1, 0) * snakeSize, snakeSize, snakeSize);
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
}