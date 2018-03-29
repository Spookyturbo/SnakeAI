NeuralNetwork net;

float[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
float[][] answers = {{0}, {1}, {1}, {0}};

boolean trainingMode = false;
void setup() {
  net = new NeuralNetwork(2, new int[] {20, 20, 20, 20, 20}, 1);
}

void draw() {
}

void keyPressed() {
  if (key == 'q') {
    if (trainingMode) {
      net.train(inputs[0], answers[0]);
    } else {
      println(net.feedForward(inputs[0])[0]);
    }
  } else if (key == 'e') {
    if (trainingMode) {
      net.train(inputs[1], answers[1]);
    } else {
      println(net.feedForward(inputs[1])[0]);
    }
  } else if (key == 'z') {
    if (trainingMode) {
      net.train(inputs[2], answers[2]);
    } else {
      println(net.feedForward(inputs[2])[0]);
    }
  } else if (key == 'c') {
    if (trainingMode) {
      net.train(inputs[3], answers[3]);
    } else {
      println(net.feedForward(inputs[3])[0]);
    }
  } else if (key == 'a') {
    trainingMode = !trainingMode;
  } else {
    for (int i = 0; i < 100; i++) {
      int r = floor(random(inputs.length)); 
      net.train(inputs[r], answers[r]);
    }
  }
}