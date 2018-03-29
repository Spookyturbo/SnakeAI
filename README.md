# SnakeAI
This includes a few variants of using a neural network on a snake game. Right now the code is very messy and will hopefully be refactored in a bit.

To describe some of these files really fast:

Genetic Snake V3 is poorly named. It is a snake with 24 inputs. 8 Directions * 3 searches
It looks in each of the 8 directions (Up, down, left, right, and the diagonals) and it checks for distance to apple, distance to wall, and distance to snake. Using this it outputs a value to determine if the snake moves forward, left, or right. It has some issues that need fixing such as when it gets too long it falls into a local minima and decides it can no longer go after food fairly often

MultilayeredEvolutionaryNeuroEvolutionarySnake is the same snake game as the Genetic Snake V3 but with the neural network having more then one hidden layer. It does not have much difference, but feel free to look at it

Multiple Genetic Snake is the first example of the snake being trained using a genetic algorithm. It shows ALL of the snakes at once, so it can be quite chaotic. Over time it should get better. The inputs for this are different however. It receives a desired direction to go as well as the angle from the snake head to the apple, and then it outputs a value between 0-1 saying how good that option is with 1 being very good.

NeuralNetworkSnakeAI is the very first snake AI made. It does not use a genetic algorithm but instead uses backpropagation and gradient descent to tune the network every move it makes. This one will slowly improve over time even while it is still alive. It also learns the fastest out of these. You will see almost all of these snake AIs have the problem of not knowing how to not trap themselves.

NewNeuralNetwork is the neural network present in the MulitlayeredEvolutionaryNeuroEvolutionarySnake. It is NOT the same as the one in multiple genetic snake, however it is more advanced and should probably be used in all these scenarios. This example is showing the neural network being used on the XOR problem. If you hold down any key other then q e z and c it will tune itself. Once it is tuned pressing q and c should give around 0 and e and z should give about 1.

For all of the snakes with a lot of snakes at once, if you press 'a' while it is running, once the current generation dies, it will show you only what it perceives to be the current best snake. To return to training mode press 's' and they will immediately begin retraining. Make sure the application has focus (is clicked) at the time of pressing the button or it will not receive any input.

I will try and update these to be more readable later, other then that enjoy.
