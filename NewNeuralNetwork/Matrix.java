import processing.core.PApplet;

public class Matrix {

  int rows, cols;

  public float[][] data;

  public Matrix(int rows, int cols) {
    this.rows = rows;
    this.cols = cols;

    data = new float[rows][cols];

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = 0;
      }
    }
  }

  public Matrix(float[][] info) { 
    this.rows = info[0].length;
    this.cols = info.length;
    this.data = Matrix.fromArray(info).data;
  }

  public void randomize() { //initializes matrix with values between -1 and 1
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = (float)(Math.random() * 2 - 1);
      }
    }
  }

  public void print() {
    for (int i = 0; i < rows; i++) {
      PApplet.println("");
      for (int j = 0; j < cols; j++) {
        PApplet.print(data[i][j] + ", ");
      }
    }

    PApplet.println("");
  }

  public static Matrix vectorFromArray(float[] inputArray) { //Used for turning inputs to Matrix
    Matrix m = new Matrix(inputArray.length, 1);

    for (int i = 0; i < inputArray.length; i++) {
      m.data[i][0] = inputArray[i];
    }

    return m;
  }

  public static Matrix fromArray(float[][] inputArray) {
    Matrix m = new Matrix(inputArray[0].length, inputArray.length);

    for (int i = 0; i < inputArray.length; i++) {
      for (int j = 0; j < inputArray[0].length; j++) {
        m.data[j][i] = inputArray[i][j];
      }
    }

    return m;
  }

  public static Matrix multiply(Matrix a, float value) {
    Matrix m = new Matrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < a.cols; j++) {
        m.data[i][j] = a.data[i][j] * value;
      }
    }

    return m;
  }

  public static Matrix multiply(Matrix a, Matrix b, boolean elementWise) {

    if (elementWise) {
      if (a.rows == b.rows && a.cols == b.cols) {
        Matrix m = new Matrix(a.rows, a.cols);

        for (int i = 0; i < a.rows; i++) {
          for (int j = 0; j < a.cols; j++) {
            m.data[i][j] = a.data[i][j] * b.data[i][j];
          }
        }

        return m;
      } else {
        PApplet.println("Scalar multiplication has non equal dimensions!");
        return null;
      }
    } else {
      if (a.cols == b.rows) {
        Matrix m = new Matrix(a.rows, b.cols);

        for (int row = 0; row < a.rows; row++) {
          for (int col = 0; col < b.cols; col++) {
            float sum = 0;
            for (int i = 0; i < a.cols; i++) {
              sum += a.data[row][i] * b.data[i][col];
            }
            m.data[row][col] = sum;
          }
        }

        return m;
      } else {
        PApplet.println("Non Scalar multiplication dimensions are not valid");
        return null;
      }
    }
  }

  public static Matrix add(Matrix a, Matrix b) {
    if (a.rows == b.rows && a.cols == b.cols) {
      Matrix m = new Matrix(a.rows, a.cols);

      for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
          m.data[i][j] = a.data[i][j] + b.data[i][j];
        }
      }

      return m;
    } else {
      PApplet.println("These matrices do not have the same dimensions!");
      return null;
    }
  }

  public static Matrix subtract(Matrix a, Matrix b) {
    if (a.rows == b.rows && a.cols == b.cols) {
      Matrix m = new Matrix(a.rows, a.cols);

      for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
          m.data[i][j] = a.data[i][j] - b.data[i][j];
        }
      }

      return m;
    } else {
      PApplet.println("These matrices do not have the same dimensions!");
      return null;
    }
  }

  public static Matrix transpose(Matrix a) { //Maps rows to cols and cols to rows
    Matrix m = new Matrix(a.cols, a.rows);

    for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < a.cols; j++) {
        m.data[j][i] = a.data[i][j];
      }
    }

    return m;
  }

  public static Matrix map(Matrix a, MapInput change) { //Takes fuction change and applies to all values in a
    Matrix m = new Matrix(a.rows, a.cols);

    for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < a.cols; j++) {  
        m.data[i][j] = change.change(a.data[i][j]);
      }
    }

    return m;
  }

  public void map(MapInput change) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {  
        data[i][j] = change.change(data[i][j]);
      }
    }
  }
}