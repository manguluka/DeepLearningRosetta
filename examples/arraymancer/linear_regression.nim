import strformat, random
import ../../../Arraymancer/src/arraymancer 

# Set environment variables
let 
  epochs = 1000
  input_dimension = 1
  output_dimension = 1


# Define network context 
let ctx = newContext Tensor[float32]

# Define training data
let
  x = ctx.variable([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]].toTensor().asType(float32), requires_grad = true)
  y = [[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]].toTensor().asType(float32)

# Define Network 
network ctx, LinearRegression:
  layers:
    linear: Linear(input_dimension, output_dimension)
  forward x:
    x.linear


# Initialize model and Optimizer
let
  model = ctx.init(LinearRegression)
  optimizer = model.optimizerSGD(learning_rate = 1e-4'f32)

# Training
for epoch in 0 ..< epochs:
  # Forward Pass
  let
    y_pred = model.forward(x)
    loss = mse_loss(y_pred, y)

  # Backword Pass
  loss.backprop()
  optimizer.update()

  echo &"Epoch {epoch}: loss {loss.value[0]}"
