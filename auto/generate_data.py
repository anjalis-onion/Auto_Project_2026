import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from complete_control import (
    make_car,
    update,
    calculate_desired_acceleration,
    acceleration_to_throttle_percentage
)



K_P = 0.5
K_I = 0.0
K_D = 0.0

STEPS = 100

velocities = []
desired_velocities = []
accelerations = []


# Try many different desired velocities
for desired_v in range(5, 31):

    car = make_car(
        desired_v=desired_v,
        dt=0.1
    )

    for _ in range(STEPS):

        desired_acceleration, error = calculate_desired_acceleration(
            car,
            K_P,
            K_I,
            K_D
        )

        # Save the current state and PID's answer
        velocities.append(car["v"])
        desired_velocities.append(car["desired_v"])
        accelerations.append(desired_acceleration)

        # Actually move the simulated car
        throttle = acceleration_to_throttle_percentage(
            desired_acceleration
        )

        update(car, throttle)



X = torch.tensor(
    list(zip(velocities, desired_velocities)),
    dtype=torch.float32
)

y = torch.tensor(
    accelerations,
    dtype=torch.float32
).reshape(-1, 1)


print("Input shape:", X.shape)
print("Target shape:", y.shape)



model = nn.Linear(2, 1)



loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

epochs = 1000

loss_history = []


for epoch in range(epochs):

    # Make predictions
    predictions = model(X)

    # Compare predictions to PID answers
    loss = loss_fn(predictions, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Calculate gradients
    loss.backward()

    # Update weights
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: Loss = {loss.item():.6f}"
        )



print("\nLearned parameters:")

print(
    "Weight for current velocity:",
    model.weight[0][0].item()
)

print(
    "Weight for desired velocity:",
    model.weight[0][1].item()
)

print(
    "Bias:",
    model.bias.item()
)



plt.figure()

plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")

plt.show()



with torch.no_grad():
    predictions = model(X)


plt.figure()

plt.scatter(
    y.numpy(),
    predictions.numpy()
)

plt.xlabel("PID Desired Acceleration")
plt.ylabel("ML Predicted Acceleration")
plt.title("PID vs ML Predictions")

plt.show()