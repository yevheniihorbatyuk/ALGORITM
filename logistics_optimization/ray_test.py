import ray

# Check if running inside Kubernetes
import os
ray_address = "ray://127.0.0.1:10001" #if os.getenv("KUBERNETES_SERVICE_HOST") else "local"

# Connect to Ray (local or cluster)
ray.init(address=ray_address)

@ray.remote
def square(x):
    return x * x

@ray.remote
def say_hello():
    return "Hello from Ray!"

# Run tasks
futures = [square.remote(i) for i in range(20)]
hello_future = say_hello.remote()

print("Squared Results:", ray.get(futures))
print(ray.get(hello_future))

# Verify cluster resources
print("Cluster Resources:", ray.cluster_resources())

