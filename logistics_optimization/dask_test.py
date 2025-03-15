
from dask.distributed import Client

# Connect to the Dask scheduler
client = Client("tcp://127.0.0.1:8786")

def square(x):
        return x ** 2

future = client.submit(square, 10)
print("Result:", future.result())

# Check cluster info
print("Dask Cluster Info:", client)

