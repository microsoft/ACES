import docker
from time import sleep

def start_container(container_name):

    client = docker.from_env()
    container = client.containers.get(container_name)
    if container.status == "running":
        print(f"Container {container_name} is already running.")
        return container
    else:
        container.start()
        print(f"Restarting stopped container with ID: {container.id}...")
        sleep(10)
        return container
