from kubernetes import client, config

def delete_deployment(namespace, deployment_name):
    config.load_kube_config()  # Load kubeconfig from default location

    api_instance = client.AppsV1Api()
    try:
        api_instance.delete_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )
        print(f"Deployment '{deployment_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting deployment: {str(e)}")

if __name__ == "__main__":
    namespace = "your-namespace"
    deployment_name = "your-deployment-name"
    delete_deployment(namespace, deployment_name)
