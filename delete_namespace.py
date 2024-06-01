from kubernetes import client, config

def delete_namespace(namespace):
    config.load_kube_config()  # Load kubeconfig from default location

    api_instance = client.CoreV1Api()
    try:
        api_instance.delete_namespace(name=namespace)
        print(f"Namespace '{namespace}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting namespace: {str(e)}")

if __name__ == "__main__":
    namespace_to_delete = "your-namespace"
    delete_namespace(namespace_to_delete)
