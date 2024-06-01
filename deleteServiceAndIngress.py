from kubernetes import client, config

def delete_service_and_ingress(namespace, service_name, ingress_name):
    config.load_kube_config()  # Load kubeconfig from default location

    api_instance = client.CoreV1Api()
    try:
        api_instance.delete_namespaced_service(name=service_name, namespace=namespace)
        print(f"Service '{service_name}' deleted successfully.")

        api_instance = client.NetworkingV1Api()
        api_instance.delete_namespaced_ingress(name=ingress_name, namespace=namespace)
        print(f"Ingress '{ingress_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting Service or Ingress: {str(e)}")

if __name__ == "__main__":
    namespace = "your-namespace"
    service_name = "your-service-name"
    ingress_name = "your-ingress-name"
    delete_service_and_ingress(namespace, service_name, ingress_name)
