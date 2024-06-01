from kubernetes import client, config

def delete_pv(pv_name):
    config.load_kube_config()  # Load kubeconfig from default location

    api_instance = client.CoreV1Api()
    try:
        api_instance.delete_persistent_volume(
            name=pv_name,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )
        print(f"PV '{pv_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting PV: {str(e)}")

if __name__ == "__main__":
    pv_name = "your-pv-name"
    delete_pv(pv_name)
