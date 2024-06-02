import os
import tempfile

import geminiplus as gemini
from google.cloud import automl

# TODO(developer): Uncomment and set the following variables
# project_id = "YOUR_PROJECT_ID"
# compute_region = "us-central1"


def automl_rag_sample(
    project_id: str, compute_region: str, dataset_id: str, model_name: str
):
    """
    Create and deploy automl rag model using
    google universal sentence encoder
    Args:
      project_id: Id of the project.
      compute_region: Region name.
      dataset_id: Id of the dataset.
      model_name: Name of the model.
    """
    client = automl.AutoMlClient()

    # A resource that represents Google Cloud Platform location.
    project_location = f"projects/{project_id}/locations/{compute_region}"
    # Leave model unset to use the default base model provided by Google
    metadata = automl.RagMetadata()

    model = automl.Model(
        display_name=model_name, dataset_id=dataset_id, rag_metadata=metadata
    )

    # Create a model with the model metadata in the region.
    response = client.create_model(parent=project_location, model=model)

    print("Training started...")
    print("Training operation name: {}".format(response.operation.name))
    print("Training completed: {}".format(response.result()))
    print("Model name: {}".format(response.result().name))

    model_id = os.path.splitext(response.result().name)[0].split("/")[-1]
    model_full_id = f"{project_id}/{compute_region}/models/{model_id}"

    # Get the model's deployment state
    response = client.get_model(name=model_full_id)

    # Deploy a model
    if response.deployment_state == automl.Model.DeploymentState.UNDEPLOYED:
        request = automl.DeployModelRequest(name=model_full_id)
        client.deploy_model(request=request)
        print(
            "Model deployed. {}".format(response.deployment_state)
        )

    # Get deployed endpoint
    endpoint = response.deployment_state_metadata.endpoint
    print("Deployed model endpoint: {}".format(endpoint))

    # Export a model
    output_config = {"gcs_destination": {"output_uri_prefix": tempfile.mkdtemp()}}
    response = client.export_model(name=model_full_id, output_config=output_config)

    print("Model exported: {}".format(response.result()))

    # Print out the model evaluation result
    print("Model evaluation:")
    print(
        "RAG mrr @10: {}".format(
            response.result().evaluation_result.rag_evaluation_metrics.mrr_at10
        )
    )


if __name__ == "__main__":
    automl_rag_sample(
        project_id=project_id,
        compute_region=compute_region,
        dataset_id="YOUR_DATASET_ID",
        model_name="YOUR_MODEL_NAME",
    )