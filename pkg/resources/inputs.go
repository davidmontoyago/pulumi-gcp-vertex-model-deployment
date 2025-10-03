// Package resources provides Pulumi resource implementations for GCP Vertex model upload to the registry and endpoint deployment.
package resources

import "github.com/pulumi/pulumi-go-provider/infer"

// VertexModelDeploymentArgs defines the input arguments for creating a Vertex AI model deployment.
type VertexModelDeploymentArgs struct {
	ProjectID     string `pulumi:"projectId"`
	Region        string `pulumi:"region"`
	ModelImageURL string `pulumi:"modelImageUrl"`

	ModelArtifactsBucketURI          string `pulumi:"modelArtifactsBucketUri,optional"`
	ModelPredictionInputSchemaURI    string `pulumi:"modelPredictionInputSchemaUri,optional"`
	ModelPredictionOutputSchemaURI   string `pulumi:"modelPredictionOutputSchemaUri,optional"`
	ModelPredictionBehaviorSchemaURI string `pulumi:"modelPredictionBehaviorSchemaUri,optional"`
	// If ModelImage is pointing to a private registry, this service account
	// must have read access to the registry.
	ServiceAccount string `pulumi:"serviceAccount"`

	// Path on the container to send prediction requests to.
	// Not required for Endpoints.
	PredictRoute string `pulumi:"predictRoute,optional"`
	// Path on the container to send health requests to.
	// Not required for Endpoints.
	HealthRoute string `pulumi:"healthRoute,optional"`

	Args    []string          `pulumi:"args,optional"`
	EnvVars map[string]string `pulumi:"env,optional"`
	Port    int32             `pulumi:"port,optional"`

	// Target endpoint for the model deployment.
	//
	// Set only when serving the model on a Vertex AI Endpoint.
	// Deploying a custom or dockerized model to a Vertex AI Endpoint is not yet supported
	// by Terraform nor the Pulumi Google Cloud Native provider, hence, this custom provider
	// exists.
	// See: https://github.com/hashicorp/terraform-provider-google/issues/15303
	//
	// When deploying the model as a Batched Prediction Job, this field must be
	// unset and the batch job must be created using the Pulumi Google Cloud Native
	// provider.
	EndpointModelDeployment *EndpointModelDeploymentArgs `pulumi:"endpointModelDeployment,optional"`

	Labels map[string]string `pulumi:"labels,optional"`
}

// EndpointModelDeploymentArgs defines the input arguments for deploying an
// uploaded model to a Vertex AI endpoint.
type EndpointModelDeploymentArgs struct {
	EndpointID       string `pulumi:"endpointId"`
	MachineType      string `pulumi:"machineType,optional"`
	AcceleratorType  string `pulumi:"acceleratorType,optional"`
	AcceleratorCount int32  `pulumi:"acceleratorCount,optional"`
	MinReplicas      int    `pulumi:"minReplicas,optional"`
	MaxReplicas      int    `pulumi:"maxReplicas,optional"`
	TrafficPercent   int    `pulumi:"trafficPercent,optional"`
}

// Annotate provides metadata and default values for the VertexModelDeploymentArgs.
func (args *VertexModelDeploymentArgs) Annotate(annotator infer.Annotator) {
	annotator.Describe(&args.ProjectID, "Google Cloud Project ID")
	annotator.Describe(&args.Region, "Google Cloud region")
	annotator.Describe(&args.ModelImageURL, "Vertex AI Image URL of a custom or prebuilt container model server. See: https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers")
	annotator.Describe(&args.ModelArtifactsBucketURI, "Bucket URI to the model artifacts. For instance, gs://my-bucket/my-model-artifacts/ - See: https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts")
	annotator.Describe(&args.ModelPredictionInputSchemaURI, "Bucket URI to the schema for the model input")
	annotator.Describe(&args.ModelPredictionOutputSchemaURI, "Bucket URI to the schema for the model output")
	annotator.Describe(&args.ModelPredictionBehaviorSchemaURI, "Bucket URI to the schema for the model inference behavior")
	annotator.Describe(&args.ServiceAccount, "Service account for the model. If ModelImage is pointing to a private registry, this service account must have read access to the registry.")
	annotator.Describe(&args.Args, "Dockerized model server command line arguments")
	annotator.Describe(&args.EnvVars, "Environment variables")
	annotator.Describe(&args.Port, "Port for the model server. Defaults to 8080.")
	annotator.Describe(&args.EndpointModelDeployment, "Configuration for deploying the model to a Vertex AI endpoint. Leave empty to upload model only for batched predictions.")
	annotator.Describe(&args.Labels, "Labels for the deployment")
}

// Annotate provides metadata and default values for the EndpointModelDeploymentArgs.
func (args *EndpointModelDeploymentArgs) Annotate(annotator infer.Annotator) {
	annotator.Describe(&args.EndpointID, "Vertex AI Endpoint ID")
	annotator.Describe(&args.MachineType, "Machine type for deployment")
	annotator.Describe(&args.AcceleratorType, "Accelerator type for endpoint deployment. Defaults to ACCELERATOR_TYPE_UNSPECIFIED. E.g.: NVIDIA_TESLA_P4, NVIDIA_TESLA_T4")
	annotator.Describe(&args.AcceleratorCount, "Accelerator count for deployment")
	annotator.Describe(&args.MinReplicas, "Minimum number of replicas")
	annotator.Describe(&args.MaxReplicas, "Maximum number of replicas")
	annotator.Describe(&args.TrafficPercent, "Traffic percentage for this deployment")

	// Set defaults
	annotator.SetDefault(&args.MachineType, "n1-standard-8")
	annotator.SetDefault(&args.MinReplicas, 1)
	annotator.SetDefault(&args.MaxReplicas, 3)
	annotator.SetDefault(&args.TrafficPercent, 100)
}
