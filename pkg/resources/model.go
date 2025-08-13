// Package resources provides Pulumi resource implementations for GCP Vertex AI model upload and deployment.
package resources

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	gax "github.com/googleapis/gax-go/v2"
	"github.com/googleapis/gax-go/v2/apierror"
	"github.com/pulumi/pulumi-go-provider/infer"
	"google.golang.org/api/option"
)

// VertexModelDeployment represents a Pulumi resource for deploying models to Vertex AI endpoints.
type VertexModelDeployment struct{}

// Annotate provides metadata and descriptions for the VertexModelDeployment resource.
func (VertexModelDeployment) Annotate(annotator infer.Annotator) {
	annotator.Describe(&VertexModelDeployment{}, "Deploys a model to a Vertex AI endpoint")
}

// VertexModelDeploymentArgs defines the input arguments for creating a Vertex AI model deployment.
type VertexModelDeploymentArgs struct {
	ProjectID               string            `pulumi:"projectId"`
	Region                  string            `pulumi:"region"`
	EndpointID              string            `pulumi:"endpointId"`
	ModelImageURL           string            `pulumi:"modelImageUrl"`
	ModelArtifactsBucketURI string            `pulumi:"modelArtifactsBucketUri"`
	MachineType             string            `pulumi:"machineType,optional"`
	MinReplicas             int               `pulumi:"minReplicas,optional"`
	MaxReplicas             int               `pulumi:"maxReplicas,optional"`
	TrafficPercent          int               `pulumi:"trafficPercent,optional"`
	ServiceAccount          string            `pulumi:"serviceAccount,optional"`
	Labels                  map[string]string `pulumi:"labels,optional"`
}

// Annotate provides metadata and default values for the VertexModelDeploymentArgs.
func (args *VertexModelDeploymentArgs) Annotate(annotator infer.Annotator) {
	annotator.Describe(&args.ProjectID, "Google Cloud Project ID")
	annotator.Describe(&args.Region, "Google Cloud region")
	annotator.Describe(&args.EndpointID, "Vertex AI Endpoint ID")
	annotator.Describe(&args.ModelImageURL, "Vertex AI Image URL of a custom or prebuilt container model server. See: https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers")
	annotator.Describe(&args.ModelArtifactsBucketURI, "Bucket URI to the model artifacts. For instance, gs://my-bucket/my-model-artifacts/ - See: https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts")
	annotator.Describe(&args.MachineType, "Machine type for deployment")
	annotator.Describe(&args.MinReplicas, "Minimum number of replicas")
	annotator.Describe(&args.MaxReplicas, "Maximum number of replicas")
	annotator.Describe(&args.TrafficPercent, "Traffic percentage for this deployment")
	annotator.Describe(&args.ServiceAccount, "Service account for the deployment")

	// Set defaults
	annotator.SetDefault(&args.MachineType, "n1-standard-2")
	annotator.SetDefault(&args.MinReplicas, 1)
	annotator.SetDefault(&args.MaxReplicas, 3)
	annotator.SetDefault(&args.TrafficPercent, 100)
}

// VertexModelDeploymentState represents the state of a deployed Vertex AI model.
type VertexModelDeploymentState struct {
	VertexModelDeploymentArgs
	DeployedModelID string `pulumi:"deployedModelId"`
	EndpointName    string `pulumi:"endpointName"`
	CreateTime      string `pulumi:"createTime"`
}

// Annotate provides metadata and descriptions for the VertexModelDeploymentState outputs.
func (state *VertexModelDeploymentState) Annotate(annotator infer.Annotator) {
	annotator.Describe(&state.DeployedModelID, "ID of the deployed model")
	annotator.Describe(&state.EndpointName, "Full name of the endpoint")
	annotator.Describe(&state.CreateTime, "Creation timestamp")
}

// Create implements the creation logic
func (VertexModelDeployment) Create(
	ctx context.Context,
	req infer.CreateRequest[VertexModelDeploymentArgs],
) (infer.CreateResponse[VertexModelDeploymentState], error) {

	state := VertexModelDeploymentState{
		VertexModelDeploymentArgs: req.Inputs,
	}

	if req.DryRun {
		return infer.CreateResponse[VertexModelDeploymentState]{
			ID: fmt.Sprintf("%s-%s", req.Inputs.EndpointID, req.Inputs.ModelImageURL),
		}, nil
	}

	// Regional models and endpoints require regional endpoints
	apiEndpoint := fmt.Sprintf("%s-aiplatform.googleapis.com:443", req.Inputs.Region)
	clientEndpointOpt := option.WithEndpoint(apiEndpoint)

	// Vertex endpoint client with Application Default Credentials
	client, err := aiplatform.NewEndpointClient(ctx, clientEndpointOpt)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to create endpoint client: %w", err)
	}
	defer func() {
		if closeErr := client.Close(); closeErr != nil {
			log.Printf("failed to close endpoint client: %v", closeErr)
		}
	}()

	// Vertex model client with Application Default Credentials
	modelClient, err := aiplatform.NewModelClient(ctx, clientEndpointOpt)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to create model client: %w", err)
	}
	defer func() {
		if closeErr := modelClient.Close(); closeErr != nil {
			log.Printf("failed to close endpoint client: %v", closeErr)
		}
	}()

	modelUploadOp, err := modelClient.UploadModel(ctx, &aiplatformpb.UploadModelRequest{
		// TODO support non traditional / global models
		// Endpoint to which the model is attached. It can be regional or global, depending on the model type.
		Parent:         fmt.Sprintf("projects/%s/locations/%s", req.Inputs.ProjectID, req.Inputs.Region),
		ServiceAccount: req.Inputs.ServiceAccount,
		Model: &aiplatformpb.Model{
			DisplayName: req.Name,
			Description: "Uploaded model for " + req.Inputs.ModelImageURL,
			ContainerSpec: &aiplatformpb.ModelContainerSpec{
				ImageUri: req.Inputs.ModelImageURL,
				// TODO make me configurable
				Args: []string{
					"--allow_precompilation=false",
					"--disable_optimizer=true",
					"--saved_model_tags='serve,tpu'",
					"--use_tfrt=true",
				},
			},
			Labels: req.Inputs.Labels,
			// May be optional for custom models but required for TensorFlow pre-built images.
			// See:
			// - https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts#framework-requirements
			ArtifactUri: req.Inputs.ModelArtifactsBucketURI,
		},
	}, gax.WithTimeout(5*time.Minute))
	if err != nil {
		var apiError *apierror.APIError
		if errors.As(err, &apiError) {
			// TODO DRY up
			log.Printf("Model upload returned APIError details: %v\n", err)
			log.Printf("APIError reason: %v\n", apiError.Reason())
			log.Printf("APIError details : %v\n", apiError.Details())
			// If a gRPC transport was used you can extract the
			// google.golang.org/grpc/status.Status from the error
			log.Printf("APIError GRPCStatus: %+v\n", apiError.GRPCStatus())
			log.Printf("APIError HTTPCode: %+v\n", apiError.HTTPCode())
		}

		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to upload model: %w", err)
	}

	modelUploadResult, err := modelUploadOp.Wait(context.Background(), gax.WithTimeout(10*time.Minute))
	if err != nil {
		if modelUploadOp.Done() {
			log.Printf("Model upload operation completed with failure: %v\n", err)
		}
		var apiError *apierror.APIError
		if errors.As(err, &apiError) {
			// TODO DRY up
			log.Printf("Model upload returned APIError details: %v\n", err)
			log.Printf("APIError reason: %v\n", apiError.Reason())
			log.Printf("APIError details : %v\n", apiError.Details())
			log.Printf("APIError help: %v\n", apiError.Details().Help)
			// If a gRPC transport was used you can extract the
			// google.golang.org/grpc/status.Status from the error
			log.Printf("APIError GRPCStatus: %v\n", apiError.GRPCStatus())
			log.Printf("APIError HTTPCode: %v\n", apiError.HTTPCode())
		}

		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to wait for model upload: %w", err)
	}

	// Build the deployment request
	deployedModel := &aiplatformpb.DeployedModel{
		// Expected format: "projects/%s/locations/%s/models/%s"
		Model:       modelUploadResult.GetModel(),
		DisplayName: req.Name,
		PredictionResources: &aiplatformpb.DeployedModel_DedicatedResources{
			DedicatedResources: &aiplatformpb.DedicatedResources{
				MachineSpec: &aiplatformpb.MachineSpec{
					MachineType: req.Inputs.MachineType,
				},
				MinReplicaCount: safeIntToInt32(req.Inputs.MinReplicas),
				MaxReplicaCount: safeIntToInt32(req.Inputs.MaxReplicas),
			},
		},
	}

	if req.Inputs.ServiceAccount != "" {
		deployedModel.ServiceAccount = req.Inputs.ServiceAccount
	}

	deployReq := &aiplatformpb.DeployModelRequest{
		Endpoint: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			req.Inputs.ProjectID, req.Inputs.Region, req.Inputs.EndpointID),
		DeployedModel: deployedModel,
		TrafficSplit:  map[string]int32{
			// TODO set for parallel model deployments
		},
	}

	// Execute the deployment
	deployOperation, err := client.DeployModel(ctx, deployReq)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to deploy model: %w", err)
	}

	// Wait for completion with timeout
	result, err := deployOperation.Wait(ctx)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to wait for deployment: %w", err)
	}

	deployedModelID := result.GetDeployedModel().GetId()

	state.DeployedModelID = deployedModelID
	state.EndpointName = req.Inputs.EndpointID
	state.CreateTime = time.Now().Format(time.RFC3339)

	return infer.CreateResponse[VertexModelDeploymentState]{
		ID:     deployedModelID,
		Output: state,
	}, nil
}

// Delete implements the deletion logic
func (VertexModelDeployment) Delete(
	ctx context.Context,
	req infer.DeleteRequest[VertexModelDeploymentState],
) (infer.DeleteResponse, error) {

	// TODO set location
	// With Application Default Credentials
	client, err := aiplatform.NewEndpointClient(ctx)
	if err != nil {
		return infer.DeleteResponse{}, fmt.Errorf("failed to create endpoint client: %w", err)
	}
	defer func() {
		if closeErr := client.Close(); closeErr != nil {
			log.Printf("failed to close endpoint client: %v", closeErr)
		}
	}()

	undeployReq := &aiplatformpb.UndeployModelRequest{
		Endpoint: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			req.State.ProjectID, req.State.Region, req.State.EndpointID),
		DeployedModelId: req.State.DeployedModelID,
	}

	undeployOperation, err := client.UndeployModel(ctx, undeployReq)
	if err != nil {
		return infer.DeleteResponse{}, fmt.Errorf("failed to undeploy model: %w", err)
	}

	_, err = undeployOperation.Wait(ctx)
	if err != nil {
		return infer.DeleteResponse{}, fmt.Errorf("failed to wait for undeployment: %w", err)
	}

	return infer.DeleteResponse{}, nil
}

// Update implements the update logic
func (VertexModelDeployment) Update(
	_ context.Context,
	req infer.UpdateRequest[VertexModelDeploymentArgs, VertexModelDeploymentState],
) (infer.UpdateResponse[VertexModelDeploymentState], error) {

	// TODO For simplicity, we'll recreate on any change
	return infer.UpdateResponse[VertexModelDeploymentState]{
		Output: VertexModelDeploymentState{
			VertexModelDeploymentArgs: req.Inputs,
			DeployedModelID:           req.State.DeployedModelID,
			EndpointName:              req.State.EndpointName,
			CreateTime:                req.State.CreateTime,
		},
	}, nil
}

// Read implements the read logic for drift detection
func (VertexModelDeployment) Read(
	ctx context.Context,
	req infer.ReadRequest[VertexModelDeploymentArgs, VertexModelDeploymentState],
) (infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState], error) {

	// TODO set location
	// With Application Default Credentials
	client, err := aiplatform.NewEndpointClient(ctx)
	if err != nil {
		return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, err
	}
	defer func() {
		if closeErr := client.Close(); closeErr != nil {
			log.Printf("failed to close endpoint client: %v", closeErr)
		}
	}()

	getReq := &aiplatformpb.GetEndpointRequest{
		Name: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			req.State.ProjectID, req.State.Region, req.State.EndpointID),
	}

	endpoint, err := client.GetEndpoint(ctx, getReq)
	if err != nil {
		return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, err
	}

	// Verify the deployed model still exists
	var found bool
	for _, deployedModel := range endpoint.DeployedModels {
		if deployedModel.Id == req.ID {
			found = true

			break
		}
	}

	if !found {
		// Model is no longer deployed
		return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, nil
	}

	return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState](req), nil
}
