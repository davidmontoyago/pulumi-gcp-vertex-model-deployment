// Package resources provides Pulumi resource implementations for GCP Vertex AI model upload and deployment.
package resources

import (
	"context"
	"fmt"
	"log"
	"time"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"github.com/pulumi/pulumi-go-provider/infer"

	"github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/pkg/services"
)

// VertexModelDeployment represents a Pulumi resource for deploying models to Vertex AI endpoints.
type VertexModelDeployment struct{}

// testFactoryRegistry holds test factories for dependency injection during testing
var testFactoryRegistry struct {
	modelClientFactory    services.ModelClientFactory
	endpointClientFactory services.EndpointClientFactory
}

// Annotate provides metadata and descriptions for the VertexModelDeployment resource.
func (VertexModelDeployment) Annotate(annotator infer.Annotator) {
	annotator.Describe(&VertexModelDeployment{}, "Deploys a model to a Vertex AI endpoint")
}

// VertexModelDeploymentArgs defines the input arguments for creating a Vertex AI model deployment.
type VertexModelDeploymentArgs struct {
	ProjectID                        string            `pulumi:"projectId"`
	Region                           string            `pulumi:"region"`
	EndpointID                       string            `pulumi:"endpointId"`
	ModelImageURL                    string            `pulumi:"modelImageUrl"`
	ModelArtifactsBucketURI          string            `pulumi:"modelArtifactsBucketUri"`
	ModelPredictionInputSchemaURI    string            `pulumi:"modelPredictionInputSchemaUri"`
	ModelPredictionOutputSchemaURI   string            `pulumi:"modelPredictionOutputSchemaUri"`
	ModelPredictionBehaviorSchemaURI string            `pulumi:"modelPredictionBehaviorSchemaUri,optional"`
	MachineType                      string            `pulumi:"machineType,optional"`
	MinReplicas                      int               `pulumi:"minReplicas,optional"`
	MaxReplicas                      int               `pulumi:"maxReplicas,optional"`
	TrafficPercent                   int               `pulumi:"trafficPercent,optional"`
	ServiceAccount                   string            `pulumi:"serviceAccount,optional"`
	Labels                           map[string]string `pulumi:"labels,optional"`
}

// Annotate provides metadata and default values for the VertexModelDeploymentArgs.
func (args *VertexModelDeploymentArgs) Annotate(annotator infer.Annotator) {
	annotator.Describe(&args.ProjectID, "Google Cloud Project ID")
	annotator.Describe(&args.Region, "Google Cloud region")
	annotator.Describe(&args.EndpointID, "Vertex AI Endpoint ID")
	annotator.Describe(&args.ModelImageURL, "Vertex AI Image URL of a custom or prebuilt container model server. See: https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers")
	annotator.Describe(&args.ModelArtifactsBucketURI, "Bucket URI to the model artifacts. For instance, gs://my-bucket/my-model-artifacts/ - See: https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts")
	annotator.Describe(&args.ModelPredictionInputSchemaURI, "Bucket URI to the schema for the model input")
	annotator.Describe(&args.ModelPredictionOutputSchemaURI, "Bucket URI to the schema for the model output")
	annotator.Describe(&args.ModelPredictionBehaviorSchemaURI, "Bucket URI to the schema for the model inference behavior")
	annotator.Describe(&args.MachineType, "Machine type for deployment")
	annotator.Describe(&args.MinReplicas, "Minimum number of replicas")
	annotator.Describe(&args.MaxReplicas, "Maximum number of replicas")
	annotator.Describe(&args.TrafficPercent, "Traffic percentage for this deployment")
	annotator.Describe(&args.ServiceAccount, "Service account for the deployment")
	annotator.Describe(&args.Labels, "Labels for the deployment")

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
func (v VertexModelDeployment) Create(
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

	// Create clients using the factories
	modelClientFactory := v.getModelClientFactory()
	endpointClientFactory := v.getEndpointClientFactory()

	modelClient, err := modelClientFactory(ctx, req.Inputs.Region)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to create model client: %w", err)
	}
	defer func() {
		if closeErr := modelClient.Close(); closeErr != nil {
			log.Printf("failed to close model client: %v", closeErr)
		}
	}()

	endpointClient, err := endpointClientFactory(ctx, req.Inputs.Region)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to create endpoint client: %w", err)
	}
	defer func() {
		if closeErr := endpointClient.Close(); closeErr != nil {
			log.Printf("failed to close endpoint client: %v", closeErr)
		}
	}()

	// Create the model upload service
	uploader := services.NewVertexModelUpload(ctx, modelClient, req.Inputs.ProjectID, req.Inputs.Region, req.Inputs.Labels)
	defer func() {
		if closeErr := uploader.Close(); closeErr != nil {
			log.Printf("failed to close model upload service: %v", closeErr)
		}
	}()

	// Upload the model
	modelName, err := uploader.Upload(ctx, services.ModelUpload{
		Name:                             req.Name,
		ModelImageURL:                    req.Inputs.ModelImageURL,
		ModelArtifactsBucketURI:          req.Inputs.ModelArtifactsBucketURI,
		ServiceAccountEmail:              req.Inputs.ServiceAccount,
		ModelPredictionInputSchemaURI:    req.Inputs.ModelPredictionInputSchemaURI,
		ModelPredictionOutputSchemaURI:   req.Inputs.ModelPredictionOutputSchemaURI,
		ModelPredictionBehaviorSchemaURI: req.Inputs.ModelPredictionBehaviorSchemaURI,
	})
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to upload model: %w", err)
	}

	// Create the model deployment service
	deployer := services.NewVertexModelDeploy(ctx, endpointClient, req.Inputs.ProjectID, req.Inputs.Region)
	defer func() {
		if closeErr := deployer.Close(); closeErr != nil {
			log.Printf("failed to close model deployment service: %v", closeErr)
		}
	}()

	// Deploy the model
	deployedModelID, err := deployer.Deploy(
		ctx,
		req.Inputs.EndpointID,
		modelName,
		req.Name,
		req.Inputs.MachineType,
		req.Inputs.ServiceAccount,
		safeIntToInt32(req.Inputs.MinReplicas),
		safeIntToInt32(req.Inputs.MaxReplicas),
	)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to deploy model: %w", err)
	}

	state.DeployedModelID = deployedModelID
	state.EndpointName = req.Inputs.EndpointID
	state.CreateTime = time.Now().Format(time.RFC3339)

	return infer.CreateResponse[VertexModelDeploymentState]{
		ID:     deployedModelID,
		Output: state,
	}, nil
}

// Delete implements the deletion logic
func (v VertexModelDeployment) Delete(
	ctx context.Context,
	req infer.DeleteRequest[VertexModelDeploymentState],
) (infer.DeleteResponse, error) {

	// Create endpoint client using the factory
	endpointClientFactory := v.getEndpointClientFactory()
	endpointClient, err := endpointClientFactory(ctx, req.State.Region)
	if err != nil {
		return infer.DeleteResponse{}, fmt.Errorf("failed to create endpoint client: %w", err)
	}
	defer func() {
		if closeErr := endpointClient.Close(); closeErr != nil {
			log.Printf("failed to close endpoint client: %v", closeErr)
		}
	}()

	undeployReq := &aiplatformpb.UndeployModelRequest{
		Endpoint: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			req.State.ProjectID, req.State.Region, req.State.EndpointID),
		DeployedModelId: req.State.DeployedModelID,
	}

	undeployOperation, err := endpointClient.UndeployModel(ctx, undeployReq)
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
func (v VertexModelDeployment) Read(
	ctx context.Context,
	req infer.ReadRequest[VertexModelDeploymentArgs, VertexModelDeploymentState],
) (infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState], error) {

	// Create endpoint client using the factory
	endpointClientFactory := v.getEndpointClientFactory()
	endpointClient, err := endpointClientFactory(ctx, req.State.Region)
	if err != nil {
		return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, err
	}
	defer func() {
		if closeErr := endpointClient.Close(); closeErr != nil {
			log.Printf("failed to close endpoint client: %v", closeErr)
		}
	}()

	getReq := &aiplatformpb.GetEndpointRequest{
		Name: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			req.State.ProjectID, req.State.Region, req.State.EndpointID),
	}

	endpoint, err := endpointClient.GetEndpoint(ctx, getReq)
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

// getModelClientFactory returns the model client factory, defaulting to production factory if nil.
func (v VertexModelDeployment) getModelClientFactory() services.ModelClientFactory {
	if testFactoryRegistry.modelClientFactory == nil {
		return services.DefaultModelClientFactory
	}

	return testFactoryRegistry.modelClientFactory
}

// getEndpointClientFactory returns the endpoint client factory, defaulting to production factory if nil.
func (v VertexModelDeployment) getEndpointClientFactory() services.EndpointClientFactory {
	if testFactoryRegistry.endpointClientFactory == nil {
		return services.DefaultEndpointClientFactory
	}

	return testFactoryRegistry.endpointClientFactory
}
