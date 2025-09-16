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

// Annotate provides metadata and descriptions for the VertexModelDeployment resource.
func (VertexModelDeployment) Annotate(annotator infer.Annotator) {
	annotator.Describe(&VertexModelDeployment{}, "Deploys a model to a Vertex AI endpoint")
}

// VertexModelDeploymentState represents the state of a deployed Vertex AI model.
type VertexModelDeploymentState struct {
	VertexModelDeploymentArgs
	DeployedModelID string `pulumi:"deployedModelId"`
	ModelName       string `pulumi:"modelName"`
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
			ID: fmt.Sprintf("%s-%s-%s", req.Inputs.ProjectID, req.Inputs.Region, req.Name),
		}, nil
	}

	modelClientFactory := v.getModelClientFactory()

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

	state.ModelName = modelName
	state.CreateTime = time.Now().Format(time.RFC3339)

	// Only deploy to endpoint if endpoint deployment is configured
	if isEndpointDeploymentEnabled(req.Inputs) {
		endpointClientFactory := v.getEndpointClientFactory()
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

		// Create the model deployment service
		deployer := services.NewVertexModelDeploy(ctx, endpointClient, req.Inputs.ProjectID, req.Inputs.Region)
		defer func() {
			if closeErr := deployer.Close(); closeErr != nil {
				log.Printf("failed to close endpoint model deployment service: %v", closeErr)
			}
		}()

		// Deploy the model to the endpoint
		endpointConfig := convertEndpointDeploymentArgs(req.Inputs.EndpointModelDeployment)
		deployedModelID, err := deployer.Deploy(
			ctx,
			modelName,
			req.Name,
			req.Inputs.ServiceAccount,
			endpointConfig,
		)
		if err != nil {
			return infer.CreateResponse[VertexModelDeploymentState]{},
				fmt.Errorf("failed to deploy model: %w", err)
		}

		state.DeployedModelID = deployedModelID
		state.EndpointName = req.Inputs.EndpointModelDeployment.EndpointID
	}

	resourceID := fmt.Sprintf("%s-%s-%s", req.Inputs.ProjectID, req.Inputs.Region, req.Name)

	return infer.CreateResponse[VertexModelDeploymentState]{
		ID:     resourceID,
		Output: state,
	}, nil
}

// Delete implements the deletion logic
func (v VertexModelDeployment) Delete(
	ctx context.Context,
	req infer.DeleteRequest[VertexModelDeploymentState],
) (infer.DeleteResponse, error) {

	// Only undeploy from endpoint if the model was deployed to an endpoint
	if req.State.DeployedModelID != "" && req.State.EndpointName != "" {
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
				req.State.ProjectID, req.State.Region, req.State.EndpointName),
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
			ModelName:                 req.State.ModelName,
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

	// Only check endpoint deployment if the model was deployed to an endpoint
	if req.State.DeployedModelID == "" || req.State.EndpointName == "" {
		return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, nil
	}

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
			req.State.ProjectID, req.State.Region, req.State.EndpointName),
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

// testFactoryRegistry holds test factories for dependency injection during testing
var testFactoryRegistry struct {
	modelClientFactory    services.ModelClientFactory
	endpointClientFactory services.EndpointClientFactory
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

// convertEndpointDeploymentArgs converts EndpointModelDeploymentArgs to services.EndpointModelDeploymentConfig
func convertEndpointDeploymentArgs(args EndpointModelDeploymentArgs) services.EndpointModelDeploymentConfig {
	return services.EndpointModelDeploymentConfig{
		EndpointID:     args.EndpointID,
		MachineType:    args.MachineType,
		MinReplicas:    safeIntToInt32(args.MinReplicas),
		MaxReplicas:    safeIntToInt32(args.MaxReplicas),
		TrafficPercent: safeIntToInt32(args.TrafficPercent),
	}
}

// isEndpointDeploymentEnabled checks if endpoint deployment is configured
func isEndpointDeploymentEnabled(args VertexModelDeploymentArgs) bool {
	return args.EndpointModelDeployment.EndpointID != ""
}
