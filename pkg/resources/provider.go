// Package resources provides Pulumi resource implementations for GCP Vertex AI model upload and deployment.
package resources

import (
	"context"
	"fmt"
	"log"
	"time"

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
		PredictRoute:                     req.Inputs.PredictRoute,
		HealthRoute:                      req.Inputs.HealthRoute,
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
		endpointConfig := toEndpointDeploymentConfig(req.Inputs.EndpointModelDeployment)
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
			return infer.DeleteResponse{}, fmt.Errorf("failed to create endpoint client for undeployment: %w", err)
		}
		defer func() {
			if closeErr := endpointClient.Close(); closeErr != nil {
				log.Printf("failed to close endpoint client for undeployment: %v", closeErr)
			}
		}()

		undeployer := services.NewVertexModelUndeploy(ctx, endpointClient, req.State.ProjectID, req.State.Region)
		err = undeployer.Undeploy(ctx, req.State.EndpointName, req.State.DeployedModelID)
		if err != nil {
			return infer.DeleteResponse{}, fmt.Errorf("failed to undeploy model: %w", err)
		}
	}

	// After undeploying, delete the model
	modelClientFactory := v.getModelClientFactory()
	modelClient, err := modelClientFactory(ctx, req.State.Region)
	if err != nil {
		return infer.DeleteResponse{}, fmt.Errorf("failed to create model client: %w", err)
	}
	defer func() {
		if closeErr := modelClient.Close(); closeErr != nil {
			log.Printf("failed to close model client: %v", closeErr)
		}
	}()

	deleter := services.NewVertexModelDelete(ctx, modelClient, req.State.ModelName)
	err = deleter.Delete(ctx, req.State.ModelName)
	if err != nil {
		return infer.DeleteResponse{}, fmt.Errorf("failed to delete model: %w", err)
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

	// Create a copy of the current state to modify
	state := req.State

	if req.State.ModelName != "" {
		// Read the model from the registry

		modelClientFactory := v.getModelClientFactory()
		modelClient, err := modelClientFactory(ctx, req.State.Region)
		if err != nil {
			return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, fmt.Errorf("failed to create model client: %w", err)
		}
		defer func() {
			if closeErr := modelClient.Close(); closeErr != nil {
				log.Printf("failed to close model client: %v", closeErr)
			}
		}()

		err = readRegistryModel(ctx, modelClient, req, &state)
		if err != nil {
			return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, fmt.Errorf("failed to read model from registry: %w", err)
		}
	}

	if req.State.DeployedModelID != "" && req.State.EndpointName != "" {
		// Read the endpoint if model is deployed to an endpoint

		endpointClientFactory := v.getEndpointClientFactory()
		endpointClient, err := endpointClientFactory(ctx, req.State.Region)
		if err != nil {
			return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, fmt.Errorf("failed to create endpoint client: %w", err)
		}
		defer func() {
			if closeErr := endpointClient.Close(); closeErr != nil {
				log.Printf("failed to close endpoint client: %v", closeErr)
			}
		}()

		err = readEndpointModel(ctx, endpointClient, req, &state)
		if err != nil {
			return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, fmt.Errorf("failed to read model endpoint: %w", err)
		}
	}

	return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{
		Inputs: req.Inputs,
		State:  state,
	}, nil
}

func readEndpointModel(ctx context.Context,
	endpointClient services.VertexEndpointClient,
	req infer.ReadRequest[VertexModelDeploymentArgs, VertexModelDeploymentState],
	state *VertexModelDeploymentState) error {

	endpointGetter := services.NewVertexEndpointModelGetter(endpointClient, req.State.ProjectID, req.State.Region)
	endpoint, foundDeployedModel, err := endpointGetter.Get(ctx, req.State.EndpointName, req.State.DeployedModelID)
	if err != nil {
		return err
	}

	if foundDeployedModel == nil {
		// Model is no longer deployed - return empty response to indicate resource doesn't exist
		return nil
	}

	// Update state with current endpoint and deployed model information
	state.EndpointName = endpoint.Name
	state.DeployedModelID = foundDeployedModel.Id

	// Update endpoint deployment configuration with current values if available
	if state.EndpointModelDeployment == nil {
		return nil
	}

	// Extract current deployment configuration from the deployed model
	if dedicatedResources := foundDeployedModel.GetDedicatedResources(); dedicatedResources != nil {
		if machineSpec := dedicatedResources.MachineSpec; machineSpec != nil {
			state.EndpointModelDeployment.MachineType = machineSpec.MachineType
		}
		state.EndpointModelDeployment.MinReplicas = int(dedicatedResources.MinReplicaCount)
		state.EndpointModelDeployment.MaxReplicas = int(dedicatedResources.MaxReplicaCount)
	}

	// Update traffic percentage from endpoint's traffic split if available
	if endpoint.TrafficSplit != nil {
		if trafficPercent, exists := endpoint.TrafficSplit[foundDeployedModel.Id]; exists {
			state.EndpointModelDeployment.TrafficPercent = int(trafficPercent)
		}
	}

	return nil
}

func readRegistryModel(ctx context.Context,
	modelClient services.VertexModelClient,
	req infer.ReadRequest[VertexModelDeploymentArgs,
		VertexModelDeploymentState], state *VertexModelDeploymentState) error {

	modelGetter := services.NewVertexModelGet(ctx, modelClient, req.State.ModelName)
	model, err := modelGetter.Get(ctx, req.State.ModelName)
	if err != nil {
		return fmt.Errorf("failed to get model: %w", err)
	}

	// Update state with current model values
	state.ModelName = model.Name
	state.ModelArtifactsBucketURI = model.ArtifactUri
	state.Labels = model.Labels

	// Safely access ContainerSpec fields
	if model.ContainerSpec != nil {
		state.ModelImageURL = model.ContainerSpec.ImageUri
		state.PredictRoute = model.ContainerSpec.PredictRoute
		state.HealthRoute = model.ContainerSpec.HealthRoute
	}

	// Safely access PredictSchemata fields
	if model.PredictSchemata != nil {
		state.ModelPredictionInputSchemaURI = model.PredictSchemata.InstanceSchemaUri
		state.ModelPredictionOutputSchemaURI = model.PredictSchemata.PredictionSchemaUri
	}

	return nil
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

// toEndpointDeploymentConfig converts EndpointModelDeploymentArgs to services.EndpointModelDeploymentConfig
func toEndpointDeploymentConfig(args *EndpointModelDeploymentArgs) services.EndpointModelDeploymentConfig {
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
	return args.EndpointModelDeployment != nil && args.EndpointModelDeployment.EndpointID != ""
}
