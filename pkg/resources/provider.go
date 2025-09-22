// Package resources provides Pulumi resource implementations for GCP Vertex AI model upload and deployment.
package resources

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"strings"
	"time"

	p "github.com/pulumi/pulumi-go-provider"
	provider "github.com/pulumi/pulumi-go-provider"
	"github.com/pulumi/pulumi-go-provider/infer"

	"github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/pkg/services"
)

// VertexModelDeployment represents a Pulumi resource for deploying models to Vertex AI endpoints.
type VertexModelDeployment struct{}

// Compile-time interface compliance checks
var _ infer.CustomCreate[VertexModelDeploymentArgs, VertexModelDeploymentState] = (*VertexModelDeployment)(nil)
var _ infer.CustomRead[VertexModelDeploymentArgs, VertexModelDeploymentState] = (*VertexModelDeployment)(nil)
var _ infer.CustomUpdate[VertexModelDeploymentArgs, VertexModelDeploymentState] = (*VertexModelDeployment)(nil)
var _ infer.CustomDelete[VertexModelDeploymentState] = (*VertexModelDeployment)(nil)
var _ infer.CustomDiff[VertexModelDeploymentArgs, VertexModelDeploymentState] = (*VertexModelDeployment)(nil)

// Annotate provides metadata and descriptions for the VertexModelDeployment resource.
func (VertexModelDeployment) Annotate(annotator infer.Annotator) {
	annotator.Describe(&VertexModelDeployment{}, "Deploys a model to a Vertex AI endpoint")
}

// VertexModelDeploymentState represents the state of a deployed Vertex AI model.
type VertexModelDeploymentState struct {
	VertexModelDeploymentArgs
	ModelName       string `pulumi:"modelName"`
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

	resourceID := fmt.Sprintf("%s-%s-%s", req.Inputs.ProjectID, req.Inputs.Region, req.Name)

	if req.DryRun {
		return infer.CreateResponse[VertexModelDeploymentState]{
			ID: resourceID,
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

	deleter := services.NewVertexModelDelete(ctx, modelClient)
	err = deleter.Delete(ctx, req.State.ModelName)
	if err != nil {
		return infer.DeleteResponse{}, fmt.Errorf("failed to delete model: %w", err)
	}

	return infer.DeleteResponse{}, nil
}

// Update implements the update logic
func (v VertexModelDeployment) Update(
	ctx context.Context,
	req infer.UpdateRequest[VertexModelDeploymentArgs, VertexModelDeploymentState],
) (infer.UpdateResponse[VertexModelDeploymentState], error) {

	// Handle dry run - return the updated state without actually making changes
	if req.DryRun {
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

	// Check if any model properties actually need updating to avoid unnecessary API calls
	needsUpdate, updatePaths := collectUpdates(req)

	// If no updates are needed, return current state
	if !needsUpdate {
		return infer.UpdateResponse[VertexModelDeploymentState]{
			Output: req.State,
		}, nil
	}

	modelClientFactory := v.getModelClientFactory()
	modelClient, err := modelClientFactory(ctx, req.State.Region)
	if err != nil {
		return infer.UpdateResponse[VertexModelDeploymentState]{}, fmt.Errorf("failed to create model client: %w", err)
	}
	defer func() {
		if closeErr := modelClient.Close(); closeErr != nil {
			log.Printf("failed to close model client: %v", closeErr)
		}
	}()

	// Build prediction schema (consistent with model creation)
	updatedModel, err := updateRegistryModel(ctx, req, modelClient, updatePaths)
	if err != nil {
		return infer.UpdateResponse[VertexModelDeploymentState]{}, fmt.Errorf("failed to update registry model: %w", err)
	}

	// Create updated state with the response from the API
	updatedState := setModelStateUpdates(req, updatedModel)

	// TODO update endpoint model deployment

	return infer.UpdateResponse[VertexModelDeploymentState]{
		Output: updatedState,
	}, nil
}

// Read implements the read logic for drift detection
func (v VertexModelDeployment) Read(
	ctx context.Context,
	req infer.ReadRequest[VertexModelDeploymentArgs, VertexModelDeploymentState],
) (infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState], error) {

	// Validate that we have the minimum required information to read the resource
	if req.ID == "" {
		return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{},
			fmt.Errorf("resource ID is required for read operation")
	}

	// Create a copy of the current state to modify
	state := req.State

	// Always attempt to read the model if we have a model name
	if req.State.ModelName != "" {
		modelClientFactory := v.getModelClientFactory()
		modelClient, err := modelClientFactory(ctx, req.State.Region)
		if err != nil {
			// If we can't create the client, don't assume the resource is gone
			return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{},
				fmt.Errorf("failed to create model client: %w", err)
		}
		defer modelClient.Close()

		err = readRegistryModel(ctx, modelClient, req, &state)
		if err != nil {
			// Check if this is a "not found" error specifically
			// If the model truly doesn't exist, return empty response
			// Otherwise, return the error
			if isResourceNotFoundError(err) {
				slog.Warn("Model no longer exists", "modelName", req.State.ModelName)

				return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, nil
			}

			return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{},
				fmt.Errorf("failed to read model from registry: %w", err)
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
		ID:     req.ID,
		Inputs: req.Inputs,
		State:  state,
	}, nil
}

// Diff implements the diff logic to control what changes require replacement vs update
func (v VertexModelDeployment) Diff(
	ctx context.Context,
	req infer.DiffRequest[VertexModelDeploymentArgs, VertexModelDeploymentState],
) (p.DiffResponse, error) {

	diff := p.DiffResponse{
		HasChanges:   false,
		DetailedDiff: make(map[string]provider.PropertyDiff),
	}

	// Properties that require replacement (immutable)
	immutableProperties := map[string]bool{
		"projectId": true,
		"region":    true,
	}

	// Check ProjectID
	if req.Inputs.ProjectID != req.State.ProjectID {
		diff.HasChanges = true
		if immutableProperties["projectId"] {
			diff.DetailedDiff["projectId"] = p.PropertyDiff{
				Kind:      p.UpdateReplace,
				InputDiff: true,
			}
		} else {
			diff.DetailedDiff["projectId"] = p.PropertyDiff{
				Kind:      p.Update,
				InputDiff: true,
			}
		}
	}

	// Check Region
	if req.Inputs.Region != req.State.Region {
		diff.HasChanges = true
		if immutableProperties["region"] {
			diff.DetailedDiff["region"] = p.PropertyDiff{
				Kind:      p.UpdateReplace,
				InputDiff: true,
			}
		} else {
			diff.DetailedDiff["region"] = p.PropertyDiff{
				Kind:      p.Update,
				InputDiff: true,
			}
		}
	}

	// Check ModelImageURL - this can be updated
	if req.Inputs.ModelImageURL != req.State.ModelImageURL {
		diff.HasChanges = true
		diff.DetailedDiff["modelImageUrl"] = p.PropertyDiff{
			// Image URL is immutable, requires replacement of the model resource
			// See: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/ModelContainerSpec
			Kind:      p.UpdateReplace,
			InputDiff: true,
		}
	}

	// Check ModelArtifactsBucketURI - this can be updated
	if req.Inputs.ModelArtifactsBucketURI != req.State.ModelArtifactsBucketURI {
		diff.HasChanges = true
		diff.DetailedDiff["modelArtifactsBucketUri"] = p.PropertyDiff{
			Kind:      p.Update,
			InputDiff: true,
		}
	}

	// Prediction schema URIs are immutable
	// See: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/PredictSchemata

	// Check service account - this can be updated
	if req.Inputs.ServiceAccount != req.State.ServiceAccount {
		diff.HasChanges = true
		diff.DetailedDiff["serviceAccount"] = p.PropertyDiff{
			Kind:      p.Update,
			InputDiff: true,
		}
	}

	// Check route configurations - these can be updated
	if req.Inputs.PredictRoute != req.State.PredictRoute {
		diff.HasChanges = true
		diff.DetailedDiff["predictRoute"] = p.PropertyDiff{
			Kind:      p.Update,
			InputDiff: true,
		}
	}

	if req.Inputs.HealthRoute != req.State.HealthRoute {
		diff.HasChanges = true
		diff.DetailedDiff["healthRoute"] = p.PropertyDiff{
			Kind:      p.Update,
			InputDiff: true,
		}
	}

	// Check labels - these can be updated
	if !mapsEqual(req.Inputs.Labels, req.State.Labels) {
		diff.HasChanges = true
		diff.DetailedDiff["labels"] = p.PropertyDiff{
			Kind:      p.Update,
			InputDiff: true,
		}
	}

	// Check endpoint model deployment configuration - this can be updated
	if !endpointDeploymentEqual(req.Inputs.EndpointModelDeployment, req.State.EndpointModelDeployment) {
		diff.HasChanges = true
		diff.DetailedDiff["endpointModelDeployment"] = p.PropertyDiff{
			Kind:      p.Update,
			InputDiff: true,
		}
	}

	return diff, nil
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

	// Read schema URIs if available.
	if model.PredictSchemata != nil {
		// These are immutable and should be ignored during updates.
		// URI given on output will be immutable and probably different,
		// including the URI scheme, than the one given on input.
		// See: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/PredictSchemata
		state.ModelPredictionInputSchemaURI = model.PredictSchemata.InstanceSchemaUri
		state.ModelPredictionOutputSchemaURI = model.PredictSchemata.PredictionSchemaUri
		state.ModelPredictionBehaviorSchemaURI = model.PredictSchemata.ParametersSchemaUri
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

// mapsEqual checks if two string maps are equal
func mapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}

	return true
}

// endpointDeploymentEqual checks if two EndpointModelDeploymentArgs are equal
func endpointDeploymentEqual(a, b *EndpointModelDeploymentArgs) bool {
	// Both nil
	if a == nil && b == nil {
		return true
	}
	// One nil, one not nil
	if a == nil || b == nil {
		return false
	}
	// Compare all fields
	return a.EndpointID == b.EndpointID &&
		a.MachineType == b.MachineType &&
		a.MinReplicas == b.MinReplicas &&
		a.MaxReplicas == b.MaxReplicas &&
		a.TrafficPercent == b.TrafficPercent
}

// isResourceNotFoundError detects if the error indicates the resource doesn't exist
func isResourceNotFoundError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()

	return strings.Contains(errStr, "not found") ||
		strings.Contains(errStr, "does not exist") ||
		strings.Contains(errStr, "404")
}
