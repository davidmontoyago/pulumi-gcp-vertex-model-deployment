package resources

import (
	"context"
	"fmt"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/pkg/services"
	"github.com/pulumi/pulumi-go-provider/infer"
	"google.golang.org/protobuf/types/known/fieldmaskpb"
)

func updateRegistryModel(ctx context.Context, req infer.UpdateRequest[VertexModelDeploymentArgs, VertexModelDeploymentState], modelClient services.VertexModelClient, updatePaths []string) (*aiplatformpb.Model, error) {
	predictionSchema := &aiplatformpb.PredictSchemata{}

	if req.Inputs.ModelPredictionInputSchemaURI != "" {
		predictionSchema.InstanceSchemaUri = req.Inputs.ModelPredictionInputSchemaURI
	}
	if req.Inputs.ModelPredictionOutputSchemaURI != "" {
		predictionSchema.PredictionSchemaUri = req.Inputs.ModelPredictionOutputSchemaURI
	}
	if req.Inputs.ModelPredictionBehaviorSchemaURI != "" {
		predictionSchema.ParametersSchemaUri = req.Inputs.ModelPredictionBehaviorSchemaURI
	}

	// Build container spec (consistent with model creation)
	containerSpec := &aiplatformpb.ModelContainerSpec{
		ImageUri: req.Inputs.ModelImageURL,
		Args:     req.Inputs.Args,
	}

	// Add environment variables
	envVars := []*aiplatformpb.EnvVar{}
	for name, value := range req.Inputs.EnvVars {
		envVars = append(envVars, &aiplatformpb.EnvVar{
			Name:  name,
			Value: value,
		})
	}
	containerSpec.Env = envVars

	// Add port configuration
	modelServerPort := req.Inputs.Port
	if modelServerPort == 0 {
		modelServerPort = 8080
	}
	containerSpec.Ports = []*aiplatformpb.Port{
		{
			ContainerPort: modelServerPort,
		},
	}
	if req.Inputs.PredictRoute != "" {
		containerSpec.PredictRoute = req.Inputs.PredictRoute
	}
	if req.Inputs.HealthRoute != "" {
		containerSpec.HealthRoute = req.Inputs.HealthRoute
	}

	updatedModel, err := modelClient.UpdateModel(ctx, &aiplatformpb.UpdateModelRequest{
		Model: &aiplatformpb.Model{
			Name:            req.State.ModelName,
			DisplayName:     req.ID,                                           // Use resource ID as display name (consistent with creation)
			Description:     "Uploaded model for " + req.Inputs.ModelImageURL, // Consistent with creation
			Labels:          req.Inputs.Labels,
			ArtifactUri:     req.Inputs.ModelArtifactsBucketURI,
			ContainerSpec:   containerSpec,
			PredictSchemata: predictionSchema,
		},
		UpdateMask: &fieldmaskpb.FieldMask{
			Paths: updatePaths,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to update model: %w", err)
	}

	return updatedModel, nil
}

func setModelStateUpdates(req infer.UpdateRequest[VertexModelDeploymentArgs, VertexModelDeploymentState], updatedModel *aiplatformpb.Model) VertexModelDeploymentState {
	updatedState := VertexModelDeploymentState{
		VertexModelDeploymentArgs: req.Inputs,
		DeployedModelID:           req.State.DeployedModelID,
		ModelName:                 updatedModel.Name,
		EndpointName:              req.State.EndpointName,
		CreateTime:                req.State.CreateTime,
	}

	// Update state fields from the updated model response
	if updatedModel.Labels != nil {
		updatedState.Labels = updatedModel.Labels
	}

	// Update ModelArtifactsBucketURI from the model response
	if updatedModel.ArtifactUri != "" {
		updatedState.ModelArtifactsBucketURI = updatedModel.ArtifactUri
	}

	// Update container spec fields if available
	if updatedModel.ContainerSpec != nil {
		// ImageUri is immutable, requires replacement.
		// See: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/ModelContainerSpec

		if updatedModel.ContainerSpec.PredictRoute != "" {
			updatedState.PredictRoute = updatedModel.ContainerSpec.PredictRoute
		}
		if updatedModel.ContainerSpec.HealthRoute != "" {
			updatedState.HealthRoute = updatedModel.ContainerSpec.HealthRoute
		}

		// Update container args
		if len(updatedModel.ContainerSpec.Args) > 0 {
			updatedState.Args = updatedModel.ContainerSpec.Args
		}

		// Update environment variables
		if len(updatedModel.ContainerSpec.Env) > 0 {
			updatedState.EnvVars = make(map[string]string)
			for _, env := range updatedModel.ContainerSpec.Env {
				updatedState.EnvVars[env.Name] = env.Value
			}
		}

		// Update port
		if len(updatedModel.ContainerSpec.Ports) > 0 {
			updatedState.Port = updatedModel.ContainerSpec.Ports[0].ContainerPort
		}
	}

	// Update predict schemata fields if available
	if updatedModel.PredictSchemata != nil {
		if updatedModel.PredictSchemata.InstanceSchemaUri != "" {
			updatedState.ModelPredictionInputSchemaURI = updatedModel.PredictSchemata.InstanceSchemaUri
		}
		if updatedModel.PredictSchemata.PredictionSchemaUri != "" {
			updatedState.ModelPredictionOutputSchemaURI = updatedModel.PredictSchemata.PredictionSchemaUri
		}
		if updatedModel.PredictSchemata.ParametersSchemaUri != "" {
			updatedState.ModelPredictionBehaviorSchemaURI = updatedModel.PredictSchemata.ParametersSchemaUri
		}
	}

	return updatedState
}

func collectUpdates(req infer.UpdateRequest[VertexModelDeploymentArgs, VertexModelDeploymentState]) (bool, []string) {
	needsUpdate := false
	updatePathsMap := make(map[string]bool)

	// Check if labels have changed
	if !mapsEqual(req.Inputs.Labels, req.State.Labels) {
		needsUpdate = true
		updatePathsMap["labels"] = true
	}

	// Check if ModelImageURL has changed (affects description and container spec)
	if req.Inputs.ModelImageURL != req.State.ModelImageURL {
		needsUpdate = true
		updatePathsMap["description"] = true
		updatePathsMap["container_spec"] = true
	}

	// Check if ModelArtifactsBucketURI has changed
	if req.Inputs.ModelArtifactsBucketURI != req.State.ModelArtifactsBucketURI {
		needsUpdate = true
		updatePathsMap["artifact_uri"] = true
	}

	// Check if prediction schema URIs have changed
	if req.Inputs.ModelPredictionInputSchemaURI != req.State.ModelPredictionInputSchemaURI ||
		req.Inputs.ModelPredictionOutputSchemaURI != req.State.ModelPredictionOutputSchemaURI ||
		req.Inputs.ModelPredictionBehaviorSchemaURI != req.State.ModelPredictionBehaviorSchemaURI {
		needsUpdate = true
		updatePathsMap["predict_schemata"] = true
	}

	// Check if container routes have changed
	if req.Inputs.PredictRoute != req.State.PredictRoute ||
		req.Inputs.HealthRoute != req.State.HealthRoute {
		needsUpdate = true
		updatePathsMap["container_spec"] = true
	}

	// Check if container args have changed
	if !slicesEqual(req.Inputs.Args, req.State.Args) {
		needsUpdate = true
		updatePathsMap["container_spec"] = true
	}

	// Check if environment variables have changed
	if !mapsEqual(req.Inputs.EnvVars, req.State.EnvVars) {
		needsUpdate = true
		updatePathsMap["container_spec"] = true
	}

	// Check if port has changed
	if req.Inputs.Port != req.State.Port {
		needsUpdate = true
		updatePathsMap["container_spec"] = true
	}

	// Convert map to slice
	updatePaths := make([]string, 0, len(updatePathsMap))
	for path := range updatePathsMap {
		updatePaths = append(updatePaths, path)
	}

	return needsUpdate, updatePaths
}

// slicesEqual compares two string slices for equality
func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}

	return true
}
