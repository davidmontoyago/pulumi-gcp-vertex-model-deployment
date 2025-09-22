package resources

import (
	"context"
	"fmt"

	"github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/pkg/services"
	"github.com/pulumi/pulumi-go-provider/infer"
)

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
