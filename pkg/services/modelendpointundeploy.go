package services

import (
	"context"
	"fmt"
	"log"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
)

// ModelUndeployer allows undeploying models from Vertex AI endpoints.
type ModelUndeployer interface {
	Undeploy(ctx context.Context, endpointName, deployedModelID string) error
	Close() error
}

// VertexModelUndeploy implements the ModelUndeployer interface for Vertex AI.
type VertexModelUndeploy struct {
	endpointClient VertexEndpointClient
}

// NewVertexModelUndeploy creates a new VertexModelUndeploy with the provided endpoint client.
func NewVertexModelUndeploy(_ context.Context, endpointClient VertexEndpointClient, _, _ string) *VertexModelUndeploy {
	return &VertexModelUndeploy{
		endpointClient: endpointClient,
	}
}

// Undeploy undeploys a model from an endpoint.
func (u *VertexModelUndeploy) Undeploy(ctx context.Context, endpointName, deployedModelID string) error {
	undeployReq := &aiplatformpb.UndeployModelRequest{
		Endpoint:        endpointName, // endpointName is already fully qualified
		DeployedModelId: deployedModelID,
	}

	undeployOperation, err := u.endpointClient.UndeployModel(ctx, undeployReq)
	if err != nil {
		return fmt.Errorf("failed to undeploy model: %w", err)
	}

	if undeployOperation == nil {
		log.Printf("Warning: model undeploy operation is nil?!? This must be a mocked client. Logging error and moving on.")
	} else {
		_, err = undeployOperation.Wait(ctx)
		if err != nil {
			return fmt.Errorf("failed to wait for undeployment: %w", err)
		}
	}

	return nil
}
