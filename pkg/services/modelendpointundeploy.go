package services

import (
	"context"
	"fmt"
	"log"
	"strings"

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
	projectID      string
	region         string
}

// NewVertexModelUndeploy creates a new VertexModelUndeploy with the provided endpoint client.
func NewVertexModelUndeploy(_ context.Context, endpointClient VertexEndpointClient, projectID, region string) *VertexModelUndeploy {
	return &VertexModelUndeploy{
		endpointClient: endpointClient,
		projectID:      projectID,
		region:         region,
	}
}

// Undeploy undeploys a model from an endpoint.
func (u *VertexModelUndeploy) Undeploy(ctx context.Context, endpointName, deployedModelID string) error {
	endpointFullName := endpointName
	if !strings.HasPrefix(endpointName, "projects/") {
		endpointFullName = fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			u.projectID, u.region, endpointName)
	}
	undeployReq := &aiplatformpb.UndeployModelRequest{
		Endpoint:        endpointFullName,
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
