package services

import (
	"context"
	"fmt"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
)

// EndpointModelGetter allows getting endpoints and their deployed models from the registry.
type EndpointModelGetter interface {
	Get(ctx context.Context, endpointName, deployedModelID string) (*aiplatformpb.Endpoint, *aiplatformpb.DeployedModel, error)
	Close() error
}

// VertexEndpointModelGetter implements the EndpointModelGetter interface for Vertex AI.
type VertexEndpointModelGetter struct {
	endpointClient VertexEndpointClient
	projectID      string
	region         string
}

// NewVertexEndpointModelGetter creates a new VertexEndpointModelGetter with the provided endpoint client.
func NewVertexEndpointModelGetter(endpointClient VertexEndpointClient, projectID, region string) *VertexEndpointModelGetter {
	return &VertexEndpointModelGetter{
		endpointClient: endpointClient,
		projectID:      projectID,
		region:         region,
	}
}

// Get retrieves an endpoint and finds the specified deployed model within it.
// Returns the endpoint, the deployed model (if found), and any error.
func (g *VertexEndpointModelGetter) Get(ctx context.Context, endpointName, deployedModelID string) (*aiplatformpb.Endpoint, *aiplatformpb.DeployedModel, error) {
	getReq := &aiplatformpb.GetEndpointRequest{
		Name: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			g.projectID, g.region, endpointName),
	}

	endpoint, err := g.endpointClient.GetEndpoint(ctx, getReq)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get endpoint: %w", err)
	}

	// Verify the deployed model still exists and update its properties
	var foundDeployedModel *aiplatformpb.DeployedModel
	for _, deployedModel := range endpoint.DeployedModels {
		if deployedModel.Id == deployedModelID {
			foundDeployedModel = deployedModel

			break
		}
	}

	return endpoint, foundDeployedModel, nil
}

// Close closes the endpoint client.
func (g *VertexEndpointModelGetter) Close() error {
	return g.endpointClient.Close()
}
