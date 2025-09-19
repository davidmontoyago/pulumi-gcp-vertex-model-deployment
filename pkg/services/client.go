package services

import (
	"context"
	"fmt"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	gax "github.com/googleapis/gax-go/v2"
	"google.golang.org/api/option"
)

// VertexModelClient interface defines operations for uploading models.
type VertexModelClient interface {
	UploadModel(ctx context.Context, req *aiplatformpb.UploadModelRequest, opts ...gax.CallOption) (*aiplatform.UploadModelOperation, error)
	DeleteModel(ctx context.Context, req *aiplatformpb.DeleteModelRequest, opts ...gax.CallOption) (*aiplatform.DeleteModelOperation, error)
	GetModel(ctx context.Context, req *aiplatformpb.GetModelRequest, opts ...gax.CallOption) (*aiplatformpb.Model, error)
	UpdateModel(context.Context, *aiplatformpb.UpdateModelRequest, ...gax.CallOption) (*aiplatformpb.Model, error)
	Close() error
}

// VertexEndpointClient interface defines operations for deploying models.
type VertexEndpointClient interface {
	DeployModel(ctx context.Context, req *aiplatformpb.DeployModelRequest, opts ...gax.CallOption) (*aiplatform.DeployModelOperation, error)
	UndeployModel(ctx context.Context, req *aiplatformpb.UndeployModelRequest, opts ...gax.CallOption) (*aiplatform.UndeployModelOperation, error)
	GetEndpoint(ctx context.Context, req *aiplatformpb.GetEndpointRequest, opts ...gax.CallOption) (*aiplatformpb.Endpoint, error)
	Close() error
}

// ModelClientFactory function type for creating model clients
type ModelClientFactory func(ctx context.Context, region string) (VertexModelClient, error)

// EndpointClientFactory function type for creating endpoint clients
type EndpointClientFactory func(ctx context.Context, region string) (VertexEndpointClient, error)

// DefaultModelClientFactory creates the production GCP model client.
//
//nolint:ireturn // Returning interface for testability
func DefaultModelClientFactory(ctx context.Context, region string) (VertexModelClient, error) {
	// Regional endpoints require regional endpoints
	apiEndpoint := fmt.Sprintf("%s-aiplatform.googleapis.com:443", region)
	clientEndpointOpt := option.WithEndpoint(apiEndpoint)

	// Create model client
	modelClient, err := aiplatform.NewModelClient(ctx, clientEndpointOpt)
	if err != nil {
		return nil, fmt.Errorf("failed to create model client: %w", err)
	}

	return modelClient, nil
}

// DefaultEndpointClientFactory creates the production GCP endpoint client.
//
//nolint:ireturn // Returning interface for testability
func DefaultEndpointClientFactory(ctx context.Context, region string) (VertexEndpointClient, error) {
	// Regional endpoints require regional endpoints
	apiEndpoint := fmt.Sprintf("%s-aiplatform.googleapis.com:443", region)
	clientEndpointOpt := option.WithEndpoint(apiEndpoint)

	// Create endpoint client
	endpointClient, err := aiplatform.NewEndpointClient(ctx, clientEndpointOpt)
	if err != nil {
		return nil, fmt.Errorf("failed to create endpoint client: %w", err)
	}

	return endpointClient, nil
}
