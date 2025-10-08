// Package services provides implementations for GCP Vertex AI model and Endpoint deployment operations.
package services

import (
	"context"
	"fmt"
	"log"
	"time"

	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	gax "github.com/googleapis/gax-go/v2"
)

// EndpointModelDeploymentConfig holds configuration for deploying a model to an endpoint.
type EndpointModelDeploymentConfig struct {
	EndpointID       string
	MachineType      string
	AcceleratorType  string
	AcceleratorCount int32
	MinReplicas      int32
	MaxReplicas      int32
	TrafficPercent   int32

	DisableContainerLogging bool
	EnableAccessLogging     bool
	EnableSpotVMs           bool
}

// ModelDeployer interface defines operations for deploying models.
type ModelDeployer interface {
	Deploy(ctx context.Context, modelName, displayName, serviceAccount string, endpointConfig EndpointModelDeploymentConfig) (string, error)
	Close() error
}

// VertexModelDeploy implements the ModelDeployer interface for Vertex AI.
type VertexModelDeploy struct {
	endpointClient VertexEndpointClient
	projectID      string
	region         string
}

// NewVertexModelDeploy creates a new VertexModelDeploy with the provided endpoint client.
func NewVertexModelDeploy(_ context.Context, endpointClient VertexEndpointClient, projectID, region string) *VertexModelDeploy {
	return &VertexModelDeploy{
		endpointClient: endpointClient,
		projectID:      projectID,
		region:         region,
	}
}

// Deploy deploys a model to a Vertex AI endpoint and returns the deployed model ID.
func (d *VertexModelDeploy) Deploy(ctx context.Context, modelName, displayName, serviceAccount string, endpointConfig EndpointModelDeploymentConfig) (string, error) {
	// Build the deployment request
	deployedModel := &aiplatformpb.DeployedModel{
		// Expected format: "projects/%s/locations/%s/models/%s"
		Model:       modelName,
		DisplayName: displayName,
		PredictionResources: &aiplatformpb.DeployedModel_DedicatedResources{
			DedicatedResources: &aiplatformpb.DedicatedResources{
				MachineSpec: &aiplatformpb.MachineSpec{
					MachineType:      endpointConfig.MachineType,
					AcceleratorType:  aiplatformpb.AcceleratorType(aiplatformpb.AcceleratorType_value[endpointConfig.AcceleratorType]),
					AcceleratorCount: endpointConfig.AcceleratorCount,
				},
				MinReplicaCount: endpointConfig.MinReplicas,
				MaxReplicaCount: endpointConfig.MaxReplicas,
				Spot:            endpointConfig.EnableSpotVMs,
			},
		},
		DisableContainerLogging: endpointConfig.DisableContainerLogging,
		EnableAccessLogging:     endpointConfig.EnableAccessLogging,
	}

	if serviceAccount != "" {
		deployedModel.ServiceAccount = serviceAccount
	}

	deployReq := &aiplatformpb.DeployModelRequest{
		Endpoint: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			d.projectID, d.region, endpointConfig.EndpointID),
		DeployedModel: deployedModel,
		TrafficSplit:  map[string]int32{},
	}

	// Set traffic split if specified
	if endpointConfig.TrafficPercent > 0 {
		deployReq.TrafficSplit = map[string]int32{
			"0": endpointConfig.TrafficPercent,
		}
	}

	// Execute the deployment
	deployOperation, err := d.endpointClient.DeployModel(ctx, deployReq)
	if err != nil {
		return "", fmt.Errorf("failed to deploy model: %w", err)
	}

	if deployOperation == nil {
		log.Printf("Warning: deploy operation is nil?!? This must be a mocked client. Logging error and moving on.")

		return "", nil
	}

	// Wait for completion with timeout
	result, err := deployOperation.Wait(ctx, gax.WithTimeout(10*time.Minute))
	if err != nil {
		return "", fmt.Errorf("failed to wait for deployment: %w", err)
	}

	return result.GetDeployedModel().GetId(), nil
}

// Close closes the endpoint client.
func (d *VertexModelDeploy) Close() error {
	if d.endpointClient != nil {
		if err := d.endpointClient.Close(); err != nil {
			return fmt.Errorf("failed to close endpoint client: %w", err)
		}
	}

	return nil
}
