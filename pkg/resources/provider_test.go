//nolint:testpackage // Using same package for better coverage tracking
package resources

import (
	"context"
	"testing"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	gax "github.com/googleapis/gax-go/v2"
	"github.com/pulumi/pulumi-go-provider/infer"

	"github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/pkg/services"
)

func TestVertexModelDeploymentCreate_ModelUploadAndDeployRequests(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Test inputs
	projectID := "test-project"
	region := "us-central1"
	endpointID := "test-endpoint"
	modelImageURL := "gcr.io/test-project/custom-model:latest"
	modelArtifactsBucketURI := "gs://test-bucket/model-artifacts/"
	modelPredictionInputSchemaURI := "gs://test-bucket/schemas/input_schema.json"
	modelPredictionOutputSchemaURI := "gs://test-bucket/schemas/output_schema.json"
	modelPredictionBehaviorSchemaURI := "gs://test-bucket/schemas/behavior_schema.json"
	machineType := "n1-standard-2"
	minReplicas := 1
	maxReplicas := 3
	trafficPercent := 100
	serviceAccount := "test-service-account@test-project.iam.gserviceaccount.com"
	resourceName := "test-model-deployment"

	// Variables to capture request parameters
	var capturedUploadRequest *aiplatformpb.UploadModelRequest
	var capturedDeployRequest *aiplatformpb.DeployModelRequest

	req := infer.CreateRequest[VertexModelDeploymentArgs]{
		Name:   resourceName,
		DryRun: false, // Explicitly set to false to ensure real execution
		Inputs: VertexModelDeploymentArgs{
			ProjectID:                        projectID,
			Region:                           region,
			EndpointID:                       endpointID,
			ModelImageURL:                    modelImageURL,
			ModelArtifactsBucketURI:          modelArtifactsBucketURI,
			ModelPredictionInputSchemaURI:    modelPredictionInputSchemaURI,
			ModelPredictionOutputSchemaURI:   modelPredictionOutputSchemaURI,
			ModelPredictionBehaviorSchemaURI: modelPredictionBehaviorSchemaURI,
			MachineType:                      machineType,
			MinReplicas:                      minReplicas,
			MaxReplicas:                      maxReplicas,
			TrafficPercent:                   trafficPercent,
			ServiceAccount:                   serviceAccount,
			Labels:                           map[string]string{"env": "test", "component": "ml"},
		},
	}

	modelClientFactory := MockModelClientFactory(&MockModelClient{
		UploadModelFunc: func(_ context.Context, req *aiplatformpb.UploadModelRequest, _ ...gax.CallOption) (*aiplatform.UploadModelOperation, error) {
			capturedUploadRequest = req

			return nil, nil
		},
	})
	endpointClientFactory := MockEndpointClientFactory(&MockEndpointClient{
		DeployModelFunc: func(_ context.Context, req *aiplatformpb.DeployModelRequest, _ ...gax.CallOption) (*aiplatform.DeployModelOperation, error) {
			capturedDeployRequest = req

			return nil, nil
		},
	})

	provider := &VertexModelDeployment{}
	testFactoryRegistry.modelClientFactory = modelClientFactory
	testFactoryRegistry.endpointClientFactory = endpointClientFactory

	// Execute - we expect it to fail with nil operation error
	_, err := provider.Create(ctx, req)
	if err != nil {
		t.Fatalf("Expected nil error from mock, but got %v", err)
	}

	// Validate UploadModelRequest was captured and has correct parameters
	if capturedUploadRequest == nil {
		t.Fatal("UploadModelRequest was not captured")
	}

	// Assert parent format
	expectedParent := "projects/test-project/locations/us-central1"
	if capturedUploadRequest.Parent != expectedParent {
		t.Errorf("Expected Parent %s, got %s", expectedParent, capturedUploadRequest.Parent)
	}

	// Assert service account
	if capturedUploadRequest.ServiceAccount != serviceAccount {
		t.Errorf("Expected ServiceAccount %s, got %s", serviceAccount, capturedUploadRequest.ServiceAccount)
	}

	// Assert model properties
	model := capturedUploadRequest.Model
	if model == nil {
		t.Fatal("Model is nil in upload request")
	}

	if model.DisplayName != resourceName {
		t.Errorf("Expected DisplayName %s, got %s", resourceName, model.DisplayName)
	}

	// Assert container spec - ImageUri
	if model.ContainerSpec == nil {
		t.Fatal("ContainerSpec is nil")
	}

	if model.ContainerSpec.ImageUri != modelImageURL {
		t.Errorf("Expected ImageUri %s, got %s", modelImageURL, model.ContainerSpec.ImageUri)
	}

	// Assert artifact URI
	if model.ArtifactUri != modelArtifactsBucketURI {
		t.Errorf("Expected ArtifactUri %s, got %s", modelArtifactsBucketURI, model.ArtifactUri)
	}

	// Assert prediction schemas
	if model.PredictSchemata == nil {
		t.Fatal("PredictSchemata is nil")
	}
	if model.PredictSchemata.InstanceSchemaUri != modelPredictionInputSchemaURI {
		t.Errorf("Expected InstanceSchemaUri %s, got %s", modelPredictionInputSchemaURI, model.PredictSchemata.InstanceSchemaUri)
	}
	if model.PredictSchemata.PredictionSchemaUri != modelPredictionOutputSchemaURI {
		t.Errorf("Expected PredictionSchemaUri %s, got %s", modelPredictionOutputSchemaURI, model.PredictSchemata.PredictionSchemaUri)
	}
	if model.PredictSchemata.ParametersSchemaUri != modelPredictionBehaviorSchemaURI {
		t.Errorf("Expected ParametersSchemaUri %s, got %s", modelPredictionBehaviorSchemaURI, model.PredictSchemata.ParametersSchemaUri)
	}

	// Assert labels
	if len(model.Labels) != 2 {
		t.Errorf("Expected 2 labels, got %d", len(model.Labels))
	}
	if model.Labels["env"] != "test" {
		t.Errorf("Expected label env=test, got %s", model.Labels["env"])
	}
	if model.Labels["component"] != "ml" {
		t.Errorf("Expected label component=ml, got %s", model.Labels["component"])
	}

	// Assert endpoint format
	expectedEndpoint := "projects/test-project/locations/us-central1/endpoints/test-endpoint"
	if capturedDeployRequest.Endpoint != expectedEndpoint {
		t.Errorf("Expected Endpoint %s, got %s", expectedEndpoint, capturedDeployRequest.Endpoint)
	}

	// Assert deployed model properties
	validateDeployedModel(t, capturedDeployRequest.DeployedModel, resourceName, serviceAccount, machineType, int32(minReplicas), int32(maxReplicas))
}

// validateDeployedModel validates the deployed model properties to reduce nesting complexity
func validateDeployedModel(t *testing.T, deployedModel *aiplatformpb.DeployedModel, resourceName, serviceAccount, machineType string, minReplicas, maxReplicas int32) {
	t.Helper()

	if deployedModel == nil {
		return
	}

	// Assert model name (should come from upload result, but we can't validate exact format without full mock)
	if deployedModel.DisplayName != resourceName {
		t.Errorf("Expected DisplayName %s, got %s", resourceName, deployedModel.DisplayName)
	}

	// Assert service account in deployed model
	if deployedModel.ServiceAccount != serviceAccount {
		t.Errorf("Expected ServiceAccount %s, got %s", serviceAccount, deployedModel.ServiceAccount)
	}

	// Assert machine specs
	predResources := deployedModel.GetDedicatedResources()
	if predResources == nil {
		return
	}

	machineSpec := predResources.MachineSpec
	if machineSpec != nil {
		if machineSpec.MachineType != machineType {
			t.Errorf("Expected MachineType %s, got %s", machineType, machineSpec.MachineType)
		}
	}

	// Assert replica counts
	if predResources.MinReplicaCount != minReplicas {
		t.Errorf("Expected MinReplicaCount %d, got %d", minReplicas, predResources.MinReplicaCount)
	}

	if predResources.MaxReplicaCount != maxReplicas {
		t.Errorf("Expected MaxReplicaCount %d, got %d", maxReplicas, predResources.MaxReplicaCount)
	}
}

// MockModelClient implements the VertexModelClient interface for testing.
type MockModelClient struct {
	UploadModelFunc func(ctx context.Context, req *aiplatformpb.UploadModelRequest, opts ...gax.CallOption) (*aiplatform.UploadModelOperation, error)
	CloseFunc       func() error
}

func (m *MockModelClient) UploadModel(ctx context.Context, req *aiplatformpb.UploadModelRequest, opts ...gax.CallOption) (*aiplatform.UploadModelOperation, error) {
	if m.UploadModelFunc != nil {
		return m.UploadModelFunc(ctx, req, opts...)
	}

	return nil, nil
}

func (m *MockModelClient) Close() error {
	if m.CloseFunc != nil {
		return m.CloseFunc()
	}

	return nil
}

// MockEndpointClient implements the VertexEndpointClient interface for testing.
type MockEndpointClient struct {
	DeployModelFunc   func(ctx context.Context, req *aiplatformpb.DeployModelRequest, opts ...gax.CallOption) (*aiplatform.DeployModelOperation, error)
	UndeployModelFunc func(ctx context.Context, req *aiplatformpb.UndeployModelRequest, opts ...gax.CallOption) (*aiplatform.UndeployModelOperation, error)
	GetEndpointFunc   func(ctx context.Context, req *aiplatformpb.GetEndpointRequest, opts ...gax.CallOption) (*aiplatformpb.Endpoint, error)
	CloseFunc         func() error
}

func (m *MockEndpointClient) DeployModel(ctx context.Context, req *aiplatformpb.DeployModelRequest, opts ...gax.CallOption) (*aiplatform.DeployModelOperation, error) {
	if m.DeployModelFunc != nil {
		return m.DeployModelFunc(ctx, req, opts...)
	}

	return nil, nil
}

func (m *MockEndpointClient) UndeployModel(ctx context.Context, req *aiplatformpb.UndeployModelRequest, opts ...gax.CallOption) (*aiplatform.UndeployModelOperation, error) {
	if m.UndeployModelFunc != nil {
		return m.UndeployModelFunc(ctx, req, opts...)
	}

	return nil, nil
}

func (m *MockEndpointClient) GetEndpoint(ctx context.Context, req *aiplatformpb.GetEndpointRequest, opts ...gax.CallOption) (*aiplatformpb.Endpoint, error) {
	if m.GetEndpointFunc != nil {
		return m.GetEndpointFunc(ctx, req, opts...)
	}

	return nil, nil
}

func (m *MockEndpointClient) Close() error {
	if m.CloseFunc != nil {
		return m.CloseFunc()
	}

	return nil
}

// MockModelClientFactory creates a mock model client factory for testing.
func MockModelClientFactory(mockClient *MockModelClient) services.ModelClientFactory {
	return func(_ context.Context, _ string) (services.VertexModelClient, error) {
		return mockClient, nil
	}
}

// MockEndpointClientFactory creates a mock endpoint client factory for testing.
func MockEndpointClientFactory(mockClient *MockEndpointClient) services.EndpointClientFactory {
	return func(_ context.Context, _ string) (services.VertexEndpointClient, error) {
		return mockClient, nil
	}
}
