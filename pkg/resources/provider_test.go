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

const (
	testProjectID                      = "test-project"
	testRegion                         = "us-central1"
	testEndpointID                     = "test-endpoint"
	testModelImageURL                  = "gcr.io/test-project/custom-model:latest"
	testModelArtifactsBucketURI        = "gs://test-bucket/model-artifacts/"
	testModelPredictionInputSchemaURI  = "gs://test-bucket/schemas/input_schema.json"
	testModelPredictionOutputSchemaURI = "gs://test-bucket/schemas/output_schema.json"
	testEndpointPath                   = "projects/test-project/locations/us-central1/endpoints/test-endpoint"
	testModelName                      = "projects/test-project/locations/us-central1/models/1234567890"
	testCreateTime                     = "2023-10-15T10:30:00Z"
	testEnv                            = "test"
	testPredictRoute                   = "/v1/models/custom:predict"
	testHealthRoute                    = "/v1/models/custom:health"
)

//nolint:paralleltest,tparallel // Cannot run in parallel due to shared testFactoryRegistry
func TestVertexModelDeploymentCreate_ModelUploadAndDeployRequests(t *testing.T) {
	ctx := context.Background()

	// Test inputs
	projectID := testProjectID
	region := testRegion
	endpointID := testEndpointID
	modelImageURL := testModelImageURL
	modelArtifactsBucketURI := testModelArtifactsBucketURI
	modelPredictionInputSchemaURI := testModelPredictionInputSchemaURI
	modelPredictionOutputSchemaURI := testModelPredictionOutputSchemaURI
	modelPredictionBehaviorSchemaURI := "gs://test-bucket/schemas/behavior_schema.json"
	machineType := "n1-standard-2"
	minReplicas := 1
	maxReplicas := 3
	trafficPercent := 100
	serviceAccount := "test-service-account@test-project.iam.gserviceaccount.com"
	resourceName := "test-model-deployment"
	containerArgs := []string{"--model-name", "custom-model", "--batch-size", "32"}
	containerEnvVars := map[string]string{"MODEL_ENV": "production", "LOG_LEVEL": "info"}
	containerPort := int32(9090)

	// Variables to capture request parameters
	var capturedUploadRequest *aiplatformpb.UploadModelRequest
	var capturedDeployRequest *aiplatformpb.DeployModelRequest

	req := infer.CreateRequest[VertexModelDeploymentArgs]{
		Name:   resourceName,
		DryRun: false, // Explicitly set to false to ensure real execution
		Inputs: VertexModelDeploymentArgs{
			ProjectID:                        projectID,
			Region:                           region,
			ModelImageURL:                    modelImageURL,
			ModelArtifactsBucketURI:          modelArtifactsBucketURI,
			ModelPredictionInputSchemaURI:    modelPredictionInputSchemaURI,
			ModelPredictionOutputSchemaURI:   modelPredictionOutputSchemaURI,
			ModelPredictionBehaviorSchemaURI: modelPredictionBehaviorSchemaURI,
			Args:                             containerArgs,
			EnvVars:                          containerEnvVars,
			Port:                             containerPort,
			EndpointModelDeployment: &EndpointModelDeploymentArgs{
				EndpointID:     endpointID,
				MachineType:    machineType,
				MinReplicas:    minReplicas,
				MaxReplicas:    maxReplicas,
				TrafficPercent: trafficPercent,
			},
			ServiceAccount: serviceAccount,
			Labels:         map[string]string{"env": testEnv, "component": "ml"},
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

	// Execute - we expect it to succeed despite nil operations
	result, err := provider.Create(ctx, req)
	if err != nil {
		t.Fatalf("Expected nil error from mock, but got %v", err)
	}

	// Assert resource ID format
	expectedID := "test-project-us-central1-test-model-deployment"
	if result.ID != expectedID {
		t.Errorf("Expected resource ID %s, got %s", expectedID, result.ID)
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

	// Assert container arguments
	if len(model.ContainerSpec.Args) != len(containerArgs) {
		t.Errorf("Expected %d container args, got %d", len(containerArgs), len(model.ContainerSpec.Args))
	} else {
		for i, expectedArg := range containerArgs {
			if model.ContainerSpec.Args[i] != expectedArg {
				t.Errorf("Expected container arg[%d] %s, got %s", i, expectedArg, model.ContainerSpec.Args[i])
			}
		}
	}

	// Assert environment variables
	if len(model.ContainerSpec.Env) != len(containerEnvVars) {
		t.Errorf("Expected %d environment variables, got %d", len(containerEnvVars), len(model.ContainerSpec.Env))
	} else {
		envMap := make(map[string]string)
		for _, env := range model.ContainerSpec.Env {
			envMap[env.Name] = env.Value
		}
		for expectedName, expectedValue := range containerEnvVars {
			if actualValue, exists := envMap[expectedName]; !exists {
				t.Errorf("Expected environment variable %s not found", expectedName)
			} else if actualValue != expectedValue {
				t.Errorf("Expected environment variable %s=%s, got %s", expectedName, expectedValue, actualValue)
			}
		}
	}

	// Assert container port
	if len(model.ContainerSpec.Ports) != 1 {
		t.Errorf("Expected 1 container port, got %d", len(model.ContainerSpec.Ports))
	} else if model.ContainerSpec.Ports[0].ContainerPort != containerPort {
		t.Errorf("Expected container port %d, got %d", containerPort, model.ContainerSpec.Ports[0].ContainerPort)
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
	if model.Labels["env"] != testEnv {
		t.Errorf("Expected label env=test, got %s", model.Labels["env"])
	}
	if model.Labels["component"] != "ml" {
		t.Errorf("Expected label component=ml, got %s", model.Labels["component"])
	}

	// Validate DeployModelRequest was captured and has correct parameters
	if capturedDeployRequest == nil {
		t.Fatal("DeployModelRequest was not captured - endpoint deployment may not have been triggered")
	}

	// Assert endpoint format
	expectedEndpoint := testEndpointPath
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

//nolint:paralleltest,tparallel // Cannot run in parallel due to shared testFactoryRegistry
func TestVertexModelDeploymentCreate_ModelUploadOnly(t *testing.T) {
	ctx := context.Background()

	// Test inputs for model upload without endpoint deployment
	projectID := testProjectID
	region := testRegion
	modelImageURL := testModelImageURL
	modelArtifactsBucketURI := testModelArtifactsBucketURI
	modelPredictionInputSchemaURI := testModelPredictionInputSchemaURI
	modelPredictionOutputSchemaURI := testModelPredictionOutputSchemaURI
	predictRoute := testPredictRoute
	healthRoute := testHealthRoute
	serviceAccount := "test-service-account@test-project.iam.gserviceaccount.com"
	resourceName := "test-model-upload-only"
	// Test default port behavior - not setting Port should default to 8080
	defaultPort := int32(8080)

	// Variables to capture request parameters
	var capturedUploadRequest *aiplatformpb.UploadModelRequest

	req := infer.CreateRequest[VertexModelDeploymentArgs]{
		Name:   resourceName,
		DryRun: false,
		Inputs: VertexModelDeploymentArgs{
			ProjectID:                      projectID,
			Region:                         region,
			ModelImageURL:                  modelImageURL,
			ModelArtifactsBucketURI:        modelArtifactsBucketURI,
			ModelPredictionInputSchemaURI:  modelPredictionInputSchemaURI,
			ModelPredictionOutputSchemaURI: modelPredictionOutputSchemaURI,
			PredictRoute:                   predictRoute,
			HealthRoute:                    healthRoute,
			// EndpointModelDeployment is not set - model upload only
			ServiceAccount: serviceAccount,
			Labels:         map[string]string{"env": testEnv, "mode": "batch"},
		},
	}

	modelClientFactory := MockModelClientFactory(&MockModelClient{
		UploadModelFunc: func(_ context.Context, req *aiplatformpb.UploadModelRequest, _ ...gax.CallOption) (*aiplatform.UploadModelOperation, error) {
			capturedUploadRequest = req

			return nil, nil
		},
	})
	// Provide endpoint client factory that should not be called
	endpointClientFactory := MockEndpointClientFactory(&MockEndpointClient{
		DeployModelFunc: func(_ context.Context, _ *aiplatformpb.DeployModelRequest, _ ...gax.CallOption) (*aiplatform.DeployModelOperation, error) {
			t.Error("DeployModel should not be called for model-only upload")

			return nil, nil
		},
	})

	provider := &VertexModelDeployment{}
	testFactoryRegistry.modelClientFactory = modelClientFactory
	testFactoryRegistry.endpointClientFactory = endpointClientFactory

	// Execute
	result, err := provider.Create(ctx, req)
	if err != nil {
		t.Fatalf("Expected nil error from mock, but got %v", err)
	}

	// Validate UploadModelRequest was captured
	if capturedUploadRequest == nil {
		t.Fatal("UploadModelRequest was not captured")
	}

	// Assert model properties
	model := capturedUploadRequest.Model
	if model == nil {
		t.Fatal("Model is nil in upload request")
	}

	// Assert container spec routes
	if model.ContainerSpec == nil {
		t.Fatal("ContainerSpec is nil")
	}

	if model.ContainerSpec.PredictRoute != predictRoute {
		t.Errorf("Expected PredictRoute %s, got %s", predictRoute, model.ContainerSpec.PredictRoute)
	}

	if model.ContainerSpec.HealthRoute != healthRoute {
		t.Errorf("Expected HealthRoute %s, got %s", healthRoute, model.ContainerSpec.HealthRoute)
	}

	// Assert default port behavior (should default to 8080 when not specified)
	if len(model.ContainerSpec.Ports) != 1 {
		t.Errorf("Expected 1 container port, got %d", len(model.ContainerSpec.Ports))
	} else if model.ContainerSpec.Ports[0].ContainerPort != defaultPort {
		t.Errorf("Expected default container port %d, got %d", defaultPort, model.ContainerSpec.Ports[0].ContainerPort)
	}

	// DeployModel should not have been called (verified by the mock error above)

	// Validate the state doesn't include endpoint-specific fields
	state := result.Output
	if state.DeployedModelID != "" {
		t.Errorf("Expected empty DeployedModelID for model-only upload, got %s", state.DeployedModelID)
	}
	if state.EndpointName != "" {
		t.Errorf("Expected empty EndpointName for model-only upload, got %s", state.EndpointName)
	}
	if state.ModelName == "" {
		t.Error("Expected ModelName to be set even for model-only upload")
	}
	if state.CreateTime == "" {
		t.Error("Expected CreateTime to be set")
	}
}

// MockModelClient implements the VertexModelClient interface for testing.
type MockModelClient struct {
	UploadModelFunc func(ctx context.Context, req *aiplatformpb.UploadModelRequest, opts ...gax.CallOption) (*aiplatform.UploadModelOperation, error)
	DeleteModelFunc func(ctx context.Context, req *aiplatformpb.DeleteModelRequest, opts ...gax.CallOption) (*aiplatform.DeleteModelOperation, error)
	GetModelFunc    func(ctx context.Context, req *aiplatformpb.GetModelRequest, opts ...gax.CallOption) (*aiplatformpb.Model, error)
	UpdateModelFunc func(ctx context.Context, req *aiplatformpb.UpdateModelRequest, opts ...gax.CallOption) (*aiplatformpb.Model, error)
	CloseFunc       func() error
}

func (m *MockModelClient) UploadModel(ctx context.Context, req *aiplatformpb.UploadModelRequest, opts ...gax.CallOption) (*aiplatform.UploadModelOperation, error) {
	if m.UploadModelFunc != nil {
		return m.UploadModelFunc(ctx, req, opts...)
	}

	return nil, nil
}

func (m *MockModelClient) DeleteModel(ctx context.Context, req *aiplatformpb.DeleteModelRequest, opts ...gax.CallOption) (*aiplatform.DeleteModelOperation, error) {
	if m.DeleteModelFunc != nil {
		return m.DeleteModelFunc(ctx, req, opts...)
	}

	return nil, nil
}

func (m *MockModelClient) GetModel(ctx context.Context, req *aiplatformpb.GetModelRequest, opts ...gax.CallOption) (*aiplatformpb.Model, error) {
	if m.GetModelFunc != nil {
		return m.GetModelFunc(ctx, req, opts...)
	}

	return nil, nil
}

func (m *MockModelClient) UpdateModel(ctx context.Context, req *aiplatformpb.UpdateModelRequest, opts ...gax.CallOption) (*aiplatformpb.Model, error) {
	if m.UpdateModelFunc != nil {
		return m.UpdateModelFunc(ctx, req, opts...)
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

//nolint:paralleltest,tparallel // Cannot run in parallel due to shared testFactoryRegistry
func TestVertexModelDeploymentDelete_ModelOnly(t *testing.T) {
	ctx := context.Background()

	// Test state for model-only deletion (no endpoint deployment)
	projectID := testProjectID
	region := testRegion
	modelName := testModelName
	createTime := testCreateTime

	// Variables to capture request parameters
	var capturedDeleteRequest *aiplatformpb.DeleteModelRequest

	state := VertexModelDeploymentState{
		VertexModelDeploymentArgs: VertexModelDeploymentArgs{
			ProjectID: projectID,
			Region:    region,
		},
		ModelName:       modelName,
		DeployedModelID: "", // Empty - no endpoint deployment
		EndpointName:    "", // Empty - no endpoint deployment
		CreateTime:      createTime,
	}

	req := infer.DeleteRequest[VertexModelDeploymentState]{
		ID:    "test-model-only",
		State: state,
	}

	modelClientFactory := MockModelClientFactory(&MockModelClient{
		DeleteModelFunc: func(_ context.Context, req *aiplatformpb.DeleteModelRequest, _ ...gax.CallOption) (*aiplatform.DeleteModelOperation, error) {
			capturedDeleteRequest = req

			return nil, nil // Simulate mocked operation
		},
	})

	// Endpoint client should not be called for model-only deletion
	endpointClientFactory := MockEndpointClientFactory(&MockEndpointClient{
		UndeployModelFunc: func(_ context.Context, _ *aiplatformpb.UndeployModelRequest, _ ...gax.CallOption) (*aiplatform.UndeployModelOperation, error) {
			t.Error("UndeployModel should not be called for model-only deletion")

			return nil, nil
		},
	})

	provider := &VertexModelDeployment{}
	testFactoryRegistry.modelClientFactory = modelClientFactory
	testFactoryRegistry.endpointClientFactory = endpointClientFactory

	// Execute deletion
	_, err := provider.Delete(ctx, req)
	if err != nil {
		t.Fatalf("Expected nil error from mock, but got %v", err)
	}

	// Validate DeleteModelRequest was captured and has correct parameters
	if capturedDeleteRequest == nil {
		t.Fatal("DeleteModelRequest was not captured")
	}

	// Assert model name format
	if capturedDeleteRequest.Name != modelName {
		t.Errorf("Expected Name %s, got %s", modelName, capturedDeleteRequest.Name)
	}
}

//nolint:paralleltest,tparallel // Cannot run in parallel due to shared testFactoryRegistry
func TestVertexModelDeploymentDelete_ModelWithEndpoint(t *testing.T) {
	ctx := context.Background()

	// Test state for model deletion with endpoint deployment
	projectID := testProjectID
	region := testRegion
	modelName := testModelName
	deployedModelID := "deployed-model-id-123"
	createTime := testCreateTime

	// Variables to capture request parameters
	var capturedUndeployRequest *aiplatformpb.UndeployModelRequest
	var capturedDeleteRequest *aiplatformpb.DeleteModelRequest

	state := VertexModelDeploymentState{
		VertexModelDeploymentArgs: VertexModelDeploymentArgs{
			ProjectID: projectID,
			Region:    region,
		},
		ModelName:       modelName,
		DeployedModelID: deployedModelID,  // Has endpoint deployment
		EndpointName:    testEndpointPath, // Has endpoint deployment - use full path
		CreateTime:      createTime,
	}

	req := infer.DeleteRequest[VertexModelDeploymentState]{
		ID:    "test-model-with-endpoint",
		State: state,
	}

	modelClientFactory := MockModelClientFactory(&MockModelClient{
		DeleteModelFunc: func(_ context.Context, req *aiplatformpb.DeleteModelRequest, _ ...gax.CallOption) (*aiplatform.DeleteModelOperation, error) {
			capturedDeleteRequest = req

			return nil, nil // Simulate mocked operation
		},
	})

	endpointClientFactory := MockEndpointClientFactory(&MockEndpointClient{
		UndeployModelFunc: func(_ context.Context, req *aiplatformpb.UndeployModelRequest, _ ...gax.CallOption) (*aiplatform.UndeployModelOperation, error) {
			capturedUndeployRequest = req

			return nil, nil // Simulate mocked operation
		},
	})

	provider := &VertexModelDeployment{}
	testFactoryRegistry.modelClientFactory = modelClientFactory
	testFactoryRegistry.endpointClientFactory = endpointClientFactory

	// Execute deletion
	_, err := provider.Delete(ctx, req)
	if err != nil {
		t.Fatalf("Expected nil error from mock, but got %v", err)
	}

	// Validate UndeployModelRequest was captured first and has correct parameters
	if capturedUndeployRequest == nil {
		t.Fatal("UndeployModelRequest was not captured")
	}

	// Assert endpoint format
	expectedEndpoint := testEndpointPath
	if capturedUndeployRequest.Endpoint != expectedEndpoint {
		t.Errorf("Expected Endpoint %s, got %s", expectedEndpoint, capturedUndeployRequest.Endpoint)
	}

	// Assert deployed model ID
	if capturedUndeployRequest.DeployedModelId != deployedModelID {
		t.Errorf("Expected DeployedModelId %s, got %s", deployedModelID, capturedUndeployRequest.DeployedModelId)
	}

	// Validate DeleteModelRequest was captured after undeployment and has correct parameters
	if capturedDeleteRequest == nil {
		t.Fatal("DeleteModelRequest was not captured")
	}

	// Assert model name format
	if capturedDeleteRequest.Name != modelName {
		t.Errorf("Expected Name %s, got %s", modelName, capturedDeleteRequest.Name)
	}
}

//nolint:paralleltest,tparallel // Cannot run in parallel due to shared testFactoryRegistry
func TestVertexModelDeploymentRead_ModelOnly(t *testing.T) {
	ctx := context.Background()

	// Test state for model-only read (no endpoint deployment)
	projectID := testProjectID
	region := testRegion
	modelName := testModelName
	modelImageURL := testModelImageURL
	modelArtifactsBucketURI := testModelArtifactsBucketURI
	modelPredictionInputSchemaURI := testModelPredictionInputSchemaURI
	modelPredictionOutputSchemaURI := testModelPredictionOutputSchemaURI
	predictRoute := testPredictRoute
	healthRoute := testHealthRoute
	createTime := testCreateTime

	// Create initial state
	state := VertexModelDeploymentState{
		VertexModelDeploymentArgs: VertexModelDeploymentArgs{
			ProjectID:                     projectID,
			Region:                        region,
			ModelImageURL:                 "old-image-url", // Different from mock to test update
			ModelArtifactsBucketURI:       "old-bucket-uri",
			ModelPredictionInputSchemaURI: "old-input-schema",
			PredictRoute:                  "old-predict-route",
			Labels:                        map[string]string{"old": "label"},
		},
		ModelName:       modelName,
		DeployedModelID: "", // Empty - no endpoint deployment
		EndpointName:    "", // Empty - no endpoint deployment
		CreateTime:      createTime,
	}

	// Mock model response
	mockModel := &aiplatformpb.Model{
		Name:        modelName,
		ArtifactUri: modelArtifactsBucketURI,
		ContainerSpec: &aiplatformpb.ModelContainerSpec{
			ImageUri:     modelImageURL,
			PredictRoute: predictRoute,
			HealthRoute:  healthRoute,
		},
		PredictSchemata: &aiplatformpb.PredictSchemata{
			InstanceSchemaUri:   modelPredictionInputSchemaURI,
			PredictionSchemaUri: modelPredictionOutputSchemaURI,
		},
		Labels: map[string]string{"env": testEnv, "component": "ml"},
	}

	req := infer.ReadRequest[VertexModelDeploymentArgs, VertexModelDeploymentState]{
		ID:     "test-model-only",
		Inputs: state.VertexModelDeploymentArgs,
		State:  state,
	}

	// Variables to capture request parameters
	var capturedGetModelRequest *aiplatformpb.GetModelRequest

	modelClientFactory := MockModelClientFactory(&MockModelClient{
		GetModelFunc: func(_ context.Context, req *aiplatformpb.GetModelRequest, _ ...gax.CallOption) (*aiplatformpb.Model, error) {
			capturedGetModelRequest = req

			return mockModel, nil
		},
	})

	// Endpoint client should not be called for model-only read
	endpointClientFactory := MockEndpointClientFactory(&MockEndpointClient{
		GetEndpointFunc: func(_ context.Context, _ *aiplatformpb.GetEndpointRequest, _ ...gax.CallOption) (*aiplatformpb.Endpoint, error) {
			t.Error("GetEndpoint should not be called for model-only read")

			return nil, nil
		},
	})

	provider := &VertexModelDeployment{}
	testFactoryRegistry.modelClientFactory = modelClientFactory
	testFactoryRegistry.endpointClientFactory = endpointClientFactory

	// Execute Read
	result, err := provider.Read(ctx, req)
	if err != nil {
		t.Fatalf("Expected nil error from Read, but got %v", err)
	}

	// Assert that the resource ID is preserved
	if result.ID != req.ID {
		t.Errorf("Expected resource ID %s, got %s", req.ID, result.ID)
	}

	// Validate GetModelRequest was captured and has correct parameters
	if capturedGetModelRequest == nil {
		t.Fatal("GetModelRequest was not captured")
	}

	if capturedGetModelRequest.Name != modelName {
		t.Errorf("Expected model name %s, got %s", modelName, capturedGetModelRequest.Name)
	}

	// Validate the returned state reflects the mock model data
	resultState := result.State
	if resultState.ModelName != modelName {
		t.Errorf("Expected ModelName %s, got %s", modelName, resultState.ModelName)
	}
	if resultState.ModelImageURL != modelImageURL {
		t.Errorf("Expected ModelImageURL %s, got %s", modelImageURL, resultState.ModelImageURL)
	}
	if resultState.ModelArtifactsBucketURI != modelArtifactsBucketURI {
		t.Errorf("Expected ModelArtifactsBucketURI %s, got %s", modelArtifactsBucketURI, resultState.ModelArtifactsBucketURI)
	}
	if resultState.ModelPredictionInputSchemaURI != modelPredictionInputSchemaURI {
		t.Errorf("Expected ModelPredictionInputSchemaURI %s, got %s", modelPredictionInputSchemaURI, resultState.ModelPredictionInputSchemaURI)
	}
	if resultState.ModelPredictionOutputSchemaURI != modelPredictionOutputSchemaURI {
		t.Errorf("Expected ModelPredictionOutputSchemaURI %s, got %s", modelPredictionOutputSchemaURI, resultState.ModelPredictionOutputSchemaURI)
	}
	if resultState.PredictRoute != predictRoute {
		t.Errorf("Expected PredictRoute %s, got %s", predictRoute, resultState.PredictRoute)
	}
	if resultState.HealthRoute != healthRoute {
		t.Errorf("Expected HealthRoute %s, got %s", healthRoute, resultState.HealthRoute)
	}

	// Validate labels are updated from model
	if len(resultState.Labels) != 2 {
		t.Errorf("Expected 2 labels, got %d", len(resultState.Labels))
	}
	if resultState.Labels["env"] != testEnv {
		t.Errorf("Expected label env=test, got %s", resultState.Labels["env"])
	}
	if resultState.Labels["component"] != "ml" {
		t.Errorf("Expected label component=ml, got %s", resultState.Labels["component"])
	}

	// Validate endpoint-related fields remain empty
	if resultState.DeployedModelID != "" {
		t.Errorf("Expected empty DeployedModelID for model-only read, got %s", resultState.DeployedModelID)
	}
	if resultState.EndpointName != "" {
		t.Errorf("Expected empty EndpointName for model-only read, got %s", resultState.EndpointName)
	}

	// Validate original inputs are preserved
	if result.Inputs.ProjectID != projectID {
		t.Errorf("Expected Inputs.ProjectID %s, got %s", projectID, result.Inputs.ProjectID)
	}
	if result.Inputs.Region != region {
		t.Errorf("Expected Inputs.Region %s, got %s", region, result.Inputs.Region)
	}
}

//nolint:paralleltest,tparallel // Cannot run in parallel due to shared testFactoryRegistry
func TestVertexModelDeploymentUpdate_ModelOnly(t *testing.T) {
	ctx := context.Background()

	// Test inputs for model update without endpoint deployment
	projectID := testProjectID
	region := testRegion
	modelName := testModelName
	createTime := testCreateTime
	resourceName := "test-model-update-only"

	// Original state values
	originalModelImageURL := "gcr.io/test-project/old-model:v1"
	originalModelArtifactsBucketURI := "gs://test-bucket/old-artifacts/"
	originalModelPredictionInputSchemaURI := "gs://test-bucket/schemas/old_input_schema.json"
	originalModelPredictionOutputSchemaURI := "gs://test-bucket/schemas/old_output_schema.json"
	originalModelPredictionBehaviorSchemaURI := "gs://test-bucket/schemas/old_behavior_schema.json"
	originalPredictRoute := "/v1/models/old:predict"
	originalHealthRoute := "/v1/models/old:health"
	originalLabels := map[string]string{"env": "dev", "version": "v1"}

	// Updated input values
	updatedModelImageURL := testModelImageURL
	updatedModelArtifactsBucketURI := testModelArtifactsBucketURI
	updatedModelPredictionInputSchemaURI := testModelPredictionInputSchemaURI
	updatedModelPredictionOutputSchemaURI := testModelPredictionOutputSchemaURI
	updatedModelPredictionBehaviorSchemaURI := "gs://test-bucket/schemas/updated_behavior_schema.json"
	updatedPredictRoute := testPredictRoute
	updatedHealthRoute := testHealthRoute
	updatedLabels := map[string]string{"env": testEnv, "component": "ml", "version": "v2"}
	updatedContainerArgs := []string{"--model-name", "updated-model", "--batch-size", "64"}
	updatedContainerEnvVars := map[string]string{"MODEL_ENV": "staging", "LOG_LEVEL": "debug", "CACHE_SIZE": "1024"}
	updatedContainerPort := int32(9091)

	// Variables to capture request parameters
	var capturedUpdateRequest *aiplatformpb.UpdateModelRequest

	// Create initial state
	initialState := VertexModelDeploymentState{
		VertexModelDeploymentArgs: VertexModelDeploymentArgs{
			ProjectID:                        projectID,
			Region:                           region,
			ModelImageURL:                    originalModelImageURL,
			ModelArtifactsBucketURI:          originalModelArtifactsBucketURI,
			ModelPredictionInputSchemaURI:    originalModelPredictionInputSchemaURI,
			ModelPredictionOutputSchemaURI:   originalModelPredictionOutputSchemaURI,
			ModelPredictionBehaviorSchemaURI: originalModelPredictionBehaviorSchemaURI,
			PredictRoute:                     originalPredictRoute,
			HealthRoute:                      originalHealthRoute,
			Labels:                           originalLabels,
			// EndpointModelDeployment is not set - model update only
		},
		ModelName:       modelName,
		DeployedModelID: "", // Empty - no endpoint deployment
		EndpointName:    "", // Empty - no endpoint deployment
		CreateTime:      createTime,
	}

	// Create update request with new inputs
	req := infer.UpdateRequest[VertexModelDeploymentArgs, VertexModelDeploymentState]{
		ID:     resourceName,
		DryRun: false,
		State:  initialState,
		Inputs: VertexModelDeploymentArgs{
			ProjectID:                        projectID,
			Region:                           region,
			ModelImageURL:                    updatedModelImageURL,
			ModelArtifactsBucketURI:          updatedModelArtifactsBucketURI,
			ModelPredictionInputSchemaURI:    updatedModelPredictionInputSchemaURI,
			ModelPredictionOutputSchemaURI:   updatedModelPredictionOutputSchemaURI,
			ModelPredictionBehaviorSchemaURI: updatedModelPredictionBehaviorSchemaURI,
			PredictRoute:                     updatedPredictRoute,
			HealthRoute:                      updatedHealthRoute,
			Args:                             updatedContainerArgs,
			EnvVars:                          updatedContainerEnvVars,
			Port:                             updatedContainerPort,
			Labels:                           updatedLabels,
			// EndpointModelDeployment is not set - model update only
		},
	}

	// Mock updated model response
	mockUpdatedModel := &aiplatformpb.Model{
		Name:        modelName,
		DisplayName: resourceName,
		Description: "Uploaded model for " + updatedModelImageURL,
		ArtifactUri: updatedModelArtifactsBucketURI,
		ContainerSpec: &aiplatformpb.ModelContainerSpec{
			ImageUri:     updatedModelImageURL,
			PredictRoute: updatedPredictRoute,
			HealthRoute:  updatedHealthRoute,
			Args:         updatedContainerArgs,
			Env: []*aiplatformpb.EnvVar{
				{Name: "MODEL_ENV", Value: "staging"},
				{Name: "LOG_LEVEL", Value: "debug"},
				{Name: "CACHE_SIZE", Value: "1024"},
			},
			Ports: []*aiplatformpb.Port{
				{ContainerPort: updatedContainerPort},
			},
		},
		PredictSchemata: &aiplatformpb.PredictSchemata{
			InstanceSchemaUri:   updatedModelPredictionInputSchemaURI,
			PredictionSchemaUri: updatedModelPredictionOutputSchemaURI,
			ParametersSchemaUri: updatedModelPredictionBehaviorSchemaURI,
		},
		Labels: updatedLabels,
	}

	modelClientFactory := MockModelClientFactory(&MockModelClient{
		UpdateModelFunc: func(_ context.Context, req *aiplatformpb.UpdateModelRequest, _ ...gax.CallOption) (*aiplatformpb.Model, error) {
			capturedUpdateRequest = req

			return mockUpdatedModel, nil
		},
	})

	// Endpoint client should not be called for model-only update
	endpointClientFactory := MockEndpointClientFactory(&MockEndpointClient{
		DeployModelFunc: func(_ context.Context, _ *aiplatformpb.DeployModelRequest, _ ...gax.CallOption) (*aiplatform.DeployModelOperation, error) {
			t.Error("DeployModel should not be called for model-only update")

			return nil, nil
		},
		UndeployModelFunc: func(_ context.Context, _ *aiplatformpb.UndeployModelRequest, _ ...gax.CallOption) (*aiplatform.UndeployModelOperation, error) {
			t.Error("UndeployModel should not be called for model-only update")

			return nil, nil
		},
	})

	provider := &VertexModelDeployment{}
	testFactoryRegistry.modelClientFactory = modelClientFactory
	testFactoryRegistry.endpointClientFactory = endpointClientFactory

	// Execute update
	result, err := provider.Update(ctx, req)
	if err != nil {
		t.Fatalf("Expected nil error from mock, but got %v", err)
	}

	// Validate UpdateModelRequest was captured and has correct parameters
	if capturedUpdateRequest == nil {
		t.Fatal("UpdateModelRequest was not captured")
	}

	// Assert model name
	if capturedUpdateRequest.Model.Name != modelName {
		t.Errorf("Expected Model.Name %s, got %s", modelName, capturedUpdateRequest.Model.Name)
	}

	// Assert display name uses resource ID
	if capturedUpdateRequest.Model.DisplayName != resourceName {
		t.Errorf("Expected Model.DisplayName %s, got %s", resourceName, capturedUpdateRequest.Model.DisplayName)
	}

	// Assert description is consistent with creation
	expectedDescription := "Uploaded model for " + updatedModelImageURL
	if capturedUpdateRequest.Model.Description != expectedDescription {
		t.Errorf("Expected Model.Description %s, got %s", expectedDescription, capturedUpdateRequest.Model.Description)
	}

	// Assert artifact URI
	if capturedUpdateRequest.Model.ArtifactUri != updatedModelArtifactsBucketURI {
		t.Errorf("Expected ArtifactUri %s, got %s", updatedModelArtifactsBucketURI, capturedUpdateRequest.Model.ArtifactUri)
	}

	// Assert container spec
	if capturedUpdateRequest.Model.ContainerSpec == nil {
		t.Fatal("ContainerSpec is nil")
	}
	if capturedUpdateRequest.Model.ContainerSpec.ImageUri != updatedModelImageURL {
		t.Errorf("Expected ImageUri %s, got %s", updatedModelImageURL, capturedUpdateRequest.Model.ContainerSpec.ImageUri)
	}
	if capturedUpdateRequest.Model.ContainerSpec.PredictRoute != updatedPredictRoute {
		t.Errorf("Expected PredictRoute %s, got %s", updatedPredictRoute, capturedUpdateRequest.Model.ContainerSpec.PredictRoute)
	}
	if capturedUpdateRequest.Model.ContainerSpec.HealthRoute != updatedHealthRoute {
		t.Errorf("Expected HealthRoute %s, got %s", updatedHealthRoute, capturedUpdateRequest.Model.ContainerSpec.HealthRoute)
	}

	// Assert updated container arguments
	if len(capturedUpdateRequest.Model.ContainerSpec.Args) != len(updatedContainerArgs) {
		t.Errorf("Expected %d container args, got %d", len(updatedContainerArgs), len(capturedUpdateRequest.Model.ContainerSpec.Args))
	} else {
		for i, expectedArg := range updatedContainerArgs {
			if capturedUpdateRequest.Model.ContainerSpec.Args[i] != expectedArg {
				t.Errorf("Expected container arg[%d] %s, got %s", i, expectedArg, capturedUpdateRequest.Model.ContainerSpec.Args[i])
			}
		}
	}

	// Assert updated environment variables
	if len(capturedUpdateRequest.Model.ContainerSpec.Env) != len(updatedContainerEnvVars) {
		t.Errorf("Expected %d environment variables, got %d", len(updatedContainerEnvVars), len(capturedUpdateRequest.Model.ContainerSpec.Env))
	} else {
		envMap := make(map[string]string)
		for _, env := range capturedUpdateRequest.Model.ContainerSpec.Env {
			envMap[env.Name] = env.Value
		}
		for expectedName, expectedValue := range updatedContainerEnvVars {
			if actualValue, exists := envMap[expectedName]; !exists {
				t.Errorf("Expected environment variable %s not found", expectedName)
			} else if actualValue != expectedValue {
				t.Errorf("Expected environment variable %s=%s, got %s", expectedName, expectedValue, actualValue)
			}
		}
	}

	// Assert updated container port
	if len(capturedUpdateRequest.Model.ContainerSpec.Ports) != 1 {
		t.Errorf("Expected 1 container port, got %d", len(capturedUpdateRequest.Model.ContainerSpec.Ports))
	} else if capturedUpdateRequest.Model.ContainerSpec.Ports[0].ContainerPort != updatedContainerPort {
		t.Errorf("Expected container port %d, got %d", updatedContainerPort, capturedUpdateRequest.Model.ContainerSpec.Ports[0].ContainerPort)
	}

	// Assert prediction schemas
	if capturedUpdateRequest.Model.PredictSchemata == nil {
		t.Fatal("PredictSchemata is nil")
	}
	if capturedUpdateRequest.Model.PredictSchemata.InstanceSchemaUri != updatedModelPredictionInputSchemaURI {
		t.Errorf("Expected InstanceSchemaUri %s, got %s", updatedModelPredictionInputSchemaURI, capturedUpdateRequest.Model.PredictSchemata.InstanceSchemaUri)
	}
	if capturedUpdateRequest.Model.PredictSchemata.PredictionSchemaUri != updatedModelPredictionOutputSchemaURI {
		t.Errorf("Expected PredictionSchemaUri %s, got %s", updatedModelPredictionOutputSchemaURI, capturedUpdateRequest.Model.PredictSchemata.PredictionSchemaUri)
	}
	if capturedUpdateRequest.Model.PredictSchemata.ParametersSchemaUri != updatedModelPredictionBehaviorSchemaURI {
		t.Errorf("Expected ParametersSchemaUri %s, got %s", updatedModelPredictionBehaviorSchemaURI, capturedUpdateRequest.Model.PredictSchemata.ParametersSchemaUri)
	}

	// Assert labels
	if len(capturedUpdateRequest.Model.Labels) != 3 {
		t.Errorf("Expected 3 labels, got %d", len(capturedUpdateRequest.Model.Labels))
	}
	if capturedUpdateRequest.Model.Labels["env"] != testEnv {
		t.Errorf("Expected label env=test, got %s", capturedUpdateRequest.Model.Labels["env"])
	}
	if capturedUpdateRequest.Model.Labels["component"] != "ml" {
		t.Errorf("Expected label component=ml, got %s", capturedUpdateRequest.Model.Labels["component"])
	}
	if capturedUpdateRequest.Model.Labels["version"] != "v2" {
		t.Errorf("Expected label version=v2, got %s", capturedUpdateRequest.Model.Labels["version"])
	}

	// Assert update mask contains expected fields
	if capturedUpdateRequest.UpdateMask == nil {
		t.Fatal("UpdateMask is nil")
	}
	expectedPaths := []string{"labels", "description", "container_spec", "artifact_uri", "predict_schemata"}
	if len(capturedUpdateRequest.UpdateMask.Paths) != len(expectedPaths) {
		t.Errorf("Expected %d update paths, got %d", len(expectedPaths), len(capturedUpdateRequest.UpdateMask.Paths))
	}
	// Verify all expected paths are present (order doesn't matter)
	pathsMap := make(map[string]bool)
	for _, path := range capturedUpdateRequest.UpdateMask.Paths {
		pathsMap[path] = true
	}
	for _, expectedPath := range expectedPaths {
		if !pathsMap[expectedPath] {
			t.Errorf("Expected update path %s not found in UpdateMask", expectedPath)
		}
	}

	// Validate the returned state reflects the mock response
	resultState := result.Output
	if resultState.ModelName != modelName {
		t.Errorf("Expected ModelName %s, got %s", modelName, resultState.ModelName)
	}
	if resultState.ModelImageURL != updatedModelImageURL {
		t.Errorf("Expected ModelImageURL %s, got %s", updatedModelImageURL, resultState.ModelImageURL)
	}
	if resultState.ModelArtifactsBucketURI != updatedModelArtifactsBucketURI {
		t.Errorf("Expected ModelArtifactsBucketURI %s, got %s", updatedModelArtifactsBucketURI, resultState.ModelArtifactsBucketURI)
	}
	if resultState.ModelPredictionInputSchemaURI != updatedModelPredictionInputSchemaURI {
		t.Errorf("Expected ModelPredictionInputSchemaURI %s, got %s", updatedModelPredictionInputSchemaURI, resultState.ModelPredictionInputSchemaURI)
	}
	if resultState.ModelPredictionOutputSchemaURI != updatedModelPredictionOutputSchemaURI {
		t.Errorf("Expected ModelPredictionOutputSchemaURI %s, got %s", updatedModelPredictionOutputSchemaURI, resultState.ModelPredictionOutputSchemaURI)
	}
	if resultState.ModelPredictionBehaviorSchemaURI != updatedModelPredictionBehaviorSchemaURI {
		t.Errorf("Expected ModelPredictionBehaviorSchemaURI %s, got %s", updatedModelPredictionBehaviorSchemaURI, resultState.ModelPredictionBehaviorSchemaURI)
	}
	if resultState.PredictRoute != updatedPredictRoute {
		t.Errorf("Expected PredictRoute %s, got %s", updatedPredictRoute, resultState.PredictRoute)
	}
	if resultState.HealthRoute != updatedHealthRoute {
		t.Errorf("Expected HealthRoute %s, got %s", updatedHealthRoute, resultState.HealthRoute)
	}

	// Validate labels are updated from mock response
	if len(resultState.Labels) != 3 {
		t.Errorf("Expected 3 labels in result state, got %d", len(resultState.Labels))
	}
	if resultState.Labels["env"] != testEnv {
		t.Errorf("Expected result state label env=test, got %s", resultState.Labels["env"])
	}
	if resultState.Labels["component"] != "ml" {
		t.Errorf("Expected result state label component=ml, got %s", resultState.Labels["component"])
	}
	if resultState.Labels["version"] != "v2" {
		t.Errorf("Expected result state label version=v2, got %s", resultState.Labels["version"])
	}

	// Validate endpoint-related fields remain empty
	if resultState.DeployedModelID != "" {
		t.Errorf("Expected empty DeployedModelID for model-only update, got %s", resultState.DeployedModelID)
	}
	if resultState.EndpointName != "" {
		t.Errorf("Expected empty EndpointName for model-only update, got %s", resultState.EndpointName)
	}

	// Validate original state fields are preserved
	if resultState.CreateTime != createTime {
		t.Errorf("Expected CreateTime %s, got %s", createTime, resultState.CreateTime)
	}
}

//nolint:paralleltest,tparallel // Cannot run in parallel due to shared testFactoryRegistry
func TestVertexModelDeploymentRead_ModelWithEndpoint(t *testing.T) {
	ctx := context.Background()

	// Test state for model with endpoint deployment read using SHORT endpoint name
	// This test verifies that when the state contains a short endpoint name,
	// the VertexEndpointModelGetter properly converts it to a fully qualified name for the API call
	projectID := testProjectID
	region := testRegion
	endpointID := testEndpointID
	fullEndpointName := testEndpointPath
	modelName := testModelName
	deployedModelID := "deployed-model-id-123"
	modelImageURL := testModelImageURL
	modelArtifactsBucketURI := testModelArtifactsBucketURI
	modelPredictionInputSchemaURI := testModelPredictionInputSchemaURI
	modelPredictionOutputSchemaURI := testModelPredictionOutputSchemaURI
	machineType := "n1-standard-4"
	minReplicas := int32(2)
	maxReplicas := int32(5)
	trafficPercent := int32(80)
	createTime := testCreateTime

	// Create initial state with endpoint deployment
	state := VertexModelDeploymentState{
		VertexModelDeploymentArgs: VertexModelDeploymentArgs{
			ProjectID:                     projectID,
			Region:                        region,
			ModelImageURL:                 "old-image-url", // Different from mock to test update
			ModelArtifactsBucketURI:       "old-bucket-uri",
			ModelPredictionInputSchemaURI: "old-input-schema",
			EndpointModelDeployment: &EndpointModelDeploymentArgs{
				EndpointID:     endpointID,
				MachineType:    "old-machine-type", // Different from mock to test update
				MinReplicas:    1,                  // Different from mock
				MaxReplicas:    3,                  // Different from mock
				TrafficPercent: 100,                // Different from mock
			},
		},
		ModelName:       modelName,
		DeployedModelID: deployedModelID,
		EndpointName:    endpointID, // Use SHORT endpoint name (just the ID) to test name conversion
		CreateTime:      createTime,
	}

	// Mock model response
	mockModel := &aiplatformpb.Model{
		Name:        modelName,
		ArtifactUri: modelArtifactsBucketURI,
		ContainerSpec: &aiplatformpb.ModelContainerSpec{
			ImageUri: modelImageURL,
		},
		PredictSchemata: &aiplatformpb.PredictSchemata{
			InstanceSchemaUri:   modelPredictionInputSchemaURI,
			PredictionSchemaUri: modelPredictionOutputSchemaURI,
		},
		Labels: map[string]string{"env": "production", "team": "ml"},
	}

	// Mock endpoint response with deployed model
	mockEndpoint := &aiplatformpb.Endpoint{
		Name: fullEndpointName,
		DeployedModels: []*aiplatformpb.DeployedModel{
			{
				Id: deployedModelID,
				PredictionResources: &aiplatformpb.DeployedModel_DedicatedResources{
					DedicatedResources: &aiplatformpb.DedicatedResources{
						MachineSpec: &aiplatformpb.MachineSpec{
							MachineType: machineType,
						},
						MinReplicaCount: minReplicas,
						MaxReplicaCount: maxReplicas,
					},
				},
			},
		},
		TrafficSplit: map[string]int32{
			deployedModelID: trafficPercent,
		},
	}

	req := infer.ReadRequest[VertexModelDeploymentArgs, VertexModelDeploymentState]{
		ID:     "test-model-with-endpoint",
		Inputs: state.VertexModelDeploymentArgs,
		State:  state,
	}

	// Variables to capture request parameters
	var capturedGetModelRequest *aiplatformpb.GetModelRequest
	var capturedGetEndpointRequest *aiplatformpb.GetEndpointRequest

	modelClientFactory := MockModelClientFactory(&MockModelClient{
		GetModelFunc: func(_ context.Context, req *aiplatformpb.GetModelRequest, _ ...gax.CallOption) (*aiplatformpb.Model, error) {
			capturedGetModelRequest = req

			return mockModel, nil
		},
	})

	endpointClientFactory := MockEndpointClientFactory(&MockEndpointClient{
		GetEndpointFunc: func(_ context.Context, req *aiplatformpb.GetEndpointRequest, _ ...gax.CallOption) (*aiplatformpb.Endpoint, error) {
			capturedGetEndpointRequest = req

			return mockEndpoint, nil
		},
	})

	provider := &VertexModelDeployment{}
	testFactoryRegistry.modelClientFactory = modelClientFactory
	testFactoryRegistry.endpointClientFactory = endpointClientFactory

	// Execute Read
	result, err := provider.Read(ctx, req)
	if err != nil {
		t.Fatalf("Expected nil error from Read, but got %v", err)
	}

	// Assert that the resource ID is preserved
	if result.ID != req.ID {
		t.Errorf("Expected resource ID %s, got %s", req.ID, result.ID)
	}

	// Validate GetModelRequest was captured and has correct parameters
	if capturedGetModelRequest == nil {
		t.Fatal("GetModelRequest was not captured")
	}
	if capturedGetModelRequest.Name != modelName {
		t.Errorf("Expected model name %s, got %s", modelName, capturedGetModelRequest.Name)
	}

	// Validate GetEndpointRequest was captured and has correct parameters
	if capturedGetEndpointRequest == nil {
		t.Fatal("GetEndpointRequest was not captured")
	}
	expectedEndpointPath := testEndpointPath
	if capturedGetEndpointRequest.Name != expectedEndpointPath {
		t.Errorf("Expected endpoint path %s, got %s", expectedEndpointPath, capturedGetEndpointRequest.Name)
	}

	// Validate the returned state reflects the mock model data
	resultState := result.State
	if resultState.ModelName != modelName {
		t.Errorf("Expected ModelName %s, got %s", modelName, resultState.ModelName)
	}
	if resultState.ModelImageURL != modelImageURL {
		t.Errorf("Expected ModelImageURL %s, got %s", modelImageURL, resultState.ModelImageURL)
	}
	if resultState.ModelArtifactsBucketURI != modelArtifactsBucketURI {
		t.Errorf("Expected ModelArtifactsBucketURI %s, got %s", modelArtifactsBucketURI, resultState.ModelArtifactsBucketURI)
	}
	if resultState.ModelPredictionInputSchemaURI != modelPredictionInputSchemaURI {
		t.Errorf("Expected ModelPredictionInputSchemaURI %s, got %s", modelPredictionInputSchemaURI, resultState.ModelPredictionInputSchemaURI)
	}
	if resultState.ModelPredictionOutputSchemaURI != modelPredictionOutputSchemaURI {
		t.Errorf("Expected ModelPredictionOutputSchemaURI %s, got %s", modelPredictionOutputSchemaURI, resultState.ModelPredictionOutputSchemaURI)
	}

	// Validate labels are updated from model
	if len(resultState.Labels) != 2 {
		t.Errorf("Expected 2 labels, got %d", len(resultState.Labels))
	}
	if resultState.Labels["env"] != "production" {
		t.Errorf("Expected label env=production, got %s", resultState.Labels["env"])
	}
	if resultState.Labels["team"] != "ml" {
		t.Errorf("Expected label team=ml, got %s", resultState.Labels["team"])
	}

	// Validate endpoint-related fields are updated from endpoint response
	if resultState.EndpointName != fullEndpointName {
		t.Errorf("Expected EndpointName %s, got %s", fullEndpointName, resultState.EndpointName)
	}
	if resultState.DeployedModelID != deployedModelID {
		t.Errorf("Expected DeployedModelID %s, got %s", deployedModelID, resultState.DeployedModelID)
	}

	// Validate endpoint deployment configuration is updated from live endpoint
	if resultState.EndpointModelDeployment == nil {
		t.Fatal("EndpointModelDeployment should not be nil")
	}
	if resultState.EndpointModelDeployment.MachineType != machineType {
		t.Errorf("Expected MachineType %s, got %s", machineType, resultState.EndpointModelDeployment.MachineType)
	}
	if resultState.EndpointModelDeployment.MinReplicas != int(minReplicas) {
		t.Errorf("Expected MinReplicas %d, got %d", minReplicas, resultState.EndpointModelDeployment.MinReplicas)
	}
	if resultState.EndpointModelDeployment.MaxReplicas != int(maxReplicas) {
		t.Errorf("Expected MaxReplicas %d, got %d", maxReplicas, resultState.EndpointModelDeployment.MaxReplicas)
	}
	if resultState.EndpointModelDeployment.TrafficPercent != int(trafficPercent) {
		t.Errorf("Expected TrafficPercent %d, got %d", trafficPercent, resultState.EndpointModelDeployment.TrafficPercent)
	}

	// Validate original inputs are preserved
	if result.Inputs.ProjectID != projectID {
		t.Errorf("Expected Inputs.ProjectID %s, got %s", projectID, result.Inputs.ProjectID)
	}
	if result.Inputs.Region != region {
		t.Errorf("Expected Inputs.Region %s, got %s", region, result.Inputs.Region)
	}
}

//nolint:paralleltest,tparallel // Cannot run in parallel due to shared testFactoryRegistry
func TestVertexModelDeploymentRead_ModelWithEndpointNameFullyQualified(t *testing.T) {
	ctx := context.Background()

	// Test state for model with endpoint deployment read using FULLY QUALIFIED endpoint name
	// This test verifies that when the state already contains a fully qualified endpoint name,
	// it is passed through correctly to the endpoint client
	projectID := testProjectID
	region := testRegion
	endpointID := testEndpointID
	fullEndpointName := testEndpointPath
	modelName := testModelName
	deployedModelID := "deployed-model-id-456"
	modelImageURL := testModelImageURL
	modelArtifactsBucketURI := testModelArtifactsBucketURI
	modelPredictionInputSchemaURI := testModelPredictionInputSchemaURI
	modelPredictionOutputSchemaURI := testModelPredictionOutputSchemaURI
	machineType := "n1-standard-8"
	minReplicas := int32(3)
	maxReplicas := int32(10)
	trafficPercent := int32(50)
	createTime := testCreateTime

	// Create initial state with endpoint deployment using FULLY QUALIFIED endpoint name
	state := VertexModelDeploymentState{
		VertexModelDeploymentArgs: VertexModelDeploymentArgs{
			ProjectID:                     projectID,
			Region:                        region,
			ModelImageURL:                 "old-image-url", // Different from mock to test update
			ModelArtifactsBucketURI:       "old-bucket-uri",
			ModelPredictionInputSchemaURI: "old-input-schema",
			EndpointModelDeployment: &EndpointModelDeploymentArgs{
				EndpointID:     endpointID,
				MachineType:    "old-machine-type", // Different from mock to test update
				MinReplicas:    1,                  // Different from mock
				MaxReplicas:    3,                  // Different from mock
				TrafficPercent: 100,                // Different from mock
			},
		},
		ModelName:       modelName,
		DeployedModelID: deployedModelID,
		EndpointName:    testEndpointPath, // Use FULLY QUALIFIED endpoint name - this is the key difference
		CreateTime:      createTime,
	}

	// Mock model response
	mockModel := &aiplatformpb.Model{
		Name:        modelName,
		ArtifactUri: modelArtifactsBucketURI,
		ContainerSpec: &aiplatformpb.ModelContainerSpec{
			ImageUri: modelImageURL,
		},
		PredictSchemata: &aiplatformpb.PredictSchemata{
			InstanceSchemaUri:   modelPredictionInputSchemaURI,
			PredictionSchemaUri: modelPredictionOutputSchemaURI,
		},
		Labels: map[string]string{"env": "staging", "team": "ai"},
	}

	// Mock endpoint response with deployed model
	mockEndpoint := &aiplatformpb.Endpoint{
		Name: fullEndpointName,
		DeployedModels: []*aiplatformpb.DeployedModel{
			{
				Id: deployedModelID,
				PredictionResources: &aiplatformpb.DeployedModel_DedicatedResources{
					DedicatedResources: &aiplatformpb.DedicatedResources{
						MachineSpec: &aiplatformpb.MachineSpec{
							MachineType: machineType,
						},
						MinReplicaCount: minReplicas,
						MaxReplicaCount: maxReplicas,
					},
				},
			},
		},
		TrafficSplit: map[string]int32{
			deployedModelID: trafficPercent,
		},
	}

	req := infer.ReadRequest[VertexModelDeploymentArgs, VertexModelDeploymentState]{
		ID:     "test-model-with-endpoint",
		Inputs: state.VertexModelDeploymentArgs,
		State:  state,
	}

	// Variables to capture request parameters
	var capturedGetModelRequest *aiplatformpb.GetModelRequest
	var capturedGetEndpointRequest *aiplatformpb.GetEndpointRequest

	modelClientFactory := MockModelClientFactory(&MockModelClient{
		GetModelFunc: func(_ context.Context, req *aiplatformpb.GetModelRequest, _ ...gax.CallOption) (*aiplatformpb.Model, error) {
			capturedGetModelRequest = req

			return mockModel, nil
		},
	})

	endpointClientFactory := MockEndpointClientFactory(&MockEndpointClient{
		GetEndpointFunc: func(_ context.Context, req *aiplatformpb.GetEndpointRequest, _ ...gax.CallOption) (*aiplatformpb.Endpoint, error) {
			capturedGetEndpointRequest = req

			return mockEndpoint, nil
		},
	})

	provider := &VertexModelDeployment{}
	testFactoryRegistry.modelClientFactory = modelClientFactory
	testFactoryRegistry.endpointClientFactory = endpointClientFactory

	// Execute Read
	result, err := provider.Read(ctx, req)
	if err != nil {
		t.Fatalf("Expected nil error from Read, but got %v", err)
	}

	// Assert that the resource ID is preserved
	if result.ID != req.ID {
		t.Errorf("Expected resource ID %s, got %s", req.ID, result.ID)
	}

	// Validate GetModelRequest was captured and has correct parameters
	if capturedGetModelRequest == nil {
		t.Fatal("GetModelRequest was not captured")
	}
	if capturedGetModelRequest.Name != modelName {
		t.Errorf("Expected model name %s, got %s", modelName, capturedGetModelRequest.Name)
	}

	// Validate GetEndpointRequest was captured and has correct parameters
	if capturedGetEndpointRequest == nil {
		t.Fatal("GetEndpointRequest was not captured")
	}
	expectedEndpointPath := testEndpointPath
	if capturedGetEndpointRequest.Name != expectedEndpointPath {
		t.Errorf("Expected endpoint path %s, got %s", expectedEndpointPath, capturedGetEndpointRequest.Name)
	}

	// Validate the returned state reflects the mock model data
	resultState := result.State
	if resultState.ModelName != modelName {
		t.Errorf("Expected ModelName %s, got %s", modelName, resultState.ModelName)
	}
	if resultState.ModelImageURL != modelImageURL {
		t.Errorf("Expected ModelImageURL %s, got %s", modelImageURL, resultState.ModelImageURL)
	}
	if resultState.ModelArtifactsBucketURI != modelArtifactsBucketURI {
		t.Errorf("Expected ModelArtifactsBucketURI %s, got %s", modelArtifactsBucketURI, resultState.ModelArtifactsBucketURI)
	}
	if resultState.ModelPredictionInputSchemaURI != modelPredictionInputSchemaURI {
		t.Errorf("Expected ModelPredictionInputSchemaURI %s, got %s", modelPredictionInputSchemaURI, resultState.ModelPredictionInputSchemaURI)
	}
	if resultState.ModelPredictionOutputSchemaURI != modelPredictionOutputSchemaURI {
		t.Errorf("Expected ModelPredictionOutputSchemaURI %s, got %s", modelPredictionOutputSchemaURI, resultState.ModelPredictionOutputSchemaURI)
	}

	// Validate labels are updated from model
	if len(resultState.Labels) != 2 {
		t.Errorf("Expected 2 labels, got %d", len(resultState.Labels))
	}
	if resultState.Labels["env"] != "staging" {
		t.Errorf("Expected label env=staging, got %s", resultState.Labels["env"])
	}
	if resultState.Labels["team"] != "ai" {
		t.Errorf("Expected label team=ai, got %s", resultState.Labels["team"])
	}

	// Validate endpoint-related fields are updated from endpoint response
	if resultState.EndpointName != fullEndpointName {
		t.Errorf("Expected EndpointName %s, got %s", fullEndpointName, resultState.EndpointName)
	}
	if resultState.DeployedModelID != deployedModelID {
		t.Errorf("Expected DeployedModelID %s, got %s", deployedModelID, resultState.DeployedModelID)
	}

	// Validate endpoint deployment configuration is updated from live endpoint
	if resultState.EndpointModelDeployment == nil {
		t.Fatal("EndpointModelDeployment should not be nil")
	}
	if resultState.EndpointModelDeployment.MachineType != machineType {
		t.Errorf("Expected MachineType %s, got %s", machineType, resultState.EndpointModelDeployment.MachineType)
	}
	if resultState.EndpointModelDeployment.MinReplicas != int(minReplicas) {
		t.Errorf("Expected MinReplicas %d, got %d", minReplicas, resultState.EndpointModelDeployment.MinReplicas)
	}
	if resultState.EndpointModelDeployment.MaxReplicas != int(maxReplicas) {
		t.Errorf("Expected MaxReplicas %d, got %d", maxReplicas, resultState.EndpointModelDeployment.MaxReplicas)
	}
	if resultState.EndpointModelDeployment.TrafficPercent != int(trafficPercent) {
		t.Errorf("Expected TrafficPercent %d, got %d", trafficPercent, resultState.EndpointModelDeployment.TrafficPercent)
	}

	// Validate original inputs are preserved
	if result.Inputs.ProjectID != projectID {
		t.Errorf("Expected Inputs.ProjectID %s, got %s", projectID, result.Inputs.ProjectID)
	}
	if result.Inputs.Region != region {
		t.Errorf("Expected Inputs.Region %s, got %s", region, result.Inputs.Region)
	}
}
