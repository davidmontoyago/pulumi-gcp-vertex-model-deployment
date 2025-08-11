package resources

import (
	"context"
	"fmt"
	"time"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"github.com/pulumi/pulumi-go-provider/infer"
)

type VertexModelDeployment struct{}

func (VertexModelDeployment) Annotate(a infer.Annotator) {
	a.Describe(&VertexModelDeployment{}, "Deploys a model to a Vertex AI endpoint")
}

type VertexModelDeploymentArgs struct {
	ProjectID      string            `pulumi:"projectId"`
	Region         string            `pulumi:"region"`
	EndpointID     string            `pulumi:"endpointId"`
	ModelImageURL  string            `pulumi:"modelImageUrl"`
	MachineType    string            `pulumi:"machineType,optional"`
	MinReplicas    int               `pulumi:"minReplicas,optional"`
	MaxReplicas    int               `pulumi:"maxReplicas,optional"`
	TrafficPercent int               `pulumi:"trafficPercent,optional"`
	ServiceAccount string            `pulumi:"serviceAccount,optional"`
	Labels         map[string]string `pulumi:"labels,optional"`
}

func (args *VertexModelDeploymentArgs) Annotate(a infer.Annotator) {
	a.Describe(&args.ProjectID, "Google Cloud Project ID")
	a.Describe(&args.Region, "Google Cloud region")
	a.Describe(&args.EndpointID, "Vertex AI Endpoint ID")
	a.Describe(&args.ModelImageURL, "Vertex AI Model Image URL")
	a.Describe(&args.MachineType, "Machine type for deployment")
	a.Describe(&args.MinReplicas, "Minimum number of replicas")
	a.Describe(&args.MaxReplicas, "Maximum number of replicas")
	a.Describe(&args.TrafficPercent, "Traffic percentage for this deployment")
	a.Describe(&args.ServiceAccount, "Service account for the deployment")

	// Set defaults
	a.SetDefault(&args.MachineType, "n1-standard-2")
	a.SetDefault(&args.MinReplicas, 1)
	a.SetDefault(&args.MaxReplicas, 3)
	a.SetDefault(&args.TrafficPercent, 100)
}

type VertexModelDeploymentState struct {
	VertexModelDeploymentArgs
	DeployedModelID string `pulumi:"deployedModelId"`
	EndpointName    string `pulumi:"endpointName"`
	CreateTime      string `pulumi:"createTime"`
}

func (state *VertexModelDeploymentState) Annotate(a infer.Annotator) {
	a.Describe(&state.DeployedModelID, "ID of the deployed model")
	a.Describe(&state.EndpointName, "Full name of the endpoint")
	a.Describe(&state.CreateTime, "Creation timestamp")
}

// Create implements the creation logic
func (VertexModelDeployment) Create(
	ctx context.Context,
	req infer.CreateRequest[VertexModelDeploymentArgs],
) (infer.CreateResponse[VertexModelDeploymentState], error) {

	state := VertexModelDeploymentState{
		VertexModelDeploymentArgs: req.Inputs,
	}

	if req.DryRun {
		return infer.CreateResponse[VertexModelDeploymentState]{
			ID: fmt.Sprintf("%s-%s", req.Inputs.EndpointID, req.Inputs.ModelImageURL),
		}, nil
	}

	// Vertex endpoint client with Application Default Credentials
	client, err := aiplatform.NewEndpointClient(ctx)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to create endpoint client: %w", err)
	}
	defer client.Close()

	// Vertex model client with Application Default Credentials
	modelClient, err := aiplatform.NewModelClient(ctx)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to create model client: %w", err)
	}
	defer modelClient.Close()

	modelUploadOp, err := modelClient.UploadModel(ctx, &aiplatformpb.UploadModelRequest{
		// TODO support non traditional models
		// Endpoint to which the model is attached can be regional or global
		Parent: fmt.Sprintf("projects/%s/locations/global", req.Inputs.ProjectID),
		// Parent: fmt.Sprintf("projects/%s/locations/%s", req.Inputs.Region),
		Model: &aiplatformpb.Model{
			DisplayName: req.Name,
			Description: "Uploaded model for " + req.Inputs.ModelImageURL,
			ContainerSpec: &aiplatformpb.ModelContainerSpec{
				ImageUri:     req.Inputs.ModelImageURL,
				HealthRoute:  "/v1/models",               // Standard TF Serving health check
				PredictRoute: "/v1/models/model:predict", // Standard TF Serving prediction route
				Env: []*aiplatformpb.EnvVar{
					{
						Name:  "MODEL_NAME",
						Value: "model",
					},
					{
						Name:  "MODEL_IMAGE_URL",
						Value: req.Inputs.ModelImageURL,
					},
				},
				// TODO: Add support for custom ports if needed
				// Ports: []*aiplatformpb.Port{
				// 	{
				// 		ContainerPort: 8501, // Standard TF Serving HTTP port
				// 	},
				// },
			},
			Labels: req.Inputs.Labels,
			// Optional: specify artifact URI if you have model files in GCS
			// ArtifactUri: "gs://your-bucket/model-artifacts/",
		},
	})
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to upload model again and again!!!!!!: %w", err)
	}

	// Wait for model upload to complete
	modelUploadResult, err := modelUploadOp.Wait(ctx)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to wait for model upload: %w", err)
	}

	// Build the deployment request
	deployedModel := &aiplatformpb.DeployedModel{
		// Expected format: "projects/%s/locations/%s/models/%s"
		Model:       modelUploadResult.GetModel(),
		DisplayName: req.Name,
		PredictionResources: &aiplatformpb.DeployedModel_DedicatedResources{
			DedicatedResources: &aiplatformpb.DedicatedResources{
				MachineSpec: &aiplatformpb.MachineSpec{
					MachineType: req.Inputs.MachineType,
				},
				MinReplicaCount: int32(req.Inputs.MinReplicas),
				MaxReplicaCount: int32(req.Inputs.MaxReplicas),
			},
		},
	}

	if req.Inputs.ServiceAccount != "" {
		deployedModel.ServiceAccount = req.Inputs.ServiceAccount
	}

	deployReq := &aiplatformpb.DeployModelRequest{
		Endpoint: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			req.Inputs.ProjectID, req.Inputs.Region, req.Inputs.EndpointID),
		DeployedModel: deployedModel,
		TrafficSplit:  map[string]int32{
			// TODO set for parallel model deployments
		},
	}

	// Execute the deployment
	op, err := client.DeployModel(ctx, deployReq)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to deploy model: %w", err)
	}

	// Wait for completion with timeout
	result, err := op.Wait(ctx)
	if err != nil {
		return infer.CreateResponse[VertexModelDeploymentState]{},
			fmt.Errorf("failed to wait for deployment: %w", err)
	}

	deployedModelID := result.GetDeployedModel().GetId()

	state.DeployedModelID = deployedModelID
	state.EndpointName = req.Inputs.EndpointID
	state.CreateTime = time.Now().Format(time.RFC3339)

	return infer.CreateResponse[VertexModelDeploymentState]{
		ID:     deployedModelID,
		Output: state,
	}, nil
}

// Delete implements the deletion logic
func (VertexModelDeployment) Delete(
	ctx context.Context,
	req infer.DeleteRequest[VertexModelDeploymentState],
) (infer.DeleteResponse, error) {

	// With Application Default Credentials
	client, err := aiplatform.NewEndpointClient(ctx)
	if err != nil {
		return infer.DeleteResponse{}, fmt.Errorf("failed to create endpoint client: %w", err)
	}
	defer client.Close()

	undeployReq := &aiplatformpb.UndeployModelRequest{
		Endpoint: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			req.State.ProjectID, req.State.Region, req.State.EndpointID),
		DeployedModelId: req.State.DeployedModelID,
	}

	op, err := client.UndeployModel(ctx, undeployReq)
	if err != nil {
		return infer.DeleteResponse{}, fmt.Errorf("failed to undeploy model: %w", err)
	}

	_, err = op.Wait(ctx)
	if err != nil {
		return infer.DeleteResponse{}, fmt.Errorf("failed to wait for undeployment: %w", err)
	}

	return infer.DeleteResponse{}, nil
}

// Update implements the update logic
func (VertexModelDeployment) Update(
	ctx context.Context,
	req infer.UpdateRequest[VertexModelDeploymentArgs, VertexModelDeploymentState],
) (infer.UpdateResponse[VertexModelDeploymentState], error) {

	// TODO For simplicity, we'll recreate on any change
	return infer.UpdateResponse[VertexModelDeploymentState]{
		Output: VertexModelDeploymentState{
			VertexModelDeploymentArgs: req.Inputs,
			DeployedModelID:           req.State.DeployedModelID,
			EndpointName:              req.State.EndpointName,
			CreateTime:                req.State.CreateTime,
		},
	}, nil
}

// Read implements the read logic for drift detection
func (VertexModelDeployment) Read(
	ctx context.Context,
	req infer.ReadRequest[VertexModelDeploymentArgs, VertexModelDeploymentState],
) (infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState], error) {

	// With Application Default Credentials
	client, err := aiplatform.NewEndpointClient(ctx)
	if err != nil {
		return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, err
	}
	defer client.Close()

	getReq := &aiplatformpb.GetEndpointRequest{
		Name: fmt.Sprintf("projects/%s/locations/%s/endpoints/%s",
			req.State.ProjectID, req.State.Region, req.State.EndpointID),
	}

	endpoint, err := client.GetEndpoint(ctx, getReq)
	if err != nil {
		return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, err
	}

	// Verify the deployed model still exists
	var found bool
	for _, dm := range endpoint.DeployedModels {
		if dm.Id == req.ID {
			found = true
			break
		}
	}

	if !found {
		// Model is no longer deployed
		return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{}, nil
	}

	return infer.ReadResponse[VertexModelDeploymentArgs, VertexModelDeploymentState]{
		ID:     req.ID,
		Inputs: req.Inputs,
		State:  req.State,
	}, nil
}
