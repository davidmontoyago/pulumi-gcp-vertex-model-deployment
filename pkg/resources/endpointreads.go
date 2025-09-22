package resources

import (
	"context"

	"github.com/davidmontoyago/pulumi-gcp-vertex-model-deployment/pkg/services"
	"github.com/pulumi/pulumi-go-provider/infer"
)

func readEndpointModel(ctx context.Context,
	endpointClient services.VertexEndpointClient,
	req infer.ReadRequest[VertexModelDeploymentArgs, VertexModelDeploymentState],
	state *VertexModelDeploymentState) error {

	endpointGetter := services.NewVertexEndpointModelGetter(endpointClient, req.State.ProjectID, req.State.Region)
	endpoint, foundDeployedModel, err := endpointGetter.Get(ctx, req.State.EndpointName, req.State.DeployedModelID)
	if err != nil {
		return err
	}

	if foundDeployedModel == nil {
		// Model is no longer deployed - return empty response to indicate resource doesn't exist
		return nil
	}

	// Update state with current endpoint and deployed model information
	state.EndpointName = endpoint.Name
	state.DeployedModelID = foundDeployedModel.Id

	// Update endpoint deployment configuration with current values if available
	if state.EndpointModelDeployment == nil {
		return nil
	}

	// Extract current deployment configuration from the deployed model
	if dedicatedResources := foundDeployedModel.GetDedicatedResources(); dedicatedResources != nil {
		if machineSpec := dedicatedResources.MachineSpec; machineSpec != nil {
			state.EndpointModelDeployment.MachineType = machineSpec.MachineType
		}
		state.EndpointModelDeployment.MinReplicas = int(dedicatedResources.MinReplicaCount)
		state.EndpointModelDeployment.MaxReplicas = int(dedicatedResources.MaxReplicaCount)
	}

	// Update traffic percentage from endpoint's traffic split if available
	if endpoint.TrafficSplit != nil {
		if trafficPercent, exists := endpoint.TrafficSplit[foundDeployedModel.Id]; exists {
			state.EndpointModelDeployment.TrafficPercent = int(trafficPercent)
		}
	}

	return nil
}
