# pulumi-gcp-vertex-model-deployment

Pulumi custom provider for Vertex models.

Upload and deploy a model from a Docker image to a Vertex endpoint:
```go
_, err = vertexmodeldeployment.NewVertexModelDeployment(ctx, v.newResourceName("vertex-model-deployment", "", 64), &vertexmodeldeployment.VertexModelDeploymentArgs{
  ProjectId:      pulumi.String(v.Project),
  Region:         pulumi.String(v.Region),
  EndpointId:     endpoint.Name,
  MachineType:    v.MachineType,
  MinReplicas:    v.MinReplicaCount,
  MaxReplicas:    v.MaxReplicaCount,
  ServiceAccount: modelServiceAccount.Email,
  ModelImageUrl:  v.ModelImageURL,
}, pulumi.Parent(v))
if err != nil {
  return fmt.Errorf("failed to deploy model: %w", err)
}
```
