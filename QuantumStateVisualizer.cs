using UnityEngine;

public class QuantumStateVisualizer : MonoBehaviour
{
	public GameObject particlePrefab; // Assign a prefab in Unity editor

	public void VisualizeQuantumStates(QuantumStateCalculator.QuantumState[] states)
	{
		foreach (var state in states)
		{
			GameObject particle = Instantiate(particlePrefab, state.Position, Quaternion.identity);
			// Adjust the visualization based on the probability, size, color, etc.
			// Example: particle.transform.localScale = Vector3.one * state.Probability;
		}
	}
}
